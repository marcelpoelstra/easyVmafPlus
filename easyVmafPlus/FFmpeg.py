"""
MIT License

Copyright (c) 2020 Gabriel Davila - gdavila.revelo@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from . import config
from dataclasses import dataclass, field
from fractions import Fraction
from statistics import mean
from typing import Dict, List, Optional
import csv
import re
import subprocess
import json
import logging
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from ffmpeg_progress_yield import FfmpegProgress

logger = logging.getLogger(__name__)


# Model files: copies of model/vmaf_v1.0.16 and model/vmaf_v1.0.16_hfr of Netflix/vmaf v3.2.1,
# and of model/hdr_vmaf_v1.1.8 and model/hdr_vmaf_v1.1.8_hfr of Netflix/vmaf a6ed0f18, under
# the licence in models/LICENSE
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
VMAF_V1_VERSION = 'vmaf_v1.0.16'
HDR_VMAF_VERSION = 'hdr_vmaf_v1.1.8'
MODEL_STEMS = {'v1': VMAF_V1_VERSION, 'hdr': HDR_VMAF_VERSION}

# Per model generation and -model value, each model: (score name in the VMAF log,
# label in the results, source, extra model parameters, view). The source of a v0 model
# is its libvmaf built-in version, the source of a v1 or hdr model its model file variant.
# The view is the viewing distance of a v1 or hdr model in screen heights, None for v0.
VMAF_MODELS = {
    'v0': {
        'HD': [
            ('vmaf_hd', 'VMAF HD', 'vmaf_v0.6.1', {}, None),
            ('vmaf_hd_neg', 'VMAF Neg', 'vmaf_v0.6.1neg', {}, None),
            ('vmaf_hd_phone', 'VMAF Phone', 'vmaf_v0.6.1', {'enable_transform': 'true'}, None),
        ],
        '4K': [
            ('vmaf_4k', 'VMAF 4K', 'vmaf_4k_v0.6.1', {}, None),
        ],
    },
    'v1': {
        'HD': [
            ('vmaf_v1_hd', 'VMAF v1 HD', '3d0h', {}, '3h'),
            ('vmaf_v1_hd_phone', 'VMAF v1 Phone', '5d0h', {}, '5h'),
        ],
        '4K': [
            ('vmaf_v1_4k', 'VMAF v1 4K', '1d5h_2160', {}, '1.5h'),
            ('vmaf_v1_4k_3h', 'VMAF v1 4K 3H', '3d0h_2160', {}, '3h'),
        ],
    },
    'hdr': {
        'HD': [
            ('hdr_vmaf_v1_hd', 'HDR-VMAF v1 HD', '3d0h', {}, '3h'),
            ('hdr_vmaf_v1_hd_phone', 'HDR-VMAF v1 Phone', '5d0h', {}, '5h'),
        ],
        '4K': [
            ('hdr_vmaf_v1_4k', 'HDR-VMAF v1 4K', '1d5h_2160', {}, '1.5h'),
            ('hdr_vmaf_v1_4k_3h', 'HDR-VMAF v1 4K 3H', '3d0h_2160', {}, '3h'),
        ],
    },
}

VMAF_VERSIONS = ('1', '0.6')
VIEW_ALIASES = {'phone': '5h'}

# Per frame, libvmaf allocates two pictures, 50 MB each at 4K 4:4:4 16 bits, above glibc's
# mmap threshold; keeping them on the heap measured 10.5 to 19.8 fps per process upstream
# (upstream easyvmaf/ffmpeg.py:65-70)
_VMAF_GLIBC_TUNABLES = 'glibc.malloc.mmap_max=0:glibc.malloc.trim_threshold=1073741824'


def _vmaf_env():
    """
    Environment of the ffmpeg process that runs libvmaf, or None to inherit ours: the
    malloc tunables on Linux only, unless GLIBC_TUNABLES is already set.
    """
    if not sys.platform.startswith('linux') or 'GLIBC_TUNABLES' in os.environ:
        return None
    return {**os.environ, 'GLIBC_TUNABLES': _VMAF_GLIBC_TUNABLES}


def _cleanup_interrupted_process(process):
    """Stop and reap the ffmpeg process of an interrupted run"""
    if process is None:
        return
    try:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        logger.warning("Could not reap interrupted FFmpeg process: %s", error)
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


@dataclass
class ModelConfig:
    """
    One model of the libvmaf model= option: its score name in the VMAF log, its
    label in the results, its built-in version (v0) or model file path (v1, hdr),
    and its view for a v1 or hdr model.
    """
    name: str
    label: str
    version: Optional[str] = None
    path: Optional[str] = None
    params: Dict[str, str] = field(default_factory=dict)
    view: Optional[str] = None


def v1_model_path(variant, hfr=False, stem=VMAF_V1_VERSION):
    """Bundled model file, e.g. models/vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_3d0h.json"""
    stem = f'{stem}_hfr' if hfr else stem
    return os.path.join(MODELS_DIR, stem, f'{stem}_{variant}.json')


def _generation(version, dynamic_range):
    if version == '0.6':
        return 'v0'
    return 'hdr' if dynamic_range == 'hdr' else 'v1'


def _filter_views(entries, model, views):
    valid = [entry[4] for entry in entries]
    wanted = []
    for view in views:
        canonical = VIEW_ALIASES.get(str(view).lower(), str(view).lower())
        if canonical not in valid:
            raise ValueError(f"View {view!r} is not available for -model {model}. Supported: {', '.join(valid)}")
        wanted.append(canonical)
    if not wanted:
        raise ValueError(f"At least one view is required for -model {model}. Supported: {', '.join(valid)}")
    return [entry for entry in entries if entry[4] in wanted]


def select_models(model, vmaf_versions=('1',), views=None, hfr=False, dynamic_range='sdr'):
    """
    The models of a run for -model HD or 4K: per VMAF version in order, the v0.6.1 models
    for '0.6', the v1 models for '1', or the HDR-VMAF models for '1' with dynamic_range
    'hdr', in their HFR variants with hfr. views keeps the v1 or hdr models with those
    viewing distances ('3h', '5h' or its alias 'phone', '1.5h'); None keeps every model
    of the display.
    """
    versions = []
    for version in (str(v) for v in vmaf_versions):
        if version not in VMAF_VERSIONS:
            raise ValueError(f"Invalid VMAF version: {version!r}. Supported: {', '.join(VMAF_VERSIONS)}")
        if version not in versions:
            versions.append(version)
    if not versions:
        raise ValueError("At least one VMAF version is required")
    if dynamic_range not in ('sdr', 'hdr'):
        raise ValueError(f"Invalid dynamic range: {dynamic_range!r}. Supported: sdr, hdr")
    if dynamic_range == 'hdr' and '0.6' in versions:
        raise ValueError("VMAF 0.6 has no HDR models: HDR-VMAF is VMAF version 1 only")
    models: List[ModelConfig] = []
    for version in versions:
        generation = _generation(version, dynamic_range)
        if model not in VMAF_MODELS[generation]:
            raise ValueError(f"Invalid VMAF model: {model!r}. Supported: {', '.join(VMAF_MODELS[generation])}")
        entries = VMAF_MODELS[generation][model]
        if generation != 'v0' and views is not None:
            entries = _filter_views(entries, model, views)
        for name, label, source, params, view in entries:
            if generation == 'v0':
                models.append(ModelConfig(name, label, version=source, params=dict(params)))
            elif hfr:
                models.append(ModelConfig(f'{name}_hfr', f'{label} HFR',
                                          path=v1_model_path(source, hfr=True, stem=MODEL_STEMS[generation]),
                                          params=dict(params), view=view))
            else:
                models.append(ModelConfig(name, label,
                                          path=v1_model_path(source, stem=MODEL_STEMS[generation]),
                                          params=dict(params), view=view))
    return models


# A select expression of FFmpeg 9.0.2 takes at most 100 eq() terms (101 fail to parse):
# up to 99 frames per branch of the frames pass, plus frame 0
FRAMES_PER_SELECT = 99


def create_unique_file(path):
    """
    Create path with exclusive creation, so an existing file is never replaced. When the name
    exists, _2, _3 and so on goes before the extension. Returns the open binary file and its path.
    """
    stem, extension = os.path.splitext(path)
    number = 1
    while True:
        candidate = path if number == 1 else f'{stem}_{number}{extension}'
        try:
            return open(candidate, 'xb'), candidate
        except FileExistsError:
            number += 1


def create_unique_dir(path):
    """
    Create the folder path; when the name exists, _2, _3 and so on goes at its end.
    Returns the path created.
    """
    number = 1
    while True:
        candidate = path if number == 1 else f'{path}_{number}'
        try:
            os.mkdir(candidate)
            return candidate
        except FileExistsError:
            number += 1


def known_scores():
    """
    The VMAF score names of VMAF_MODELS in their order, each v1 or hdr name followed by its
    HFR name, as {score name: (label, score cap)}. The cap of a v1 or hdr model is the upper
    score_clip of its model file; the v0 built-in models score up to 100.
    """
    scores = {}
    for generation, model_sets in VMAF_MODELS.items():
        for entries in model_sets.values():
            for name, label, source, _, _ in entries:
                if generation == 'v0':
                    scores[name] = (label, 100.0)
                    continue
                for hfr in (False, True):
                    with open(v1_model_path(source, hfr=hfr, stem=MODEL_STEMS[generation])) as model_file:
                        cap = float(json.load(model_file)['model_dict']['score_clip'][1])
                    scores[f'{name}_hfr' if hfr else name] = (f'{label} HFR' if hfr else label, cap)
    return scores


@dataclass
class VmafLog:
    """
    The VMAF scores of a VMAF log: its path, the frame numbers, the per-frame scores per score
    name, and per score name the mean and the harmonic mean.
    """
    path: str
    frame_numbers: List[int]
    scores: Dict[str, List[float]]
    means: Dict[str, float]
    harmonic_means: Dict[str, float]


def read_vmaf_log(path):
    """
    The VMAF scores of a VMAF log in the format of its extension: json, xml or csv. The means come
    from the pooled metrics of a json or xml log. A csv log has none, so they are computed, the
    harmonic mean as libvmaf pools it: n / sum(1 / (x + 1)) - 1.
    """
    log_fmt = os.path.splitext(path)[1][1:]
    if log_fmt not in ('json', 'xml', 'csv'):
        raise ValueError(f"VMAF log {path}: unknown format {log_fmt!r}. Supported: json, xml, csv")
    try:
        if log_fmt == 'json':
            with open(path) as log_file:
                log = json.load(log_file)
            frames = [(int(frame['frameNum']), frame['metrics']) for frame in log['frames']]
            pooled = log.get('pooled_metrics', {})
        elif log_fmt == 'xml':
            root = ET.parse(path).getroot()
            frames = [(int(frame.attrib['frameNum']), frame.attrib) for frame in root.findall('frames/frame')]
            pooled = {metric.attrib['name']: metric.attrib for metric in root.findall('pooled_metrics/metric')}
        else:
            with open(path, newline='') as log_file:
                frames = [(int(row['Frame']), row) for row in csv.DictReader(log_file)]
            pooled = {}
        if not frames:
            raise ValueError(f"VMAF log {path} holds no frames")
        names = [name for name in known_scores() if name in frames[0][1]]
        if not names:
            raise ValueError(f"VMAF log {path} holds no VMAF score of easyVmafPlus")
        scores = {name: [float(metrics[name]) for _, metrics in frames] for name in names}
        means, harmonic_means = {}, {}
        for name, values in scores.items():
            if name in pooled:
                means[name] = float(pooled[name]['mean'])
                harmonic_means[name] = float(pooled[name]['harmonic_mean'])
            else:
                means[name] = mean(values)
                harmonic_means[name] = len(values) / sum(1 / (value + 1) for value in values) - 1
    except (KeyError, TypeError, ET.ParseError) as error:
        raise ValueError(f"VMAF log {path} cannot be read: {error!r}") from None
    return VmafLog(path, [number for number, _ in frames], scores, means, harmonic_means)


class FFprobe:
    '''
    Class to interact with FFprobe.
    It gets info about stream, frames and mpeg packets

    Inputs:
        - videoSrc: path to video
    Outputs:
        - getStreamInfo()
        - getFramesInfo()
        - getPacketsInfo()
        - getFormatInfo()
    '''
    _executable = os.environ.get('FFPROBE', config.ffprobe)

    def __init__(self, videoSrc, loglevel="info"):
        self.videoSrc = videoSrc
        self.loglevel = loglevel
        self.streamInfo = None
        self.framesInfo = None
        self.packetsInfo = None
        self.formatInfo = None
        self._cmd = None

    ''' private methods '''

    def _commitBase(self):
        ffprobe_loglevel = self.loglevel if self.loglevel == "verbose" else "quiet"
        return [FFprobe._executable, '-hide_banner', '-loglevel', ffprobe_loglevel,
                '-print_format', 'json']

    def _commitStreamSelection(self):
        return ['-select_streams', 'v']

    def _commitInput(self):
        return ['-i', self.videoSrc, '-read_intervals', '%+5']

    def _commit(self, opt):
        self._cmd = (self._commitBase() + [opt] +
                     self._commitStreamSelection() + self._commitInput())

    def _run(self):
        logger.debug("FFprobe cmd: %s", self._cmd)
        return json.loads(subprocess.check_output(self._cmd, shell=False))

    ''' public methods '''

    def getStreamInfo(self):
        self._commit('-show_streams')
        self.streamInfo = self._run()['streams'][0]
        return self.streamInfo

    def getFramesInfo(self):
        self._commit('-show_frames')
        self.framesInfo = self._run()['frames']
        return self.framesInfo

    def getPacketsInfo(self):
        self._commit('-show_packets')
        self.packetsInfo = self._run()['packets']
        return self.packetsInfo

    def getFormatInfo(self):
        self._commit('-show_format')
        self.formatInfo = self._run()['format']
        return self.formatInfo


class FFmpegQos:
    '''
    Class to interact with FFmpeg QoS Filters: PSNR and VMAF.
    Particullary, it interacts with libvmaf library through lavfi filter
    '''
    _executable = os.environ.get('FFMPEG', config.ffmpeg)

    def __init__(self,  main, ref, loglevel="info"):
        self.loglevel = loglevel
        self._cmd = None
        self.main = inputFFmpeg(main, input_id=0)
        self.ref = inputFFmpeg(ref, input_id=1)
        self.psnrFilter = []
        self.vmafFilter = []
        self.invertedSrc = False
        self.vmafpath = None
        self.vmaf_cambi_heatmap_path = None
        # Cap of the decoder and filter graph threads of ffmpeg: 1 for a sync pass, the
        # CPU share for a job, None keeps ffmpeg's own threading
        self._ffmpeg_threads = None
        # The VMAF process of getVmaf without progress, and whether cancel stopped it
        self._process = None
        self._cancelled = False

    @staticmethod
    def _escape_filter_value(value, option_levels=1):
        '''
        Escape a value for an FFmpeg filter option, e.g. libvmaf=log_path=VALUE.
        Level 1 escapes the filter option characters, level 2 the filtergraph
        characters: https://ffmpeg.org/ffmpeg-filters.html#Notes-on-filtergraph-escaping
        A value inside the libvmaf feature= option, e.g. heatmaps_path, passes one
        more option parser (av_dict_parse_string in vf_libvmaf.c): option_levels=2.
        '''
        for chars in ("\\':",) * option_levels + ("\\'[],;",):
            for c in chars:
                value = value.replace(c, '\\' + c)
        return value

    def _commitBase(self):
        base = [FFmpegQos._executable, '-y', '-hide_banner', '-stats', '-loglevel', self.loglevel]
        if self._ffmpeg_threads:
            # -lavfi is a complex filtergraph: -filter_threads applies to simple ones only
            base += ['-filter_complex_threads', str(self._ffmpeg_threads)]
        return base

    def _commit(self):
        """build the final cmd to run"""
        self._cmd = (self._commitBase() + self._commitInputs() +
                     self._commitFilters() + self._commitOutputs())

    def _commitInputs(self):
        """
        The input files, with hardware accelerated decoding for each. No -map: the -lavfi
        output is mapped on its own, and mapped unfiltered streams would decode both files
        to the end despite the trims. -an -sn -dn stop the selection of other streams.
        A seek keeps the original timestamps, so the filters see the timeline of a run
        from the start.
        """
        decoder = ['-threads', str(self._ffmpeg_threads)] if self._ffmpeg_threads else []
        timestamps = ['-copyts', '-start_at_zero'] if self.main.seekOptions or self.ref.seekOptions else []
        return (timestamps
                + decoder + ['-hwaccel', 'auto'] + self.main.extraOptions + ['-i', self.main.videoSrc]
                + decoder + ['-hwaccel', 'auto'] + self.ref.extraOptions + ['-i', self.ref.videoSrc]
                + ['-an', '-sn', '-dn'])

    def _commitOutputs(self):
        return ['-f', 'null', '-']

    def _commitFilters(self, filterName='lavfi'):
        """build the cmd for the filters"""
        filters = self.main.filtersList + self.ref.filtersList + self.psnrFilter + self.vmafFilter
        return [f'-{filterName}', ';'.join(filters)]

    @staticmethod
    def _build_model_string(models):
        """
        The libvmaf model= value: per model its built-in version or model file path,
        its name and extra parameters joined by escaped colons, the models joined by |.
        A model file path, like heatmaps_path, passes one more option parser
        (av_dict_parse_string in vf_libvmaf.c): option_levels=2.
        """
        entries = []
        for model in models:
            if model.version:
                source = f'version={model.version}'
            else:
                source = f'path={FFmpegQos._escape_filter_value(model.path, option_levels=2)}'
            tokens = [source, f'name={model.name}'] + [f'{k}={v}' for k, v in model.params.items()]
            entries.append('\\\\:'.join(tokens))
        return '|'.join(entries)

    def getPsnr(self):
        """
        It adds the PSNR filter to the lavfi chain, runs the ffmpeg cmd and
        returns the average PSNR from the ffmpeg output.
        """
        main = self.main.lastOutputID
        ref = self.ref.lastOutputID
        self.psnrFilter = [f'[{main}][{ref}]psnr']
        self._commit()

        logger.debug("FFmpeg PSNR cmd: %s", self._cmd)
        stdout = subprocess.check_output(
            self._cmd, stderr=subprocess.STDOUT, shell=False).decode('utf-8')
        averages = [s for s in stdout.split(" ") if "average" in s]
        if not averages:
            # The trims left no frame to compare, e.g. a start past the end
            raise RuntimeError("the PSNR filter compared no frames")
        return float(averages[0].split(":")[1])

    def getFirstFrameTimestamps(self):
        """
        Run the current chains until each one outputs its first frame. Returns
        ((pts, time_base), (pts, time_base)) of the first frame of the main and ref
        chains, in the time base of the end of each chain. Raises RuntimeError when
        ffmpeg fails or a chain outputs no frame.
        """
        firsts = {}
        filters = list(self.main.filtersList + self.ref.filtersList)
        outputs = []
        for name, stream in (('main', self.main), ('ref', self.ref)):
            # Without its checksum showinfo also accepts hardware frames
            filters.append(f'[{stream.lastOutputID}]showinfo@{name}=checksum=0[first_{name}]')
            outputs += ['-map', f'[first_{name}]', '-frames:v', '1', '-f', 'null', '-']
        # showinfo reports at the info level
        loglevel = self.loglevel if self.loglevel == 'verbose' else 'info'
        self._cmd = ([FFmpegQos._executable, '-y', '-hide_banner', '-loglevel', loglevel]
                     + self._commitInputs() + ['-lavfi', ';'.join(filters)] + outputs)
        logger.debug("FFmpeg first frame cmd: %s", self._cmd)
        result = subprocess.run(self._cmd, capture_output=True, text=True, shell=False)
        if result.returncode != 0:
            detail = (result.stderr.strip().splitlines() or ['no diagnostic output'])[-1]
            raise RuntimeError(f"FFmpeg first frame pass failed (exit code {result.returncode}): {detail}")
        for name in ('main', 'ref'):
            prefix = r'\[showinfo@%s @ [^\]]+\] ' % name
            time_base = re.search(prefix + r'config in time_base: (\d+)/(\d+)', result.stderr)
            frame = re.search(prefix + r'n:\s*0 pts:\s*(-?\d+)', result.stderr)
            if not (time_base and frame):
                raise RuntimeError(f"the {name} chain output no frame")
            firsts[name] = (int(frame.group(1)),
                            Fraction(int(time_base.group(1)), int(time_base.group(2))))
        return firsts['main'], firsts['ref']

    def getVmaf(self, log_path=None, models=None, subsample=1, output_fmt='json', threads=0, print_progress=False, end_sync=False, features=None, cambi_heatmap=False, heatmap_path=None):
        if models is None:
            # the default of the former model='HD': the v0.6.1 HD models
            models = select_models('HD', ('0.6',))
        log_fmt = output_fmt if output_fmt in ('xml', 'csv') else 'json'
        if log_path is None:
            log_path = os.path.splitext(self.main.videoSrc)[0] + f'_vmaf.{log_fmt}'
        if heatmap_path is None:
            heatmap_path = os.path.splitext(self.main.videoSrc)[0] + '_cambi_heatmap'
        if threads == 0:
            threads = os.cpu_count()
        shortest = 1 if end_sync else 0
        self.vmafpath = None
        self.vmaf_cambi_heatmap_path = None

        try:
            # No existing output is replaced: unique names are reserved, libvmaf writes into them
            log_file, self.vmafpath = create_unique_file(log_path)
            log_file.close()
            if features and cambi_heatmap:
                self.vmaf_cambi_heatmap_path = create_unique_dir(heatmap_path)

            params = (f'log_fmt={log_fmt}'
                      f':model={self._build_model_string(models)}'
                      f':n_subsample={subsample}'
                      f':log_path={self._escape_filter_value(self.vmafpath)}'
                      f':n_threads={threads}'
                      f':shortest={shortest}')
            if features:
                params += f':feature={features}'
                if cambi_heatmap:
                    params += f'\\\\:heatmaps_path={self._escape_filter_value(self.vmaf_cambi_heatmap_path, option_levels=2)}'
            main = self.main.lastOutputID
            ref = self.ref.lastOutputID
            self.vmafFilter = [f'[{main}][{ref}]libvmaf={params}']

            self._commit()
            logger.debug("FFmpeg VMAF cmd: %s", self._cmd)

            # The malloc tunables on Linux; elsewhere the process inherits the environment as before
            env = _vmaf_env()
            popen_kwargs = {'env': env} if env is not None else {}
            if print_progress:
                process = FfmpegProgress(self._cmd)
                child = None
                try:
                    # run_command_with_progress raises RuntimeError when ffmpeg exits with an error
                    for progress in process.run_command_with_progress(popen_kwargs=popen_kwargs or None):
                        child = getattr(process, 'process', None)
                        logger.info("progress = %s%% - %s", progress,
                                    "\n".join(str(process.stderr).splitlines()[-9:-8]))
                except (KeyboardInterrupt, SystemExit):
                    _cleanup_interrupted_process(child or getattr(process, 'process', None))
                    raise
            else:
                if self._cancelled:
                    raise RuntimeError("VMAF run cancelled")
                process = subprocess.Popen(self._cmd, stdout=subprocess.PIPE, shell=False, **popen_kwargs)
                self._process = process
                # cancel sets the flag before reading _process: one of the two sees the other
                if self._cancelled:
                    process.kill()
                try:
                    process.communicate()
                except (KeyboardInterrupt, SystemExit):
                    _cleanup_interrupted_process(process)
                    raise
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, self._cmd)
        except BaseException:
            self._removeEmptyOutputs()
            raise

        return process

    def cancel(self):
        """
        Stop the VMAF process of getVmaf (without progress) from another thread, now or as
        soon as it starts; getVmaf then raises.
        """
        self._cancelled = True
        process = self._process
        if process is not None and process.poll() is None:
            process.kill()

    def _removeEmptyOutputs(self):
        """The reserved VMAF log and heatmap folder of a failed or interrupted run go when still empty"""
        if self.vmafpath and os.path.isfile(self.vmafpath) and os.path.getsize(self.vmafpath) == 0:
            os.remove(self.vmafpath)
        if self.vmaf_cambi_heatmap_path and os.path.isdir(self.vmaf_cambi_heatmap_path) \
                and not os.listdir(self.vmaf_cambi_heatmap_path):
            os.rmdir(self.vmaf_cambi_heatmap_path)

    def getFrames(self, frame_numbers, folder):
        """
        Write the frames of the main chain with the given frame numbers as 8-bit RGB TIFF files into
        folder, and return (frame number, file, pts_time) per written frame, in frame order.
        A select expression takes at most 100 eq() terms (FFmpeg 9.0.2 rejects 101), so the chain
        splits into branches of up to 99 frames, each with its own select and output. Every branch
        also selects frame 0, because ffmpeg fails when an output receives no frame; those files
        are not returned. A missing frame lies beyond the end of the main chain. The image2 muxer
        reads each % of an output path as part of its pattern, so a % in folder is written as %%.
        """
        numbers = sorted(frame_numbers)
        chunks = [numbers[start:start + FRAMES_PER_SELECT] for start in range(0, len(numbers), FRAMES_PER_SELECT)]
        selections = [sorted({0, *chunk}) for chunk in chunks]
        graph = self.main.filtersList + [
            f'[{self.main.lastOutputID}]setpts=PTS-STARTPTS,split={len(chunks)}'
            + ''.join(f'[select{branch}]' for branch in range(len(chunks)))]
        outputs = []
        for branch, selection in enumerate(selections):
            select = '+'.join(f'eq(n\\,{number})' for number in selection)
            graph.append(f'[select{branch}]select={select},showinfo@frames{branch}[frames{branch}]')
            outputs += ['-map', f'[frames{branch}]', '-fps_mode', 'passthrough', '-c:v', 'tiff',
                        '-compression_algo', 'lzw', '-pix_fmt', 'rgb24',
                        os.path.join(folder.replace('%', '%%'), f'{branch:05d}_%06d.tif')]
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as graph_file:
            graph_file.write(';'.join(graph))
        self._cmd = [FFmpegQos._executable, '-y', '-hide_banner', '-nostats', '-loglevel', self.loglevel,
                     '-progress', 'pipe:2', '-stats_period', '10',
                     '-hwaccel', 'auto', '-i', self.main.videoSrc, '-/filter_complex', graph_file.name] + outputs
        logger.debug("FFmpeg frames cmd: %s", self._cmd)
        times = [[] for _ in chunks]
        try:
            process = subprocess.Popen(self._cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                                       encoding='utf-8', errors='replace', shell=False)
            for line in process.stderr:
                shown = re.match(r'\[showinfo@frames(\d+) @ [^\]]*\] .* pts_time:(\S+)', line)
                if shown:
                    times[int(shown.group(1))].append(float(shown.group(2)))
                elif line.startswith('[showinfo@frames'):
                    continue
                elif line.startswith('out_time='):
                    logger.info("Low frames pass: %s of the distorted input", line.strip().split('=', 1)[1])
                elif not re.match(r'[a-z0-9_]+=', line):
                    sys.stderr.write(line)
            process.wait()
        finally:
            os.remove(graph_file.name)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, self._cmd)
        written = []
        for branch, (chunk, selection) in enumerate(zip(chunks, selections)):
            for index, (number, pts_time) in enumerate(zip(selection, times[branch]), start=1):
                if number in chunk:
                    written.append((number, os.path.join(folder, f'{branch:05d}_{index:06d}.tif'), pts_time))
        return written

    def clearFilters(self):
        self.psnrFilter = []
        self.vmafFilter = []


class inputFFmpeg:
    '''
    Class to interact with FFmpeg inputs.
    It allows to manage Filter chains to each input. i.e., main and ref. Each
    Supported Methods:
    - setScaleFilter()
    - setOffsetFilter()
    - setDeintFrameFilter()
    - setDeintFieldFilter()
    - setTrimFilter()
    - setPreTrimFilter(), setEndTrimFilter(), setRangeTrimFilter()
    - setPtsShiftFilter()
    - setFpsFilter()
    - setFormatFilter()
    - setColorParamsFilter()
    - setSdrConversionFilter()
    - setSeek() (input option)
    - clearFilters()
    '''

    def __init__(self, videoSrc, input_id):
        self.name = f'input{input_id}_'
        self.id = input_id
        self.videoSrc = videoSrc
        self.filtersList = []
        self.seekOptions = []
        self.lastOutputID = f'{str(self.id)}:v'

    @property
    def extraOptions(self):
        """Input options, placed before -i"""
        return list(self.seekOptions)

    def _setFilter(self, filter):
        self.filtersList.append(filter)

    def _newInOutForFilter(self):
        self.n = len(self.filtersList)
        if self.n == 0:
            inputID = f'{str(self.id)}:v'
            outputID = f'{self.name}{str(self.n)}'
        else:
            inputID = f'{self.name}{str(self.n-1)}'
            outputID = f'{self.name}{str(self.n)}'
        return inputID, outputID

    def _updateOutputId(self, outputID):
        self.lastOutputID = outputID

    def setScaleFilter(self, width, height, algo='bicubic'):
        """Filter options for Upscale or Downscale"""
        inputID, outputID = self._newInOutForFilter()
        scaleFilter = f'[{inputID}]scale={width}:{height}:flags={algo}[{outputID}]'
        self._setFilter(scaleFilter)
        self._updateOutputId(outputID)

    def setOffsetFilter(self, offset):
        """set offset for videoSrc: time to wait before display frames"""
        inputID, outputID = self._newInOutForFilter()
        ptsFilter = f'[{inputID}]setpts=PTS+{offset}/TB[{outputID}]'
        self._setFilter(ptsFilter)
        self._updateOutputId(outputID)

    def setDeintFrameFilter(self):
        """
        Output one frame for each frame: 30i-> 30p
        """
        yadifOpt = '0:-1:0'
        inputID, outputID = self._newInOutForFilter()
        yadifFilter = f'[{inputID}]yadif={yadifOpt}[{outputID}]'
        self._setFilter(yadifFilter)
        self._updateOutputId(outputID)

    def setDeintFieldFilter(self):
        """
        Output one frame for each field: 30i ->  60p
        """
        yadifOpt = '1:-1:0'
        inputID, outputID = self._newInOutForFilter()
        yadifFilter = f'[{inputID}]yadif={yadifOpt}[{outputID}]'
        self._setFilter(yadifFilter)
        self._updateOutputId(outputID)

    def setTrimFilter(self, start, duration):
        inputID, outputID = self._newInOutForFilter()
        trimFilter = f'[{inputID}]trim=start={start}:duration={duration}, setpts=PTS-STARTPTS[{outputID}]'
        self._setFilter(trimFilter)
        self._updateOutputId(outputID)
        return

    def setPreTrimFilter(self, start):
        """
        Drop the frames before start, keeping their timestamps: the filters that follow see
        the original timeline, and a later setTrimFilter cuts at the same instant as without
        this filter.
        """
        inputID, outputID = self._newInOutForFilter()
        self._setFilter(f'[{inputID}]trim=start={start}[{outputID}]')
        self._updateOutputId(outputID)

    def setEndTrimFilter(self, end):
        """Drop the frames from end (seconds) on, keeping the timestamps"""
        inputID, outputID = self._newInOutForFilter()
        self._setFilter(f'[{inputID}]trim=end={end}[{outputID}]')
        self._updateOutputId(outputID)

    def setRangeTrimFilter(self, start, end=None):
        """Keep the frames in [start, end) seconds, to the end without end, then start them at 0"""
        inputID, outputID = self._newInOutForFilter()
        bounds = f'start={start}' + (f':end={end}' if end is not None else '')
        self._setFilter(f'[{inputID}]trim={bounds}, setpts=PTS-STARTPTS[{outputID}]')
        self._updateOutputId(outputID)

    def setPtsShiftFilter(self, pts):
        """Subtract pts, in time base units, from every timestamp"""
        inputID, outputID = self._newInOutForFilter()
        self._setFilter(f'[{inputID}]setpts=PTS-{int(pts)}[{outputID}]')
        self._updateOutputId(outputID)

    def setSeek(self, seconds):
        """
        Start reading the input at seconds (-ss before -i): ffmpeg decodes from the keyframe
        before it and drops the frames up to it. FFmpegQos then keeps the original timestamps
        (-copyts -start_at_zero).
        """
        self.seekOptions = ['-ss', f'{seconds:.6f}']

    def setFpsFilter(self, fps):
        inputID, outputID = self._newInOutForFilter()
        fpsFilter = f'[{inputID}]fps=fps={fps}[{outputID}]'
        self._setFilter(fpsFilter)
        self._updateOutputId(outputID)

    def setFormatFilter(self, pix_fmt, color_space=None, color_range=None):
        """
        Convert to pix_fmt and, when given, to the color_space matrix (e.g. bt2020nc) and
        the color_range (tv, pc) in the same conversion.
        """
        options = pix_fmt
        if color_space:
            options += f':color_spaces={color_space}'
        if color_range:
            options += f':color_ranges={color_range}'
        inputID, outputID = self._newInOutForFilter()
        formatFilter = f'[{inputID}]format={options}[{outputID}]'
        self._setFilter(formatFilter)
        self._updateOutputId(outputID)

    def setColorParamsFilter(self, color_range, primaries, trc, matrix):
        """Tag the frames with this colorimetry (FFmpeg names: tv, bt2020, smpte2084, bt2020nc) without converting them"""
        inputID, outputID = self._newInOutForFilter()
        self._setFilter(f'[{inputID}]setparams=range={color_range}:color_primaries={primaries}'
                        f':color_trc={trc}:colorspace={matrix}[{outputID}]')
        self._updateOutputId(outputID)

    def setSdrConversionFilter(self, width, height):
        """
        Convert tagged frames, e.g. PQ BT.2020, to BT.709 limited range at the given size with
        the colour options of the scale filter (FFmpeg 9.0), so the TIFF frames are viewable.
        """
        inputID, outputID = self._newInOutForFilter()
        self._setFilter(f'[{inputID}]scale={width}:{height}:out_primaries=bt709:out_transfer=bt709'
                        f':out_color_matrix=bt709:out_range=tv[{outputID}]')
        self._updateOutputId(outputID)

    def clearFilters(self):
        self.filtersList = []
        self.lastOutputID = f'{str(self.id)}:v'
        # The input options belong to the chain they were set for
        self.seekOptions = []


# PQ BT.2020 tags and 10 bit frames for the HDR-VMAF probe
HDR_PROBE_CHAIN = ('setparams=range=tv:color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc,'
                   'format=yuv420p10le')


def _probe_model_frame(model_path, chain=None):
    """
    True when libvmaf scores five 320x240 frames with the bundled model file, each input
    through chain first. The v1 models fail on 64x64 frames ("no feature
    'cambi_hrs_1080_cmxv_17_vlt_0.06' at index 0"); 320x240 frames pass.
    """
    source = 'nullsrc=s=320x240:r=25:d=0.2'
    model = FFmpegQos._escape_filter_value(model_path, option_levels=2)
    vmaf = f'libvmaf=model=path={model}:log_fmt=json:log_path={os.devnull}'
    if chain:
        graph = f'[0:v]{chain}[dis];[1:v]{chain}[ref];[dis][ref]{vmaf}'
    else:
        graph = vmaf
    probe_cmd = [
        FFmpegQos._executable,
        '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', source,
        '-f', 'lavfi', '-i', source,
        '-lavfi', graph,
        '-f', 'null', '-'
    ]
    return subprocess.run(probe_cmd, capture_output=True, text=True).returncode == 0


def probe_hdr_vmaf():
    """
    True when libvmaf scores PQ tagged frames with the bundled hdr_vmaf_v1.1.8_3d0h model:
    that needs a libvmaf built with zimg and an FFmpeg that passes the frame colorimetry.
    """
    return _probe_model_frame(v1_model_path('3d0h', stem=HDR_VMAF_VERSION), HDR_PROBE_CHAIN)


def check_ffmpeg():
    """
    Check the ffmpeg and ffprobe binaries before a run.

    Returns a dict with:
        'version_str':    'X.Y' from the ffmpeg version line, 'dev-build' when it has none
        'meets_minimum':  True for FFmpeg 9.0 and later, and for dev builds
        'libvmaf':        True when 'ffmpeg -filters' lists libvmaf
        'builtin_models': False when libvmaf cannot load the built-in model vmaf_v0.6.1
        'v1_models':      True when libvmaf runs the bundled VMAF v1 model vmaf_v1.0.16_3d0h
        'hdr_vmaf':       True when libvmaf runs the bundled HDR-VMAF model hdr_vmaf_v1.1.8_3d0h
                          on PQ tagged frames

    Raises:
        RuntimeError: when the ffmpeg or ffprobe binary is not found or cannot be run
    """
    if not FFmpegQos._executable:
        raise RuntimeError("ffmpeg not found on PATH. Install FFmpeg >= 9.0 built with "
                           "--enable-libvmaf, or point the FFMPEG environment variable to it.")
    if not FFprobe._executable:
        raise RuntimeError("ffprobe not found on PATH. Install FFmpeg >= 9.0, "
                           "or point the FFPROBE environment variable to it.")
    result = {
        'version_str': 'unknown',
        'meets_minimum': False,
        'libvmaf': False,
        'builtin_models': False,
        'v1_models': False,
        'hdr_vmaf': False,
    }

    try:
        version_output = subprocess.run([FFmpegQos._executable, '-version'],
                                        capture_output=True, text=True).stdout
        subprocess.run([FFprobe._executable, '-version'], capture_output=True, text=True)
    except OSError as e:
        raise RuntimeError(f"cannot run '{e.filename}': {e.strerror}") from None

    # Release builds print "ffmpeg version 7.1" or "ffmpeg version 7.1.1",
    # dev builds "ffmpeg version N-111825-gabcdef123" or "ffmpeg version git-88c46fce27e3"
    match = re.search(r'ffmpeg version (\d+)\.(\d+)', version_output)
    if match:
        major, minor = int(match.group(1)), int(match.group(2))
        result['version_str'] = f'{major}.{minor}'
        result['meets_minimum'] = (major, minor) >= (9, 0)
    else:
        result['version_str'] = 'dev-build'
        result['meets_minimum'] = True

    filters_output = subprocess.run([FFmpegQos._executable, '-hide_banner', '-filters'],
                                    capture_output=True, text=True).stdout
    result['libvmaf'] = ' libvmaf ' in filters_output

    # libvmaf reports "could not load libvmaf model" when the built-in model is
    # missing; other errors of this probe run do not concern the models
    probe_cmd = [
        FFmpegQos._executable,
        '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', 'nullsrc=s=64x64:r=1:d=0.1',
        '-f', 'lavfi', '-i', 'nullsrc=s=64x64:r=1:d=0.1',
        '-lavfi', f'libvmaf=model=version=vmaf_v0.6.1:log_fmt=json:log_path={os.devnull}',
        '-f', 'null', '-'
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    result['builtin_models'] = 'could not load libvmaf model' not in probe.stderr + probe.stdout

    result['v1_models'] = _probe_model_frame(v1_model_path('3d0h'))
    result['hdr_vmaf'] = probe_hdr_vmaf()
    return result
