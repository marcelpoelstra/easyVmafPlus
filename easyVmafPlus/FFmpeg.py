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


# VMAF v1 model files: copies of model/vmaf_v1.0.16 and model/vmaf_v1.0.16_hfr
# of Netflix/vmaf v3.2.1, under the licence in models/LICENSE
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
VMAF_V1_VERSION = 'vmaf_v1.0.16'

# Per model generation and -model value, each model: (score name in the VMAF log,
# label in the results, source, extra model parameters). The source of a v0 model
# is its libvmaf built-in version, the source of a v1 model its model file variant.
VMAF_MODELS = {
    'v0': {
        'HD': [
            ('vmaf_hd', 'VMAF HD', 'vmaf_v0.6.1', {}),
            ('vmaf_hd_neg', 'VMAF Neg', 'vmaf_v0.6.1neg', {}),
            ('vmaf_hd_phone', 'VMAF Phone', 'vmaf_v0.6.1', {'enable_transform': 'true'}),
        ],
        '4K': [
            ('vmaf_4k', 'VMAF 4K', 'vmaf_4k_v0.6.1', {}),
        ],
    },
    'v1': {
        'HD': [
            ('vmaf_v1_hd', 'VMAF v1 HD', '3d0h', {}),
            ('vmaf_v1_hd_phone', 'VMAF v1 Phone', '5d0h', {}),
        ],
        '4K': [
            ('vmaf_v1_4k', 'VMAF v1 4K', '1d5h_2160', {}),
            ('vmaf_v1_4k_3h', 'VMAF v1 4K 3H', '3d0h_2160', {}),
        ],
    },
}


@dataclass
class ModelConfig:
    """
    One model of the libvmaf model= option: its score name in the VMAF log, its
    label in the results, and its built-in version (v0) or model file path (v1).
    """
    name: str
    label: str
    version: Optional[str] = None
    path: Optional[str] = None
    params: Dict[str, str] = field(default_factory=dict)


def v1_model_path(variant, hfr=False):
    """Bundled VMAF v1 model file, e.g. models/vmaf_v1.0.16_hfr/vmaf_v1.0.16_hfr_3d0h.json"""
    stem = f'{VMAF_V1_VERSION}_hfr' if hfr else VMAF_V1_VERSION
    return os.path.join(MODELS_DIR, stem, f'{stem}_{variant}.json')


def select_models(model, vmaf_v0=False, hfr=False):
    """
    The models of a run for -model HD or 4K: the v0.6.1 models with vmaf_v0,
    otherwise the v1 models, in their HFR variants with hfr.
    """
    generation = 'v0' if vmaf_v0 else 'v1'
    if model not in VMAF_MODELS[generation]:
        raise ValueError(f"Invalid VMAF model: {model!r}. Supported: {', '.join(VMAF_MODELS[generation])}")
    models: List[ModelConfig] = []
    for name, label, source, params in VMAF_MODELS[generation][model]:
        if vmaf_v0:
            models.append(ModelConfig(name, label, version=source, params=dict(params)))
        elif hfr:
            models.append(ModelConfig(f'{name}_hfr', f'{label} HFR',
                                      path=v1_model_path(source, hfr=True), params=dict(params)))
        else:
            models.append(ModelConfig(name, label, path=v1_model_path(source), params=dict(params)))
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
    The VMAF score names of VMAF_MODELS in their order, each v1 name followed by its HFR name, as
    {score name: (label, score cap)}. The cap of a v1 model is the upper score_clip of its model
    file; the v0 built-in models score up to 100.
    """
    scores = {}
    for generation, model_sets in VMAF_MODELS.items():
        for entries in model_sets.values():
            for name, label, source, _ in entries:
                if generation == 'v0':
                    scores[name] = (label, 100.0)
                    continue
                for hfr in (False, True):
                    with open(v1_model_path(source, hfr=hfr)) as model_file:
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
        return [FFmpegQos._executable, '-y', '-hide_banner', '-stats', '-loglevel', self.loglevel]

    def _commit(self):
        """build the final cmd to run"""
        self._cmd = (self._commitBase() + self._commitInputs() +
                     self._commitFilters() + self._commitOutputs())

    def _commitInputs(self):
        """build the cmd for the inputs files, with hardware accelerated decoding for each input"""
        return ['-hwaccel', 'auto', '-i', self.main.videoSrc,
                '-hwaccel', 'auto', '-i', self.ref.videoSrc,
                '-map', '0:v', '-map', '1:v']

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
        stdout = stdout.split(" ")
        psnr = [s for s in stdout if "average" in s][0].split(":")[1]
        return float(psnr)

    def getVmaf(self, log_path=None, models=None, subsample=1, output_fmt='json', threads=0, print_progress=False, end_sync=False, features=None, cambi_heatmap=False):
        if models is None:
            # the default of the former model='HD': the v0.6.1 HD models
            models = select_models('HD', vmaf_v0=True)
        log_fmt = output_fmt if output_fmt in ('xml', 'csv') else 'json'
        if log_path is None:
            log_path = os.path.splitext(self.main.videoSrc)[0] + f'_vmaf.{log_fmt}'
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
                self.vmaf_cambi_heatmap_path = create_unique_dir(
                    os.path.splitext(self.main.videoSrc)[0] + '_cambi_heatmap')

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

            if print_progress:
                process = FfmpegProgress(self._cmd)
                # run_command_with_progress raises RuntimeError when ffmpeg exits with an error
                for progress in process.run_command_with_progress():
                    logger.info("progress = %s%% - %s", progress,
                                "\n".join(str(process.stderr).splitlines()[-9:-8]))
            else:
                process = subprocess.Popen(self._cmd, stdout=subprocess.PIPE, shell=False)
                process.communicate()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, self._cmd)
        except BaseException:
            self._removeEmptyOutputs()
            raise

        return process

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
    - setFpsFilter()
    - setFormatFilter()
    - clearFilters()
    '''

    def __init__(self, videoSrc, input_id):
        self.name = f'input{input_id}_'
        self.id = input_id
        self.videoSrc = videoSrc
        self.filtersList = []
        self.extraOptions = []
        self.lastOutputID = f'{str(self.id)}:v'

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

    def setFpsFilter(self, fps):
        inputID, outputID = self._newInOutForFilter()
        fpsFilter = f'[{inputID}]fps=fps={fps}[{outputID}]'
        self._setFilter(fpsFilter)
        self._updateOutputId(outputID)

    def setFormatFilter(self, pix_fmt):
        inputID, outputID = self._newInOutForFilter()
        formatFilter = f'[{inputID}]format={pix_fmt}[{outputID}]'
        self._setFilter(formatFilter)
        self._updateOutputId(outputID)

    def clearFilters(self):
        self.filtersList = []
        self.lastOutputID = f'{str(self.id)}:v'


def check_ffmpeg():
    """
    Check the ffmpeg and ffprobe binaries before a run.

    Returns a dict with:
        'version_str':    'X.Y' from the ffmpeg version line, 'dev-build' when it has none
        'meets_minimum':  True for FFmpeg 9.0 and later, and for dev builds
        'libvmaf':        True when 'ffmpeg -filters' lists libvmaf
        'builtin_models': False when libvmaf cannot load the built-in model vmaf_v0.6.1
        'v1_models':      True when libvmaf runs the bundled VMAF v1 model vmaf_v1.0.16_3d0h

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
    }

    try:
        version_output = subprocess.run([FFmpegQos._executable, '-version'],
                                        capture_output=True, text=True).stdout
        subprocess.run([FFprobe._executable, '-version'], capture_output=True, text=True)
    except OSError as e:
        raise RuntimeError(f"cannot run '{e.filename}': {e.strerror}") from None

    # Release builds print "ffmpeg version 7.1" or "ffmpeg version 7.1.1",
    # dev builds "ffmpeg version N-111825-gabcdef123"
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

    # The v1 models fail on the 64x64 frames of the probe above, with "no feature
    # 'cambi_hrs_1080_cmxv_17_vlt_0.06' at index 0"; 320x240 frames pass
    v1_model = FFmpegQos._escape_filter_value(v1_model_path('3d0h'), option_levels=2)
    v1_probe_cmd = [
        FFmpegQos._executable,
        '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', 'nullsrc=s=320x240:r=25:d=0.2',
        '-f', 'lavfi', '-i', 'nullsrc=s=320x240:r=25:d=0.2',
        '-lavfi', f'libvmaf=model=path={v1_model}:log_fmt=json:log_path={os.devnull}',
        '-f', 'null', '-'
    ]
    result['v1_models'] = subprocess.run(v1_probe_cmd, capture_output=True, text=True).returncode == 0
    return result
