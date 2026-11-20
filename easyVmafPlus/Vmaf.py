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
from .FFmpeg import FFprobe
from .FFmpeg import FFmpegQos
from .FFmpeg import create_unique_dir, create_unique_file, probe_hdr_vmaf, select_models
from .results import heatmap_files, join_heatmaps, join_logs, read_frames, trim_heatmaps, trim_log
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import copy
import functools
import logging
import math
import os
import re

logger = logging.getLogger(__name__)

# The VMAF v1 HFR models are calibrated for the ~50/60 fps regime
# (Netflix/vmaf v3.2.1, resource/doc/models_v1.md)
HFR_MIN_FPS = 50
HFR_MAX_CALIBRATED_FPS = 60
# libvmaf 3.2.1 CAMBI rejects an encoding size below 180x150, or with both sides below 216
# (feature/cambi.c), and the whole VMAF v1 run fails
CAMBI_MIN_ENC_WIDTH = 180
CAMBI_MIN_ENC_HEIGHT = 150
CAMBI_MIN_ENC_SIDE = 216
# A sync pass drops the frames up to this long before its offset without filtering them;
# the margin feeds deinterlacing its neighbouring frames
SYNC_PREROLL = 1.0
# A frame range starts decoding this long before its first frame, so that yadif and fps
# see the same neighbouring frames as a full run
RANGE_PREROLL = 1.0
# Frames measured and dropped on each side of a range: the motion features of a frame
# depend on the previous and the next frame
RANGE_CONTEXT_FRAMES = 2
# Range bounds sit a quarter frame before a frame: trim rounds them to the nearest timestamp
_RANGE_BOUND_MARGIN = 0.25
# Containers whose seek lands on the requested frame; MPEG-TS seeks to the next keyframe
_RANGE_FORMATS = ('mov', 'mp4', 'matroska', 'webm')
# Fewest frames per job: a shorter range measures more context than frames
_JOB_MIN_FRAMES = 2 * RANGE_CONTEXT_FRAMES + 1

# -model_option syntax: no ':', '|', '[', ']', ';' or quotes can reach the filtergraph
_MODEL_OPTION_RE = re.compile(r'^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*=[A-Za-z0-9_.\-]+$')

# Colorimetry (-reference_color, -distorted_color): the attributes libvmaf converts the
# HDR-VMAF pictures from, as FFmpeg names them, and the values vf_libvmaf maps
_COLOR_VALUES = {
    'range': ('tv', 'pc'),
    'primaries': ('bt709', 'bt2020', 'smpte432'),
    'trc': ('bt709', 'smpte2084', 'smpte170m', 'bt2020-10', 'bt2020-12'),
    'matrix': ('bt709', 'bt2020nc'),
}
_COLOR_PRESETS = {
    'hdr10': {'range': 'tv', 'primaries': 'bt2020', 'trc': 'smpte2084', 'matrix': 'bt2020nc'},
    'bt709': {'range': 'tv', 'primaries': 'bt709', 'trc': 'bt709', 'matrix': 'bt709'},
}
# ffprobe stream field of each attribute
_COLOR_STREAM_FIELDS = {'range': 'color_range', 'primaries': 'color_primaries',
                        'trc': 'color_transfer', 'matrix': 'color_space'}
_PQ_TRC = 'smpte2084'
_HLG_TRC = 'arib-std-b67'

_YUV_PIX_FMT_RE = re.compile(r'^yuvj?(420|422|444)p(\d+)?(le|be)?$')
_SEMIPLANAR_PIX_FMTS = {
    'nv12': ('420', 8), 'nv21': ('420', 8),
    'p010le': ('420', 10), 'p010be': ('420', 10),
}


@dataclass
class FeatureConfig:
    """
    One entry of the libvmaf feature= option: the feature name and its
    parameters, joined by escaped colons.
    """
    name: str
    params: Dict[str, str] = field(default_factory=dict)

    def to_string(self):
        parts = [f'name={self.name}'] + [f'{k}={v}' for k, v in self.params.items()]
        return '\\\\:'.join(parts)


class UnsupportedFramerateError(ValueError):
    """
    Raised when no deinterlace filter covers the frame rates of REF and MAIN
    for their interlace combination.
    """
    pass


class UnsupportedModelConfigError(ValueError):
    """
    Raised when the requested models cannot be computed with the given options,
    e.g. a view without a VMAF v1 model, or an HDR run on an input or a build
    that cannot be scored with HDR-VMAF.
    """
    pass


class UnsupportedRangeError(ValueError):
    """
    Raised when a frame range cannot be measured with the frames and scores of
    the full run: options not supported with a range, an input whose container
    cannot seek exactly, or frames missing after a seek.
    """
    pass


@functools.lru_cache(maxsize=None)
def _hdr_vmaf_supported():
    """check_ffmpeg()['hdr_vmaf'], probed once per process"""
    return probe_hdr_vmaf()


def _parseColor(value, label='color'):
    """
    Colorimetry overrides from a preset ('hdr10', 'bt709') or from
    'key=value[,key=value]' with the keys range, primaries, trc and matrix.
    None gives no override. Returns {attribute: value}.
    """
    if value is None:
        return {}
    text = str(value).strip().lower()
    if text in _COLOR_PRESETS:
        return dict(_COLOR_PRESETS[text])
    presets = ', '.join(_COLOR_PRESETS)
    color = {}
    for item in text.split(','):
        key, sep, val = (part.strip() for part in item.partition('='))
        if not sep or key not in _COLOR_VALUES:
            raise UnsupportedModelConfigError(
                f"Invalid {label} '{value}': expected {presets} or key=value[,key=value] "
                f"with the keys {', '.join(_COLOR_VALUES)}")
        if val not in _COLOR_VALUES[key]:
            raise UnsupportedModelConfigError(
                f"Invalid {label} {key} '{val}': expected one of {', '.join(_COLOR_VALUES[key])}")
        if key in color:
            raise UnsupportedModelConfigError(f"Invalid {label} '{value}': {key} given twice")
        color[key] = val
    return color


def _parsePixFmt(pix_fmt) -> Optional[Tuple[str, int]]:
    """(chroma subsampling, bit depth) of a pixel format, or None if unknown"""
    if pix_fmt in _SEMIPLANAR_PIX_FMTS:
        return _SEMIPLANAR_PIX_FMTS[pix_fmt]
    match = _YUV_PIX_FMT_RE.match(str(pix_fmt))
    if not match:
        return None
    return match.group(1), int(match.group(2) or 8)


def _pixFmtBitdepth(pix_fmt) -> int:
    """Bit depth of a pixel format: its trailing digits, or 8 without them"""
    parsed = _parsePixFmt(pix_fmt)
    if parsed:
        return parsed[1]
    match = re.search(r'p(\d+)(le|be)?$', str(pix_fmt))
    return int(match.group(1)) if match else 8


def cambi_encoding_size(width, height) -> Tuple[int, int]:
    """
    The encoding size passed to CAMBI: (width, height) when CAMBI accepts it,
    otherwise the smallest accepted size with the same aspect ratio, e.g. 256x144 to 267x150.
    """
    scale = max(1.0, CAMBI_MIN_ENC_WIDTH / width, CAMBI_MIN_ENC_HEIGHT / height,
                CAMBI_MIN_ENC_SIDE / max(width, height))
    # The epsilon keeps float error from rounding an exact side up by one
    return math.ceil(width * scale - 1e-9), math.ceil(height * scale - 1e-9)


def _parseSize(size) -> Tuple[int, int]:
    """(width, height) from a 'WxH' string or a two-item sequence"""
    if isinstance(size, str):
        width, _, height = size.lower().partition('x')
        size = (width, height)
    try:
        width, height = (int(v) for v in size)
    except (TypeError, ValueError):
        raise UnsupportedModelConfigError(f"Invalid enc_size {size!r}: expected WxH") from None
    if width < 1 or height < 1:
        raise UnsupportedModelConfigError(f"Invalid enc_size {size!r}: expected positive WxH")
    return width, height


def validate_model_config(model='HD', vmaf_versions=('1',), views=None, hfr='auto', bitdepth='auto',
                          enc_size=None, enc_bitdepth=None, model_options=(), labels=None,
                          dynamic_range='auto', reference_color=None, distorted_color=None):
    """
    Check that the requested models can be computed with these options, without probing
    any input. vmaf() runs it in its constructor; cli.py runs it once on the arguments.
    labels names each argument in the messages, e.g. {'views': '-view'}.
    Returns the selected models without overrides, SDR unless dynamic_range is 'hdr'.
    Raises UnsupportedModelConfigError or ValueError.
    """
    labels = labels or {}

    def label(argument):
        return labels.get(argument, argument)

    vmaf_versions = tuple(str(v) for v in vmaf_versions)
    hfr, bitdepth = str(hfr).lower(), str(bitdepth).lower()
    has_v1 = '1' in vmaf_versions
    dynamic_range = str(dynamic_range).lower()
    if dynamic_range not in ('auto', 'sdr', 'hdr'):
        raise UnsupportedModelConfigError(
            f"{label('dynamic_range')} must be auto, sdr or hdr, not '{dynamic_range}'")
    if dynamic_range == 'hdr' and '0.6' in vmaf_versions:
        raise UnsupportedModelConfigError(
            f"{label('dynamic_range')} hdr: VMAF 0.6 has no HDR models, remove 0.6 "
            f"from {label('vmaf_versions')}")
    for argument, value in (('reference_color', reference_color), ('distorted_color', distorted_color)):
        _parseColor(value, label(argument))
        if value is not None and dynamic_range == 'sdr':
            raise UnsupportedModelConfigError(
                f"{label(argument)}: only for the HDR models, not with {label('dynamic_range')} sdr")
    # select_models rejects unknown models, versions and views with ValueError
    models = select_models(model, vmaf_versions, views, dynamic_range='hdr' if dynamic_range == 'hdr' else 'sdr')
    if hfr not in ('auto', 'on', 'off'):
        raise UnsupportedModelConfigError(f"{label('hfr')} must be auto, on or off, not '{hfr}'")
    if bitdepth not in ('auto', '8', '10'):
        raise UnsupportedModelConfigError(f"{label('bitdepth')} must be auto, 8 or 10, not '{bitdepth}'")
    if not has_v1:
        v1_only = [name for name, used in (
            (label('views'), views is not None), (label('hfr'), hfr == 'on'),
            (label('enc_size'), enc_size is not None),
            (label('enc_bitdepth'), enc_bitdepth is not None),
            (label('model_options'), bool(model_options))) if used]
        if v1_only:
            raise UnsupportedModelConfigError(
                f"{', '.join(v1_only)}: only for VMAF version 1 models, add 1 to {label('vmaf_versions')}")
    for option in model_options:
        if not _MODEL_OPTION_RE.match(option):
            raise UnsupportedModelConfigError(
                f"Invalid {label('model_options')} '{option}': expected feature.option=value")
    if enc_size is not None:
        _parseSize(enc_size)
    return models


def validate_range_config(start_frame=None, frame_count=None, subsample=1, labels=None):
    """
    Check a frame range and the options it is combined with, without probing any input.
    Raises UnsupportedRangeError.
    """
    if start_frame is None and frame_count is None:
        return
    labels = labels or {}

    def label(argument):
        return labels.get(argument, argument)

    for argument, value, minimum in (('start_frame', start_frame, 0), ('frame_count', frame_count, 1)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
            raise UnsupportedRangeError(
                f"{label(argument)} must be an integer of at least {minimum}, not {value!r}")
    if subsample != 1:
        raise UnsupportedRangeError(
            f"{label('subsample')}: not supported with a frame range "
            f"({label('start_frame')}, {label('frame_count')})")


def validate_jobs_config(jobs=1, subsample=1, labels=None):
    """
    Check the number of parallel jobs and the options it is combined with.
    Raises UnsupportedRangeError: jobs above 1 split the run into frame ranges.
    """
    labels = labels or {}

    def label(argument):
        return labels.get(argument, argument)

    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise UnsupportedRangeError(f"{label('jobs')} must be an integer of at least 1, not {jobs!r}")
    if jobs > 1 and subsample != 1:
        raise UnsupportedRangeError(
            f"{label('jobs')} above 1 measures frame ranges in parallel, which do not "
            f"support {label('subsample')} above 1")


class video():
    """
    Video class to parse information of video streams obtained
    by FFmpeg.FFprobe
    """

    def __init__(self, videoSrc, loglevel="info"):
        self.videoSrc = videoSrc
        self.loglevel = loglevel
        self.streamInfo = None
        self.framesInfo = None
        self.packetsInfo = None
        self._formatInfo = None
        self._interlaced = None
        self.interlacedFrames = None
        self.totalFrames = None
        self.bytesFramesTotal = None
        self.getStreamInfo()
        self.duration = self.getDuration()

    @property
    def formatInfo(self):
        """ffprobe format info, probed on first use"""
        if self._formatInfo is None:
            self.getFormatInfo()
        return self._formatInfo

    @property
    def interlaced(self):
        """True when most frames of the first 5 seconds are interlaced, probed on first use"""
        if self._interlaced is None:
            self.getFramesInfo()
        return self._interlaced

    def _updateFramesSummary(self):
        interlacedFrames_count = 0
        bytesFramesTotal = 0
        if self.framesInfo is None:
            return
        for frame in self.framesInfo:
            interlacedFrames_count = interlacedFrames_count + \
                int(frame['interlaced_frame'])
            bytesFramesTotal = bytesFramesTotal + int(frame['pkt_size'])
        self.interlacedFrames = interlacedFrames_count
        self.totalFrames = len(self.framesInfo)
        self.bytesFramesTotal = bytesFramesTotal
        self._interlaced = bool(round(self.interlacedFrames/self.totalFrames))

    def getDuration(self):
        """Duration minus start time in seconds, floored to milliseconds"""
        try:
            duration = float(self.streamInfo['duration']) - float(self.streamInfo['start_time'])
            if duration < 0:
                duration = float(self.streamInfo['duration'])
        except KeyError:
            duration = float(self.formatInfo['duration']) - float(self.formatInfo['start_time'])
            if duration < 0:
                duration = float(self.formatInfo['duration'])
        return math.floor(duration * 1000) / 1000

    def getStreamInfo(self):
        logger.info("\n\n=======================================")
        logger.info("[easyVmafPlus] Getting stream info... %s", self.videoSrc)
        logger.info("=======================================")
        self.streamInfo = FFprobe(self.videoSrc, self.loglevel).getStreamInfo()
        return self.streamInfo

    def getFramesInfo(self):
        logger.info("\n\n=======================================")
        logger.info("[easyVmafPlus] Getting frames info... %s", self.videoSrc)
        logger.info("=======================================")
        self.framesInfo = FFprobe(self.videoSrc, self.loglevel).getFramesInfo()
        self._updateFramesSummary()
        return self.framesInfo

    def getPacketsInfo(self):
        logger.info("\n\n=======================================")
        logger.info("[easyVmafPlus] Getting packets info... %s", self.videoSrc)
        logger.info("=======================================")
        self.packetsInfo = FFprobe(self.videoSrc, self.loglevel).getPacketsInfo()
        return self.packetsInfo

    def getFormatInfo(self):
        logger.info("\n\n=======================================")
        logger.info("[easyVmafPlus] Getting format info... %s", self.videoSrc)
        logger.info("=======================================")
        self._formatInfo = FFprobe(self.videoSrc, self.loglevel).getFormatInfo()
        logger.debug("%s", self._formatInfo)
        return self._formatInfo


class vmaf():
    """
    Video class to manage VMAF computation of video streams. This class allows:
        - Upscale or downscale the MAIN or REF videos automatically according to the Vmaf model (1080, 4K, etc)
        - Deinterlace automatically the MAIN and REF videos if needed
        - To SYNC (in time) the MAIN and REF videos using psnr computation
        - Frame rate conversion (if needed)
        - HDR-VMAF for PQ inputs, frame ranges and parallel jobs
    """

    def __init__(self, mainSrc, refSrc, output_fmt, model="HD", phone=False, loglevel="info", subsample=1, threads=0, print_progress=False, end_sync=False,  manual_fps=0, cambi_heatmap=False, vmaf_v0=False, disable_hfr=False, *, vmaf_versions=None, views=None, hfr=None, bitdepth='auto', enc_size=None, enc_bitdepth=None, model_options=(), dynamic_range='auto', reference_color=None, distorted_color=None, start_frame=None, frame_count=None, jobs=1):
        """
        vmaf_versions: VMAF generations to compute, '1' and/or '0.6'; None follows vmaf_v0.
        views:         v1 viewing distances ('3h', '5h' or 'phone', '1.5h'); None keeps every
                       model of the display.
        hfr:           'auto', 'on' or 'off'; None follows disable_hfr.
        bitdepth:      measurement bit depth, 'auto', 8 or 10.
        enc_size, enc_bitdepth: CAMBI encoding size 'WxH' or (W, H) and bit depth of the
                       distorted video; None reads them from its stream.
        model_options: 'feature.option=value' overrides for the v1 models.
        dynamic_range: 'auto' (HDR-VMAF when either input is PQ), 'sdr' or 'hdr'.
        reference_color, distorted_color: colorimetry over what ffprobe reports, 'hdr10',
                       'bt709' or key=value pairs with the keys range, primaries, trc, matrix.
        start_frame, frame_count: the frame range to measure, numbered as in the log of the
                       full run; None measures everything.
        jobs:          ffmpeg processes that compute VMAF at once, each on a frame range.
        """
        self.loglevel = loglevel
        self.model = model
        self.phone = phone
        self.subsample = subsample
        if vmaf_versions:
            self.vmaf_versions = tuple(str(v) for v in vmaf_versions)
        else:
            self.vmaf_versions = ('0.6',) if vmaf_v0 else ('1',)
        self.vmaf_v0 = '1' not in self.vmaf_versions
        self.views = tuple(views) if views else None
        self.hfr = str(hfr).lower() if hfr else ('off' if disable_hfr else 'auto')
        self.disable_hfr = self.hfr == 'off'
        self.bitdepth = str(bitdepth).lower()
        self.enc_size = _parseSize(enc_size) if enc_size is not None else None
        self.enc_bitdepth = int(enc_bitdepth) if enc_bitdepth is not None else None
        self.model_options = tuple(model_options)
        self._dynamicRangeRequest = str(dynamic_range).lower()
        self._colorOverrides = {'reference': reference_color, 'distorted': distorted_color}
        validate_range_config(start_frame, frame_count, subsample)
        validate_jobs_config(jobs, subsample)
        self.start_frame = start_frame
        self.frame_count = frame_count
        self.jobs = jobs
        self._jobPart = False
        self._initResolutions()
        self._validateModelConfig()
        self.main = video(mainSrc, self.loglevel)
        self.ref = video(refSrc, self.loglevel)
        self._resolveDynamicRange()
        if self._hasRange() or self.jobs > 1:
            self._checkRangeInputs()
        if print_progress and self.jobs > 1:
            logger.warning("No progress is shown with %s parallel jobs", self.jobs)
            print_progress = False
        self.ffmpegQos = FFmpegQos(
            self.main.videoSrc, self.ref.videoSrc, self.loglevel)
        self.offset = 0
        self.manual_fps = manual_fps
        self.output_fmt = output_fmt
        self.threads = threads
        self.print_progress = print_progress
        self.end_sync = end_sync
        self.cambi_heatmap = cambi_heatmap
        self.models = []
        self.features = None
        self.pix_fmt = None
        self.output_fps = None
        self.hfr_active = False
        self.frames_scored = None

    def _hasV1(self):
        return '1' in self.vmaf_versions

    def _hasRange(self):
        return self.start_frame is not None or self.frame_count is not None

    def _initResolutions(self):
        """
        initialization of resolutions for each vmaf model
        """
        if self.model == 'HD':
            self.target_resolution = [1920, 1080]
        elif self.model == '4K':
            self.target_resolution = [3840, 2160]
        else:
            raise ValueError(f"Invalid VMAF model: {self.model!r}. Supported: HD, 4K")

    def _validateModelConfig(self):
        validate_model_config(self.model, self.vmaf_versions, self.views, self.hfr, self.bitdepth,
                              self.enc_size, self.enc_bitdepth, self.model_options,
                              dynamic_range=self._dynamicRangeRequest,
                              reference_color=self._colorOverrides['reference'],
                              distorted_color=self._colorOverrides['distorted'])

    def _checkRangeInputs(self):
        """Reject inputs whose container cannot seek to the requested frame"""
        for stream, role in ((self.main, 'distorted'), (self.ref, 'reference')):
            format_name = stream.formatInfo.get('format_name', '')
            if not set(format_name.split(',')) & set(_RANGE_FORMATS):
                what = "A frame range needs" if self._hasRange() else "Parallel jobs measure frame ranges, which need"
                hint = "" if self._hasRange() else "; remux it first, e.g. ffmpeg -i in.ts -map 0:v:0 -c copy out.mkv"
                raise UnsupportedRangeError(
                    f"{what} inputs that seek to an exact frame (MP4, MOV, Matroska or WebM); "
                    f"the {role} video {stream.videoSrc} is {format_name}{hint}")

    def _resolveDynamicRange(self):
        """
        Decide SDR or HDR and the colorimetry of each input: what ffprobe reports with the
        overrides on top. 'auto' is HDR when either input is PQ. Sets self.dynamic_range and
        self.color ({'reference': {...}, 'distorted': {...}} with HDR, None with SDR).
        """
        self.dynamic_range, self.color = 'sdr', None
        if self._dynamicRangeRequest == 'sdr':
            return
        streams = {'reference': self.ref, 'distorted': self.main}
        color = {}
        for role, stream in streams.items():
            detected = {key: stream.streamInfo.get(field) for key, field in _COLOR_STREAM_FIELDS.items()}
            color[role] = {key: value for key, value in detected.items() if value and value != 'unknown'}
            color[role].update(_parseColor(self._colorOverrides[role]))
        for role, stream in streams.items():
            if color[role].get('trc') == _HLG_TRC:
                raise UnsupportedModelConfigError(
                    f"The {role} video {stream.videoSrc} is HLG ({_HLG_TRC}): HDR-VMAF only "
                    f"measures PQ ({_PQ_TRC}). Use -dynamic_range sdr to score it with the SDR models")
        pq = [role for role in streams if color[role].get('trc') == _PQ_TRC]
        if self._dynamicRangeRequest == 'auto' and not pq:
            return
        if '0.6' in self.vmaf_versions:
            raise UnsupportedModelConfigError(
                f"The {' and '.join(pq)} video{'s are' if len(pq) > 1 else ' is'} HDR (PQ) and "
                f"VMAF 0.6 has no HDR models: compute VMAF version 1 only, or use -dynamic_range sdr "
                f"to score with the SDR models")
        if not _hdr_vmaf_supported():
            raise UnsupportedModelConfigError(
                "This FFmpeg build cannot compute the HDR-VMAF models: they need a libvmaf with the "
                "HDR models and zimg, and an FFmpeg that passes the frame colorimetry, as in the "
                "easyVmafPlus Docker image. Use -dynamic_range sdr to score with the SDR models")
        for role, stream in streams.items():
            flag = f"-{role}_color"
            for key, values in _COLOR_VALUES.items():
                value = color[role].get(key)
                if value is None:
                    raise UnsupportedModelConfigError(
                        f"The {role} video {stream.videoSrc} does not report its color {key}, "
                        f"which the HDR-VMAF models need: set it, e.g. {flag} {key}={_COLOR_PRESETS['hdr10'][key]}")
                if value not in values:
                    raise UnsupportedModelConfigError(
                        f"The {role} video {stream.videoSrc} has color {key} {value}, which the "
                        f"HDR-VMAF models do not support ({', '.join(values)}); override it with {flag} {key}=<value>")
        if len(pq) == 1:
            other = 'distorted' if pq[0] == 'reference' else 'reference'
            logger.warning("Mixed inputs: the %s video is PQ and the %s video is not; "
                           "HDR-VMAF converts the %s video to PQ", pq[0], other, other)
        self.dynamic_range = 'hdr'
        self.color = {role: {key: color[role][key] for key in _COLOR_VALUES} for role in streams}

    def _colorConversion(self, role):
        """
        With HDR, (matrix, range) the input of role is converted to before libvmaf, or None.
        vf_libvmaf negotiates one matrix and range for both inputs and would re-matrix one of
        them: the PQ input (the reference, if both are) keeps its own, the other is converted.
        """
        if self.color is None:
            return None
        kept = 'distorted' if (self.color['distorted']['trc'] == _PQ_TRC
                               and self.color['reference']['trc'] != _PQ_TRC) else 'reference'
        target = (self.color[kept]['matrix'], self.color[kept]['range'])
        if role == kept or (self.color[role]['matrix'], self.color[role]['range']) == target:
            return None
        return target

    def _applyColorParams(self):
        """With HDR, tag each input with its resolved colorimetry as the first filter of its chain"""
        if self.color is None:
            return
        for role, chain in (('distorted', self.ffmpegQos.main), ('reference', self.ffmpegQos.ref)):
            color = self.color[role]
            chain.setColorParamsFilter(color['range'], color['primaries'], color['trc'], color['matrix'])

    def _rangeSuffix(self):
        """'_f<start>-<last>' ('-end' without a count) for a range, '' otherwise; '_part' for a job"""
        if not self._hasRange():
            return ''
        start = self.start_frame or 0
        last = start + self.frame_count - 1 if self.frame_count is not None else 'end'
        return f'_f{start}-{last}' + ('_part' if self._jobPart else '')

    def _cambiHeatmapPath(self):
        return os.path.splitext(self.main.videoSrc)[0] + '_cambi_heatmap' + self._rangeSuffix()

    def _logFormat(self):
        return self.output_fmt if self.output_fmt in ('xml', 'csv') else 'json'

    def _rangeLogPath(self):
        """VMAF log name of a range, after the frames requested"""
        return f'{os.path.splitext(self.main.videoSrc)[0]}_vmaf{self._rangeSuffix()}.{self._logFormat()}'

    @staticmethod
    def _chains(qos):
        """(distorted, reference) chains of qos: a reverse sync pass opens the reference first"""
        return (qos.ref, qos.main) if qos.invertedSrc else (qos.main, qos.ref)

    def _applyScaleFilters(self, qos):
        """Scale each input that is not at the target resolution of the model"""
        distorted, reference = self._chains(qos)
        for stream, chain in ((self.main, distorted), (self.ref, reference)):
            if [stream.streamInfo['width'], stream.streamInfo['height']] != self.target_resolution:
                chain.setScaleFilter(self.target_resolution[0], self.target_resolution[1])

    def _normalizeChains(self, qos):
        """
        Bring both inputs of qos to a common frame rate, then to the target resolution:
        deinterlacing must see the original fields, which scaling would blend. The sync passes
        and the final run share it, so a sync offset lands on the same frame in both.
        Returns the effective frame rate of the distorted input.
        """
        if self.manual_fps == 0:
            output_fps = self._applyDeinterlaceFilters(qos)
        else:
            for chain in self._chains(qos):
                chain.setFpsFilter(self.manual_fps)
            output_fps = self.manual_fps
        self._applyScaleFilters(qos)
        return output_fps

    def _measurementPixFmt(self):
        """
        Pixel format both inputs are measured in: the chroma subsampling of the reference;
        10 bits with a v1 model (or the reference depth if higher), the reference depth with
        0.6 models only, or the -bitdepth value.
        """
        ref_pix_fmt = self.ref.streamInfo.get('pix_fmt')
        parsed = _parsePixFmt(ref_pix_fmt)
        if parsed is None:
            logger.warning("Unrecognized reference pixel format %s; measuring as 4:2:0", ref_pix_fmt)
            parsed = ('420', 8)
        chroma, ref_depth = parsed
        if self.bitdepth != 'auto':
            depth = int(self.bitdepth)
        elif self._hasV1():
            depth = max(10, ref_depth)
        else:
            depth = ref_depth
        return f'yuv{chroma}p' if depth == 8 else f'yuv{chroma}p{depth}le'

    def _applyPixelFormat(self):
        """
        Convert each input whose pixel format differs from the measurement format, as the
        first filter of its chain after the HDR tags: otherwise FFmpeg picks the format and
        may degrade the reference, or deinterlace at 8 bits before converting. With mixed HDR
        inputs the same conversion changes the matrix and range of the SDR input.
        """
        self.pix_fmt = self._measurementPixFmt()
        for stream, chain, role in ((self.main, self.ffmpegQos.main, 'distorted'),
                                    (self.ref, self.ffmpegQos.ref, 'reference')):
            conversion = self._colorConversion(role)
            if stream.streamInfo.get('pix_fmt') != self.pix_fmt or conversion:
                chain.setFormatFilter(self.pix_fmt, *(conversion or ()))

    def _deinterlaceFrame(self, factor, chain, fps):
        """yadif frame mode on chain, then fps when REF is not exactly factor times MAIN. Returns the fps forced, or None"""
        ref_fps = getFrameRate(self.ref.streamInfo['r_frame_rate'])
        main_fps = getFrameRate(self.main.streamInfo['r_frame_rate'])

        chain.setDeintFrameFilter()
        if round(ref_fps, 2) != round(factor*main_fps, 2):
            chain.setFpsFilter(round(fps, 5))
            return round(fps, 5)
        return None

    def _deinterlaceField(self, factor, chain, fps):
        """yadif field mode on chain, then fps when REF is not exactly factor times MAIN. Returns the fps forced, or None"""
        ref_fps = getFrameRate(self.ref.streamInfo['r_frame_rate'])
        main_fps = getFrameRate(self.main.streamInfo['r_frame_rate'])

        chain.setDeintFieldFilter()
        if round(ref_fps, 2) != round(factor*main_fps, 2):
            chain.setFpsFilter(round(fps, 5))
            return round(fps, 5)
        return None

    def _applyDeinterlaceFilters(self, qos):
        """
        Normalizes the frame rate between MAIN and REF in the given FFmpegQos, deinterlacing
        the interlaced input when the other is progressive. Returns the effective frame rate
        of MAIN after the filters.
        """
        ref_fps = getFrameRate(self.ref.streamInfo['r_frame_rate'])
        main_fps = getFrameRate(self.main.streamInfo['r_frame_rate'])
        distorted, reference = self._chains(qos)
        unsupported = (f"No deinterlace filter available for the given framerate combination. "
                       f"ref={round(ref_fps, 5)}fps (interlaced={self.ref.interlaced}), "
                       f"main={round(main_fps, 5)}fps (interlaced={self.main.interlaced}). "
                       f"Consider using the -fps flag to force a frame rate manually.")

        if self.ref.interlaced == self.main.interlaced:
            """ Not Deinterlace would be required. So this functions normalizes the fps between REF and MAIN
            """
            if round(ref_fps) < round(main_fps):
                """
                frame rate conversion over MAIN video. The lowest framerate is choosed (REF fps).
                """
                logger.warning("Frame rate conversion can produce bad vmaf scores")
                distorted.setFpsFilter(round(ref_fps, 5))
                return round(ref_fps, 5)
            elif round(ref_fps) > round(main_fps):
                """
                frame rate conversion over REF video. The lowest framerate is choosed (MAIN fps).
                """
                logger.warning("Frame rate conversion can produce bad vmaf scores")
                reference.setFpsFilter(round(main_fps, 5))
            else:
                """
                This just pass the original framerate to the ffmpeg filter (lavfi) when no frame rate
                conversion is requiered. For some reason it is mandatory by lavfi in order to work properly.
                """
                distorted.setFpsFilter(round(main_fps, 5))
                reference.setFpsFilter(round(ref_fps, 5))
            return round(main_fps, 5)

        elif self.ref.interlaced and not self.main.interlaced:
            """
            REF interlaced  | MAIN progressive: MAIN keeps its frame rate
            """
            if round(ref_fps) == round(main_fps*2):
                # Examples: REF=60i, MAIN=30p
                # REF=59.97i, MAIN=30p, etc
                self._deinterlaceFrame(2, reference, main_fps)

            elif round(ref_fps) == round(main_fps):
                # Examples:
                # REF=30i, MAIN=30p
                # REF=29.97i, MAIN=30p, etc
                self._deinterlaceFrame(1, reference, main_fps)

            elif round(ref_fps) == round(main_fps/2):
                # Examples:
                # REF=30i, MAIN=60p
                # REF=29.97i, MAIN=60p, etc
                self._deinterlaceField(0.5, reference, main_fps)

            else:
                raise UnsupportedFramerateError(unsupported)
            return round(main_fps, 5)

        elif not self.ref.interlaced and self.main.interlaced:
            """
            Input Progressive (REF) | Output Interlaced (MAIN)
            Field deinterlacing doubles MAIN's frame rate unless an fps filter follows.
            """
            if round(ref_fps) == round(main_fps*2):
                # Examples: REF=60p, MAIN=30i
                # REF=60p, MAIN=29.97i, etc
                forced = self._deinterlaceField(2, distorted, ref_fps)
                return forced or round(main_fps*2, 5)

            elif round(ref_fps) == round(main_fps):
                # Examples:
                # REF=30p, MAIN=30i
                # REF=30p, MAIN=29.97i, etc
                forced = self._deinterlaceFrame(1, distorted, ref_fps)
                return forced or round(main_fps, 5)

            elif round(ref_fps) == round(main_fps/2):
                # REF=25p, MAIN=25i reported at its field rate (50, H.264 PAFF): one MAIN
                # frame per frame, the mirror of REF interlaced at twice the MAIN rate
                forced = self._deinterlaceFrame(0.5, distorted, ref_fps)
                return forced or round(main_fps/2, 5)

            else:
                raise UnsupportedFramerateError(unsupported)

    def _useHfr(self):
        """
        True for a v1 or HDR run with -force_hfr, or without -disable_hfr at an effective
        frame rate of HFR_MIN_FPS or more. Two interlaced inputs keep the standard models:
        libvmaf then compares interlaced frames, and r_frame_rate can be the field rate.
        """
        if not self._hasV1() or self.hfr == 'off':
            return False
        if self.hfr == 'on':
            return True
        if self.manual_fps == 0 and self.ref.interlaced and self.main.interlaced:
            return False
        return self.output_fps is not None and self.output_fps >= HFR_MIN_FPS

    def _cambiEncodingSize(self, width, height):
        """CAMBI-accepted encoding size, with a warning when it differs"""
        size = cambi_encoding_size(width, height)
        if size != (width, height):
            logger.warning("CAMBI does not accept a %sx%s encoding size; using %sx%s, "
                           "the smallest accepted size with the same aspect ratio", width, height, *size)
        return size

    def _encodingSize(self):
        """(width, height) of the distorted encode for CAMBI: -enc_size or the stream size, CAMBI-accepted"""
        if self.enc_size is not None:
            return self._cambiEncodingSize(*self.enc_size)
        return self._cambiEncodingSize(self.main.streamInfo['width'], self.main.streamInfo['height'])

    def _modelOverrides(self):
        """
        cambi.enc_width, cambi.enc_height and cambi.enc_bitdepth of the distorted video for the
        v1 and HDR models (the distorted video is scaled before libvmaf, so CAMBI would measure
        banding on the scaled picture), then every -model_option value.
        """
        enc_width, enc_height = self._encodingSize()
        if self.enc_bitdepth is not None:
            enc_bitdepth = self.enc_bitdepth
        else:
            enc_bitdepth = _pixFmtBitdepth(self.main.streamInfo.get('pix_fmt'))
        overrides = {'cambi.enc_width': str(enc_width), 'cambi.enc_height': str(enc_height),
                     'cambi.enc_bitdepth': str(enc_bitdepth)}
        for option in self.model_options:
            key, value = option.split('=', 1)
            overrides[key] = value
        return overrides

    def _resolveModels(self):
        """Select the models once output_fps is known, and add the v1 overrides"""
        hfr = self._useHfr()
        self.hfr_active = hfr
        if hfr and self.output_fps is not None and self.output_fps > HFR_MAX_CALIBRATED_FPS:
            logger.warning("%s fps is above the ~50/60 fps the HFR VMAF v1 models are calibrated for",
                           round(self.output_fps, 5))
        self.models = select_models(self.model, self.vmaf_versions, self.views, hfr=hfr,
                                    dynamic_range=self.dynamic_range)
        if self._hasV1():
            overrides = self._modelOverrides()
            for model in self.models:
                if model.path:
                    model.params.update(overrides)
        return self.models

    def _computePsnrAtOffset(self, offset, reverse):
        """
        PSNR between REF trimmed at offset and MAIN trimmed at 0, both 0.5 s long, after the
        filters of the final run. With reverse, MAIN and REF swap roles. Each call builds its
        own FFmpegQos, so calls can run in parallel. Returns (offset, psnr).
        """
        if not reverse:
            qos = FFmpegQos(self.main.videoSrc, self.ref.videoSrc, self.loglevel)
        else:
            qos = FFmpegQos(self.ref.videoSrc, self.main.videoSrc, self.loglevel)
            qos.invertedSrc = True
        # One ffmpeg process per offset already fills the CPUs
        qos._ffmpeg_threads = 1

        # The ref slot holds the searched input. It is trimmed after the same filters as the
        # final run, so the offset selects the same frame or field; the pre-trim only skips
        # the filtering of what comes well before it
        preroll_start = offset - SYNC_PREROLL
        if preroll_start > 0:
            qos.ref.setPreTrimFilter(preroll_start)
        self._normalizeChains(qos)
        qos.ref.setTrimFilter(offset, 0.5)
        qos.main.setTrimFilter(0, 0.5)

        return (offset, qos.getPsnr())

    def syncOffset(self, syncWindow=3, start=0, reverse=False):
        """
        Method to get the offset needed to sync REF and MAIN (if any).
            syncWindow -->  Window Size in seconds to try to sync REF and MAIN videos. i.e., if the video to sync
                            last 600 seconds, the sync look up will be done just within a subsample of syncWindow size.
                            By default, the syncWindow is applied to REF.
            start -->  start time in seconds from the begining of the video where the syncWindow begin.
                        By default, the start time applies to REF.
            reverse --> If this option is set to TRUE. It is considered that MAIN is delayed in comparition to REF: 'syncWindow' and 'start' variables will be
                        applied to MAIN.
                        By default, it is supposed that the REF video is delayed in comparition with the MAIN video.

        It returns the offset value to get REF and MAIN synced and the PSNR computed.
        """
        # The window slides over the reference, or over the distorted video with reverse;
        # past its end ffmpeg has no frame left to compare
        searched, role = (self.main, 'distorted') if reverse else (self.ref, 'reference')
        if start + syncWindow > searched.duration:
            raise ValueError(
                f"The sync window ends at {start + syncWindow:g} s, but the {role} video "
                f"lasts {searched.duration:g} s: lower the sync start or the sync window")

        logger.info("=" * 39)
        logger.info("Syncing... Computing PSNR values...")
        logger.info("=" * 39)
        logger.info("Distorted: %s @ %s fps | %s %s",
                    self.main.videoSrc,
                    round(getFrameRate(self.main.streamInfo['r_frame_rate']), 5),
                    self.main.streamInfo['width'],
                    self.main.streamInfo['height'])
        logger.info("Reference: %s @ %s fps | %s %s",
                    self.ref.videoSrc,
                    round(getFrameRate(self.ref.streamInfo['r_frame_rate']), 5),
                    self.ref.streamInfo['width'],
                    self.ref.streamInfo['height'])
        logger.info("=" * 39)
        logger.info("%-20s %s", "offset(s)", "psnr[dB]")

        fps = getFrameRate(self.ref.streamInfo['r_frame_rate'])
        frameDuration = 1/fps
        startFrame = int(round(start/frameDuration))
        framesInSyncWindow = int(round(syncWindow/frameDuration))
        offsets = [(startFrame + i) * frameDuration for i in range(framesInSyncWindow)]

        if self.manual_fps == 0:
            # Every pass reads the interlace flags: probe them once, before the passes start
            for stream in (self.main, self.ref):
                if stream.framesInfo is None:
                    stream.getFramesInfo()

        max_workers = self.threads if self.threads > 0 else os.cpu_count()
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._computePsnrAtOffset, offset, reverse)
                       for offset in offsets]
            try:
                for future in as_completed(futures):
                    offset, psnr = future.result()
                    results.append((offset, psnr))
                    logger.info("%-20s %s", offset, psnr)
            except BaseException:
                # A failed pass or CTRL-C: drop the passes that have not started
                executor.shutdown(wait=True, cancel_futures=True)
                raise

        # Sorted by offset, so a PSNR tie picks the lowest offset
        results.sort(key=lambda result: result[0])
        offset, maxPsnr = max(results, key=lambda result: result[1])
        """
         The reverse variable/flag indicates if the offset (for time syncing ) was applied over the MAIN or over the REF.
         It is FALSE if it was applied over REF (default) and TRUE if it was applied over the MAIN.
         With reverse, every PSNR pass swapped MAIN and REF in its own FFmpegQos. self.ffmpegQos keeps
         the original MAIN and REF, so only the sign of the offset changes.
        """
        if reverse:
            offset = -1 * offset
        self.offset = offset

        return [self.offset, maxPsnr]

    def _syncTrims(self):
        """(main start, ref start, duration) in seconds of the trims for self.offset, or None at 0"""
        if self.offset > 0:
            offset = self.offset
            return 0, offset, min(self.main.duration, self.ref.duration-offset)
        if self.offset < 0:
            offset = abs(self.offset)
            return offset, 0, min(self.main.duration - offset, self.ref.duration)
        return None

    def _applyOffsetFilters(self, qos):
        """
        Trim filters for self.offset in the given FFmpegQos. Runs after the scale and frame rate filters.
            If offset > 0: Ref delayed compared to  Main. Trimfilter cuts Ref
            if offset < 0: Main delayed compared to Ref. Trimfilter cuts Main
        """
        trims = self._syncTrims()
        if trims:
            main_start, ref_start, duration = trims
            qos.ref.setTrimFilter(ref_start, duration)
            qos.main.setTrimFilter(main_start, duration)

    def setOffset(self, value=None):
        """
        Apply Offset to trim Filter of self.ffmpegQos (see _applyOffsetFilters)
        """

        if value is not None:
            """ overrides the value in self.offset"""
            self.offset = value

        self._applyOffsetFilters(self.ffmpegQos)

    def _rangeBounds(self):
        """(first, end) frames measured for the range, context included; end is exclusive, None to the end"""
        start = self.start_frame or 0
        first = max(0, start - RANGE_CONTEXT_FRAMES)
        if self.frame_count is None:
            return first, None
        return first, start + self.frame_count + RANGE_CONTEXT_FRAMES

    def _applyRange(self):
        """
        Apply the sync trims and the frame range, in place of _applyOffsetFilters. Each chain
        is the one of the full run with three changes: a seek shortly before the range keeping
        the original timestamps, setpts=PTS-<anchor> in place of setpts=PTS-STARTPTS, the
        anchor being the first frame the full run measures, and a final trim to the range plus
        RANGE_CONTEXT_FRAMES on each side. The anchors come from running the chains up to their
        first measured frame, so ffmpeg picks that frame.
        """
        fps = self.output_fps
        trims = self._syncTrims()
        chains = (self.ffmpegQos.main, self.ffmpegQos.ref)
        starts = trims[:2] if trims else (0, 0)

        probe = copy.deepcopy(self.ffmpegQos)
        for chain, start in zip((probe.main, probe.ref), starts, strict=True):
            chain.setSeek(max(0.0, start - RANGE_PREROLL))
            if trims:
                chain.setPreTrimFilter(start)
        anchors = probe.getFirstFrameTimestamps()
        times = [float(pts * time_base) for pts, time_base in anchors]
        logger.info("Range anchors: distorted %s s, reference %s s", times[0], times[1])
        if not trims and abs(times[0] - times[1]) > _RANGE_BOUND_MARGIN / fps:
            # Unsynced, libvmaf pairs frames by timestamp from the first one
            raise UnsupportedRangeError(
                f"The distorted and reference videos start at different timestamps "
                f"({times[0]:g} s and {times[1]:g} s); a frame range needs them synced: "
                f"set an offset or a sync window")

        first, end = self._rangeBounds()
        range_start = max(0.0, (first - _RANGE_BOUND_MARGIN) / fps)
        range_end = (end - _RANGE_BOUND_MARGIN) / fps if end is not None else None
        for chain, start, (pts, _), time in zip(chains, starts, anchors, times, strict=True):
            chain.setSeek(max(0.0, time + first / fps - RANGE_PREROLL))
            if trims:
                chain.setPreTrimFilter(start)
            chain.setPtsShiftFilter(pts)
            if trims:
                chain.setEndTrimFilter(trims[2])
            chain.setRangeTrimFilter(range_start, range_end)

    def _trimRangeLog(self, log_path):
        """
        Drop the context frames from the range log and from the CAMBI heatmaps, and number
        the frames as in the full run. Returns the number of frames kept.
        Raises UnsupportedRangeError when frames are missing before the end of the inputs,
        when the range starts after the last frame, or when a heatmap does not hold one
        picture per measured frame.
        """
        first, end = self._rangeBounds()
        start = self.start_frame or 0
        # libvmaf writes no frames when none reaches it
        measured = len(read_frames(log_path, self._logFormat())) if os.path.getsize(log_path) else 0
        if end is not None and first + measured < end:
            trims = self._syncTrims()
            if trims:
                last_time = trims[2]
            elif self.end_sync:
                last_time = min(self.main.duration, self.ref.duration)
            else:
                last_time = max(self.main.duration, self.ref.duration)
            # Container durations are approximate: two frames of tolerance
            if (first + measured + 2) / self.output_fps < last_time:
                raise UnsupportedRangeError(
                    f"The range measured {measured} frames from frame {first} instead of "
                    f"{end - first}, before the end of the inputs ({last_time:g} s): "
                    f"an input could not be read from the requested frame")
        kept = first + measured - start
        if self.frame_count is not None:
            kept = min(kept, self.frame_count)
        if kept <= 0:
            last = f" ({first + measured - 1})" if measured else ""
            raise UnsupportedRangeError(f"The range starts at frame {start}, after the last measured frame{last}")
        trim_log(log_path, self._logFormat(), start - first, kept, first)
        if self.cambi_heatmap and self.ffmpegQos.vmaf_cambi_heatmap_path:
            try:
                trim_heatmaps(self.ffmpegQos.vmaf_cambi_heatmap_path, start - first, kept, measured)
            except ValueError as e:
                raise UnsupportedRangeError(str(e)) from None
        return kept

    def _build_feature_string(self):
        """
        The libvmaf feature= value: psnr always, cambi with -cambi_heatmap,
        the features joined by |.
        """
        features: List[FeatureConfig] = [FeatureConfig('psnr')]
        if self.cambi_heatmap:
            enc_width, enc_height = self._encodingSize()
            features.append(FeatureConfig('cambi', {
                'full_ref': 'true',
                'enc_width': str(enc_width),
                'enc_height': str(enc_height),
                'src_width': str(self.ref.streamInfo['width']),
                'src_height': str(self.ref.streamInfo['height']),
            }))
        return '|'.join(feature.to_string() for feature in features)

    def getVmaf(self, autoSync=False):
        """
        Filter order: clearFilters, _applyColorParams (HDR), _applyPixelFormat, _normalizeChains
        (deinterlace or fps, then scale), syncOffset when autoSync, setOffset or _applyRange,
        _resolveModels. With jobs above 1, each job runs this on its own frame range.
        """
        if self.jobs > 1:
            ranges = self._planJobs()
            if len(ranges) > 1:
                return self._computeJobs(ranges)
        self.ffmpegQos.clearFilters()
        self.ffmpegQos.main.clearFilters()
        self.ffmpegQos.ref.clearFilters()

        self._applyColorParams()
        self._applyPixelFormat()
        if self.manual_fps:
            logger.warning("Forcing frame rate conversion manually")
        self.output_fps = self._normalizeChains(self.ffmpegQos)

        """Lookup for sync between Main and reference. Default: dissable
           It is suggested to run syncOffset manually before getVmaf()
        """
        if autoSync:
            self.syncOffset()
        if self._hasRange():
            self._applyRange()
        else:
            """Apply Offset filters, if offset =0 nothing happens """
            self.setOffset()

        self._resolveModels()
        self.features = self._build_feature_string()

        logger.info("=" * 39)
        logger.info("Computing VMAF...")
        logger.info("=" * 39)
        logger.info("Distorted: %s @ %s fps | %s %s",
                    self.main.videoSrc,
                    round(getFrameRate(self.main.streamInfo['r_frame_rate']), 5),
                    self.main.streamInfo['width'],
                    self.main.streamInfo['height'])
        logger.info("Reference: %s @ %s fps | %s %s",
                    self.ref.videoSrc,
                    round(getFrameRate(self.ref.streamInfo['r_frame_rate']), 5),
                    self.ref.streamInfo['width'],
                    self.ref.streamInfo['height'])
        logger.info("Offset:     %s", self.offset)
        logger.info("Model:      %s", self.model)
        logger.info("Models:     %s", ', '.join(model.name for model in self.models))
        logger.info("pix_fmt:    %s", self.pix_fmt)
        logger.info("Dynamic range: %s", self.dynamic_range)
        if self.color:
            logger.info("HDR color:  distorted %s, reference %s", *(
                '/'.join(self.color[role][key] for key in _COLOR_VALUES) for role in ('distorted', 'reference')))
        if self._hasRange():
            logger.info("Range:      frames %s, %s", self.start_frame or 0,
                        self.frame_count if self.frame_count is not None else "to the end")
        logger.debug("loglevel:   %s", self.loglevel)
        logger.info("subsample:  %s", self.subsample)
        logger.info("output_fmt: %s", self.output_fmt)
        logger.info("=" * 39)

        log_path = self._rangeLogPath() if self._hasRange() else None
        heatmap_path = self._cambiHeatmapPath() if self.cambi_heatmap else None
        vmafProcess = self.ffmpegQos.getVmaf(log_path=log_path, models=self.models, subsample=self.subsample,
                                             output_fmt=self.output_fmt, threads=self.threads,
                                             print_progress=self.print_progress, end_sync=self.end_sync,
                                             features=self.features, cambi_heatmap=self.cambi_heatmap,
                                             heatmap_path=heatmap_path)
        self.frames_scored = self._trimRangeLog(self.ffmpegQos.vmafpath) if self._hasRange() else None
        return vmafProcess

    def _planJobs(self):
        """
        Split the frames to measure, at output_fps, into contiguous (start_frame, frame_count)
        ranges, one per job, the last one to the end. The total comes from the container
        durations, which are approximate, so every range but the last stays clear of the end.
        """
        self.ffmpegQos.clearFilters()
        self.ffmpegQos.main.clearFilters()
        self.ffmpegQos.ref.clearFilters()
        # Also probes the interlacing of both inputs once, before the jobs share them
        self.output_fps = self._normalizeChains(self.ffmpegQos)
        self.ffmpegQos.main.clearFilters()
        self.ffmpegQos.ref.clearFilters()
        trims = self._syncTrims()
        duration = trims[2] if trims else min(self.main.duration, self.ref.duration)
        start = self.start_frame or 0
        span = math.floor(duration * self.output_fps) - 1 - start
        if self.frame_count is not None:
            span = min(span, self.frame_count)
        jobs = min(self.jobs, span // _JOB_MIN_FRAMES)
        if jobs < 2:
            logger.warning("Too few frames to measure in %s parallel jobs: using one", self.jobs)
            return [(self.start_frame, self.frame_count)]
        size = span // jobs
        ranges = [(start + i * size, size) for i in range(jobs - 1)]
        last = start + (jobs - 1) * size
        ranges.append((last, None if self.frame_count is None else start + self.frame_count - last))
        return ranges

    def _jobRange(self, start_frame, frame_count, threads, ffmpeg_threads):
        """
        A copy of this run that measures one range of a job: it shares the probed inputs and
        the sync offset, with its own FFmpegQos and temporary output names.
        """
        part = copy.copy(self)
        part.jobs = 1
        part._jobPart = True
        part.start_frame, part.frame_count = start_frame, frame_count
        part.threads = threads
        part.print_progress = False
        part.ffmpegQos = FFmpegQos(self.main.videoSrc, self.ref.videoSrc, self.loglevel)
        # One decoder and filter thread per CPU in every job oversubscribes the CPUs
        part.ffmpegQos._ffmpeg_threads = ffmpeg_threads
        return part

    def _removeJobOutputs(self):
        """Delete the temporary log and CAMBI heatmaps of a job's range"""
        log_path = self.ffmpegQos.vmafpath
        if log_path and os.path.exists(log_path):
            os.remove(log_path)
        heatmaps = self.ffmpegQos.vmaf_cambi_heatmap_path
        if heatmaps and os.path.isdir(heatmaps):
            for path in heatmap_files(heatmaps):
                os.remove(path)
            if not os.listdir(heatmaps):
                os.rmdir(heatmaps)

    @staticmethod
    def _runJobs(parts):
        """
        Compute every range at once, one thread per ffmpeg process. On the first failure or
        an interruption the other processes are stopped and that failure is raised.
        """
        with ThreadPoolExecutor(max_workers=len(parts)) as executor:
            futures = [executor.submit(part.getVmaf) for part in parts]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for part in parts:
                    part.ffmpegQos.cancel()
                raise

    def _computeJobs(self, ranges):
        """
        Compute VMAF in one ffmpeg process per range of _planJobs, at the same time, and join
        their logs and CAMBI heatmaps into those of one process. The joined log and heatmap
        folder are reserved first, so no existing output is replaced. The temporary outputs of
        the jobs are deleted, also on a failure, which leaves no joined log.
        """
        share = max(1, (os.cpu_count() or 1) // len(ranges))
        threads = self.threads or share
        parts = [self._jobRange(start, count, threads, share) for start, count in ranges]
        log_file, log_path = create_unique_file(
            f'{os.path.splitext(self.main.videoSrc)[0]}_vmaf{self._rangeSuffix()}.{self._logFormat()}')
        log_file.close()
        heatmap_path = create_unique_dir(self._cambiHeatmapPath()) if self.cambi_heatmap else None
        logger.info("Computing VMAF in %s parallel jobs, %s libvmaf threads each: frames %s",
                    len(parts), threads, ', '.join(
                        f"{start}-{start + count - 1}" if count is not None else f"{start}-end"
                        for start, count in ranges))
        try:
            try:
                self._runJobs(parts)
            except UnsupportedRangeError as e:
                raise UnsupportedRangeError(
                    f"jobs={self.jobs} measures frame ranges in parallel, and one failed: {e}") from e
            for (start, count), part in zip(ranges[:-1], parts[:-1], strict=True):
                # A short range before the last one would leave a gap in the log
                if part.frames_scored != count:
                    raise UnsupportedRangeError(
                        f"The job of frames {start}-{start + count - 1} measured "
                        f"{part.frames_scored} frames instead of {count}")
            join_logs([part.ffmpegQos.vmafpath for part in parts], self._logFormat(), log_path)
            if heatmap_path:
                join_heatmaps([part.ffmpegQos.vmaf_cambi_heatmap_path for part in parts], heatmap_path)
        except BaseException:
            # Only a complete join stays: the reserved names go when still empty
            if os.path.isfile(log_path) and os.path.getsize(log_path) == 0:
                os.remove(log_path)
            if heatmap_path and os.path.isdir(heatmap_path) and not os.listdir(heatmap_path):
                os.rmdir(heatmap_path)
            raise
        finally:
            for part in parts:
                part._removeJobOutputs()

        self.ffmpegQos.vmafpath = log_path
        self.ffmpegQos.vmaf_cambi_heatmap_path = heatmap_path
        self.pix_fmt = parts[0].pix_fmt
        self.output_fps = parts[0].output_fps
        # The models of one process: the jobs' models name their own heatmaps
        self._resolveModels()
        self.features = parts[0].features
        self.frames_scored = sum(part.frames_scored for part in parts) if self._hasRange() else None
        return None

    def getFrames(self, frame_numbers, folder):
        """
        Write the frames with the given frame numbers of the VMAF log as TIFF files into folder,
        through the deinterlace, frame rate and trim filters of getVmaf without scale and format
        filters, so the frames keep the resolution of the distorted input. An HDR run converts
        the frames to BT.709 SDR, because a PQ picture written as 8 bit RGB looks flat. Returns
        (frame number, file, time in the distorted input in seconds) per written frame.
        """
        qos = FFmpegQos(self.main.videoSrc, self.ref.videoSrc, self.loglevel)
        if self.manual_fps == 0:
            self._applyDeinterlaceFilters(qos)
        else:
            qos.main.setFpsFilter(self.manual_fps)
            qos.ref.setFpsFilter(self.manual_fps)
        self._applyOffsetFilters(qos)
        if self.color is not None:
            color = self.color['distorted']
            qos.main.setColorParamsFilter(color['range'], color['primaries'], color['trc'], color['matrix'])
            qos.main.setSdrConversionFilter(self.main.streamInfo['width'], self.main.streamInfo['height'])
        trim_start = abs(self.offset) if self.offset < 0 else 0
        return [(number, path, trim_start + pts_time) for number, path, pts_time in qos.getFrames(frame_numbers, folder)]


def getFrameRate(r_frame_rate):
    num, den = r_frame_rate.split('/')
    return int(num)/int(den)
