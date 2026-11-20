"""Tests for Vmaf.py: the stream info and duration of video, and the scale, deinterlace, frame rate,
pixel format, HDR, sync, trim, range and job logic and the VMAF run of vmaf, with ffprobe and ffmpeg
replaced by fakes. The stream, format and frames info in tests/data are ffprobe outputs."""

import json
import os
import re
import shutil
import subprocess
from fractions import Fraction
from types import SimpleNamespace

import pytest

from easyVmafPlus import Vmaf
from easyVmafPlus.FFmpeg import FFmpegQos
from easyVmafPlus.results import read_frames
from easyVmafPlus.Vmaf import (FeatureConfig, UnsupportedFramerateError, UnsupportedModelConfigError,
                               UnsupportedRangeError, cambi_encoding_size, getFrameRate, video, vmaf)

DATA = os.path.join(os.path.dirname(__file__), 'data')
BS = "\\"


def _json(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


STREAM = _json('ffprobe_streams_mp4.json')['streams'][0]          # h264 1920x1080 yuv420p, 30/1, 2.066667 s
MKV_STREAM = _json('ffprobe_streams_mkv.json')['streams'][0]      # the same stream in Matroska: no duration
MKV_FORMAT = _json('ffprobe_format_mkv.json')['format']
PROGRESSIVE = _json('ffprobe_frames_progressive.json')['frames']  # 3 frames, interlaced_frame 0
INTERLACED = _json('ffprobe_frames_interlaced.json')['frames']    # 3 frames, interlaced_frame 1
PACKETS = _json('ffprobe_packets_mp4.json')['packets']
MP4_FORMAT = {'format_name': 'mov,mp4,m4a,3gp,3g2,mj2', 'duration': '2.066667', 'start_time': '0.000000'}
TS_FORMAT = {'format_name': 'mpegts', 'duration': '2.066667', 'start_time': '0.000000'}
PQ = {'color_range': 'tv', 'color_primaries': 'bt2020', 'color_transfer': 'smpte2084', 'color_space': 'bt2020nc'}
BT709 = {'color_range': 'tv', 'color_primaries': 'bt709', 'color_transfer': 'bt709', 'color_space': 'bt709'}


def stream(**changes):
    """The captured stream info with some values changed"""
    return dict(STREAM, **changes)


@pytest.fixture
def probes(monkeypatch):
    """Vmaf.FFprobe replaced by a fake answering from `answers[path][kind]`, kind one of
    stream, format, frames and packets; `calls` lists the probes run as (kind, path)"""
    answers = {}
    calls = []

    class FakeFFprobe:
        def __init__(self, videoSrc, loglevel="info"):
            self.videoSrc = videoSrc

        def _answer(self, kind):
            calls.append((kind, self.videoSrc))
            return answers[self.videoSrc][kind]

        def getStreamInfo(self):
            return self._answer('stream')

        def getFormatInfo(self):
            return self._answer('format')

        def getFramesInfo(self):
            return self._answer('frames')

        def getPacketsInfo(self):
            return self._answer('packets')

    monkeypatch.setattr(Vmaf, 'FFprobe', FakeFFprobe)
    return SimpleNamespace(answers=answers, calls=calls)


@pytest.fixture
def hdr_build(monkeypatch):
    """probe_hdr_vmaf replaced by a fake; install(True) makes the build HDR capable"""
    def install(supported):
        monkeypatch.setattr(Vmaf, 'probe_hdr_vmaf', lambda: supported)
        Vmaf._hdr_vmaf_supported.cache_clear()
    install(True)
    yield install
    Vmaf._hdr_vmaf_supported.cache_clear()


def make_vmaf(probes, main=STREAM, ref=STREAM, main_frames=PROGRESSIVE, ref_frames=PROGRESSIVE,
              main_format=MP4_FORMAT, ref_format=MP4_FORMAT, **kwargs):
    """A vmaf for main.mp4 and ref.mp4 with the given stream, frames and format info"""
    probes.answers['main.mp4'] = {'stream': main, 'frames': main_frames, 'format': main_format}
    probes.answers['ref.mp4'] = {'stream': ref, 'frames': ref_frames, 'format': ref_format}
    kwargs.setdefault('output_fmt', 'json')
    return vmaf('main.mp4', 'ref.mp4', **kwargs)


def filters(qos):
    return qos.main.filtersList, qos.ref.filtersList


def test_get_frame_rate():
    assert getFrameRate('30/1') == 30
    assert getFrameRate('30000/1001') == pytest.approx(29.97002997)


def test_feature_config():
    assert FeatureConfig('psnr').to_string() == 'name=psnr'
    assert FeatureConfig('cambi', {'full_ref': 'true', 'enc_width': '1280'}).to_string() == (
        'name=cambi' + BS * 2 + ':full_ref=true' + BS * 2 + ':enc_width=1280')


def test_errors_are_value_errors():
    assert issubclass(UnsupportedFramerateError, ValueError)
    assert issubclass(UnsupportedModelConfigError, ValueError)
    assert issubclass(UnsupportedRangeError, ValueError)


@pytest.mark.parametrize("size, expected", [
    ((256, 144), (267, 150)), ((192, 108), (267, 150)), ((160, 90), (267, 150)),
    ((1920, 1080), (1920, 1080)), ((180, 150), (216, 180)), ((216, 150), (216, 150)),
], ids=["256x144", "192x108", "160x90", "1080p", "180x150", "216x150"])
def test_cambi_encoding_size(size, expected):
    assert cambi_encoding_size(*size) == expected


class TestVideo:
    """video: stream info at once; format and frames info on first use; duration and interlace state."""

    def test_stream_info_and_duration(self, probes):
        probes.answers['in.mp4'] = {'stream': STREAM}
        clip = video('in.mp4')
        assert clip.streamInfo == STREAM
        assert clip.duration == 2.066
        assert probes.calls == [('stream', 'in.mp4')]

    @pytest.mark.parametrize("duration, start_time, expected", [
        ('10.000000', '0.500000', 9.5),
        ('2.000000', '3.000000', 2.0),
        ('8.12345', '0.000000', 8.123),
    ], ids=["minus-start-time", "negative-difference", "floored-to-ms"])
    def test_duration(self, probes, duration, start_time, expected):
        probes.answers['in.mp4'] = {'stream': stream(duration=duration, start_time=start_time)}
        assert video('in.mp4').duration == expected

    def test_duration_from_format(self, probes):
        probes.answers['in.mkv'] = {'stream': MKV_STREAM, 'format': MKV_FORMAT}
        clip = video('in.mkv')
        assert clip.duration == 2.066
        assert probes.calls == [('stream', 'in.mkv'), ('format', 'in.mkv')]

    def test_format_info_probed_once(self, probes):
        probes.answers['in.mp4'] = {'stream': STREAM, 'format': MKV_FORMAT}
        clip = video('in.mp4')
        assert probes.calls == [('stream', 'in.mp4')]
        assert clip.formatInfo is MKV_FORMAT
        assert clip.formatInfo is MKV_FORMAT
        assert probes.calls == [('stream', 'in.mp4'), ('format', 'in.mp4')]

    @pytest.mark.parametrize("frames, interlaced, bytes_total", [
        (PROGRESSIVE, False, 628899 + 5619 + 64707),
        (INTERLACED, True, 13111 + 6127 + 8462),
    ], ids=["progressive", "interlaced"])
    def test_interlaced(self, probes, frames, interlaced, bytes_total):
        probes.answers['in.mp4'] = {'stream': STREAM, 'frames': frames}
        clip = video('in.mp4')
        assert clip.interlaced is interlaced
        assert clip.interlaced is interlaced
        assert probes.calls == [('stream', 'in.mp4'), ('frames', 'in.mp4')]
        assert (clip.interlacedFrames, clip.totalFrames, clip.bytesFramesTotal) == (
            3 if interlaced else 0, 3, bytes_total)

    @pytest.mark.parametrize("interlaced_count, interlaced", [(2, True), (1, False)])
    def test_interlaced_by_majority(self, probes, interlaced_count, interlaced):
        frames = INTERLACED[:interlaced_count] + PROGRESSIVE[interlaced_count:]
        probes.answers['in.mp4'] = {'stream': STREAM, 'frames': frames}
        assert video('in.mp4').interlaced is interlaced

    def test_packets_info(self, probes):
        probes.answers['in.mp4'] = {'stream': STREAM, 'packets': PACKETS}
        clip = video('in.mp4')
        assert clip.getPacketsInfo() is PACKETS
        assert clip.packetsInfo is PACKETS


class TestInit:
    """vmaf.__init__: a video and an FFmpegQos for MAIN and REF, the target resolution, the versions and the checks."""

    @pytest.mark.parametrize("model, resolution", [('HD', [1920, 1080]), ('4K', [3840, 2160])])
    def test_target_resolution(self, probes, model, resolution):
        assert make_vmaf(probes, model=model).target_resolution == resolution

    def test_invalid_model(self, probes):
        with pytest.raises(ValueError, match=r"^Invalid VMAF model: '8K'\. Supported: HD, 4K$"):
            make_vmaf(probes, model='8K')

    def test_inputs(self, probes):
        run = make_vmaf(probes, loglevel='verbose')
        assert (run.main.videoSrc, run.ref.videoSrc) == ('main.mp4', 'ref.mp4')
        assert (run.ffmpegQos.main.videoSrc, run.ffmpegQos.ref.videoSrc) == ('main.mp4', 'ref.mp4')
        assert run.ffmpegQos.loglevel == 'verbose'
        assert (run.offset, run.models, run.dynamic_range, run.color) == (0, [], 'sdr', None)
        assert probes.calls == [('stream', 'main.mp4'), ('stream', 'ref.mp4')]

    @pytest.mark.parametrize("kwargs, versions, hfr", [
        ({}, ('1',), 'auto'),
        ({'vmaf_v0': True}, ('0.6',), 'auto'),
        ({'disable_hfr': True}, ('1',), 'off'),
        ({'vmaf_versions': ('1', '0.6'), 'hfr': 'on'}, ('1', '0.6'), 'on'),
        ({'vmaf_v0': True, 'vmaf_versions': ['1']}, ('1',), 'auto'),
    ], ids=["default", "vmaf_v0", "disable_hfr", "explicit", "versions-win"])
    def test_versions_and_hfr(self, probes, kwargs, versions, hfr):
        run = make_vmaf(probes, **kwargs)
        assert (run.vmaf_versions, run.hfr, run.vmaf_v0, run.disable_hfr) == (versions, hfr, '1' not in versions, hfr == 'off')

    @pytest.mark.parametrize("kwargs, message", [
        ({'views': ['3h'], 'vmaf_v0': True}, r"views: only for VMAF version 1 models"),
        ({'model_options': ['bad']}, r"Invalid model_options 'bad'"),
        ({'dynamic_range': 'hdr', 'vmaf_v0': True}, r"dynamic_range hdr: VMAF 0.6 has no HDR models"),
        ({'reference_color': 'hdr10', 'dynamic_range': 'sdr'}, r"reference_color: only for the HDR models"),
        ({'enc_size': '12x'}, r"Invalid enc_size"),
        ({'start_frame': -1}, r"start_frame must be an integer of at least 0"),
        ({'frame_count': 0}, r"frame_count must be an integer of at least 1"),
        ({'start_frame': 0, 'subsample': 2}, r"subsample: not supported with a frame range"),
        ({'jobs': 0}, r"jobs must be an integer of at least 1"),
        ({'jobs': 2, 'subsample': 2}, r"jobs above 1 measures frame ranges in parallel"),
    ], ids=["views-v0", "model-option", "hdr-v0", "color-sdr", "enc-size", "start-frame", "frame-count",
            "range-subsample", "jobs-0", "jobs-subsample"])
    def test_rejected_options(self, probes, kwargs, message):
        with pytest.raises(ValueError, match=message):
            make_vmaf(probes, **kwargs)

    def test_jobs_disable_progress(self, probes, caplog):
        with caplog.at_level('WARNING'):
            run = make_vmaf(probes, jobs=2, print_progress=True)
        assert run.print_progress is False
        assert "No progress is shown with 2 parallel jobs" in caplog.text

    @pytest.mark.parametrize("kwargs", [{'start_frame': 0}, {'jobs': 2}], ids=["range", "jobs"])
    def test_range_inputs_checked(self, probes, kwargs):
        with pytest.raises(UnsupportedRangeError, match=r"the distorted video main.mp4 is mpegts"):
            make_vmaf(probes, main_format=TS_FORMAT, **kwargs)

    def test_jobs_hint_names_remux(self, probes):
        with pytest.raises(UnsupportedRangeError, match=r"Parallel jobs measure frame ranges.*remux it first"):
            make_vmaf(probes, ref_format=TS_FORMAT, jobs=2)


class TestScale:
    """vmaf._applyScaleFilters: inputs not at the target resolution are scaled to it."""

    def test_at_target(self, probes):
        run = make_vmaf(probes)
        run._applyScaleFilters(run.ffmpegQos)
        assert filters(run.ffmpegQos) == ([], [])

    def test_ref_scaled(self, probes):
        run = make_vmaf(probes, ref=stream(width=1280, height=720))
        run._applyScaleFilters(run.ffmpegQos)
        assert filters(run.ffmpegQos) == ([], ['[1:v]scale=1920:1080:flags=bicubic[input1_0]'])

    def test_main_scaled(self, probes):
        run = make_vmaf(probes, main=stream(width=720, height=576))
        run._applyScaleFilters(run.ffmpegQos)
        assert filters(run.ffmpegQos) == (['[0:v]scale=1920:1080:flags=bicubic[input0_0]'], [])

    def test_4k(self, probes):
        run = make_vmaf(probes, model='4K')
        run._applyScaleFilters(run.ffmpegQos)
        assert filters(run.ffmpegQos) == (['[0:v]scale=3840:2160:flags=bicubic[input0_0]'],
                                          ['[1:v]scale=3840:2160:flags=bicubic[input1_0]'])

    def test_inverted(self, probes):
        run = make_vmaf(probes, ref=stream(width=1280, height=720))
        qos = FFmpegQos('ref.mp4', 'main.mp4')
        qos.invertedSrc = True
        run._applyScaleFilters(qos)
        assert filters(qos) == (['[0:v]scale=1920:1080:flags=bicubic[input0_0]'], [])


YADIF_FRAME = 'yadif=0:-1:0'
YADIF_FIELD = 'yadif=1:-1:0'


def chain(input_id, *filter_bodies):
    """Filter strings as inputFFmpeg labels them: [<id>:v]f0[input<id>_0], [input<id>_0]f1[input<id>_1], ..."""
    result = []
    for n, body in enumerate(filter_bodies):
        source = f'{input_id}:v' if n == 0 else f'input{input_id}_{n - 1}'
        result.append(f'[{source}]{body}[input{input_id}_{n}]')
    return result


DEINTERLACE_ROWS = [
    ('25/1', False, '50/1', False, ['fps=fps=25.0'], [], 25.0),
    ('50/1', False, '25/1', False, [], ['fps=fps=25.0'], 25.0),
    ('30/1', False, '30/1', False, ['fps=fps=30.0'], ['fps=fps=30.0'], 30.0),
    ('30000/1001', False, '30000/1001', False, ['fps=fps=29.97003'], ['fps=fps=29.97003'], 29.97003),
    ('50/1', True, '50/1', True, ['fps=fps=50.0'], ['fps=fps=50.0'], 50.0),
    ('60/1', True, '30/1', False, [], [YADIF_FRAME], 30.0),
    ('60000/1001', True, '30/1', False, [], [YADIF_FRAME, 'fps=fps=30.0'], 30.0),
    ('30/1', True, '30/1', False, [], [YADIF_FRAME], 30.0),
    ('25/1', True, '50/1', False, [], [YADIF_FIELD], 50.0),
    ('50/1', False, '25/1', True, [YADIF_FIELD], [], 50.0),
    ('60/1', False, '30000/1001', True, [YADIF_FIELD, 'fps=fps=60.0'], [], 60.0),
    ('30/1', False, '30/1', True, [YADIF_FRAME], [], 30.0),
    ('30/1', False, '30000/1001', True, [YADIF_FRAME, 'fps=fps=30.0'], [], 30.0),
    ('25/1', False, '50/1', True, [YADIF_FRAME], [], 25.0),
]
DEINTERLACE_IDS = ["progressive-ref-lower", "progressive-main-lower", "progressive-equal", "progressive-29.97",
                   "both-interlaced", "ref-interlaced-2x", "ref-interlaced-2x-59.94", "ref-interlaced-1x",
                   "ref-interlaced-half", "main-interlaced-2x", "main-interlaced-2x-29.97", "main-interlaced-1x",
                   "main-interlaced-1x-29.97", "main-interlaced-paff"]


class TestDeinterlace:
    """vmaf._applyDeinterlaceFilters: the frame rate and deinterlace table of Vmaf.py, and the effective rate."""

    @pytest.mark.parametrize("ref_rate, ref_interlaced, main_rate, main_interlaced, main_filters, ref_filters, fps",
                             DEINTERLACE_ROWS, ids=DEINTERLACE_IDS)
    def test_rule(self, probes, ref_rate, ref_interlaced, main_rate, main_interlaced, main_filters, ref_filters, fps):
        run = make_vmaf(probes, main=stream(r_frame_rate=main_rate), ref=stream(r_frame_rate=ref_rate),
                        main_frames=INTERLACED if main_interlaced else PROGRESSIVE,
                        ref_frames=INTERLACED if ref_interlaced else PROGRESSIVE)
        assert run._applyDeinterlaceFilters(run.ffmpegQos) == pytest.approx(fps)
        assert filters(run.ffmpegQos) == (chain(0, *main_filters), chain(1, *ref_filters))

    @pytest.mark.parametrize("ref_rate, ref_interlaced, main_rate, main_interlaced, main_filters, ref_filters, fps",
                             DEINTERLACE_ROWS, ids=DEINTERLACE_IDS)
    def test_inverted(self, probes, ref_rate, ref_interlaced, main_rate, main_interlaced, main_filters, ref_filters, fps):
        """With invertedSrc the distorted video sits in the ref slot: every filter lands on the other chain."""
        run = make_vmaf(probes, main=stream(r_frame_rate=main_rate), ref=stream(r_frame_rate=ref_rate),
                        main_frames=INTERLACED if main_interlaced else PROGRESSIVE,
                        ref_frames=INTERLACED if ref_interlaced else PROGRESSIVE)
        qos = FFmpegQos('ref.mp4', 'main.mp4')
        qos.invertedSrc = True
        assert run._applyDeinterlaceFilters(qos) == pytest.approx(fps)
        assert filters(qos) == (chain(0, *ref_filters), chain(1, *main_filters))

    @pytest.mark.parametrize("ref_rate, ref_interlaced, main_rate, main_interlaced, message", [
        ('25/1', True, '30/1', False, "ref=25.0fps (interlaced=True), main=30.0fps (interlaced=False)"),
        ('25/1', False, '30/1', True, "ref=25.0fps (interlaced=False), main=30.0fps (interlaced=True)"),
    ], ids=["ref-interlaced", "main-interlaced"])
    def test_unsupported(self, probes, ref_rate, ref_interlaced, main_rate, main_interlaced, message):
        run = make_vmaf(probes, main=stream(r_frame_rate=main_rate), ref=stream(r_frame_rate=ref_rate),
                        main_frames=INTERLACED if main_interlaced else PROGRESSIVE,
                        ref_frames=INTERLACED if ref_interlaced else PROGRESSIVE)
        with pytest.raises(UnsupportedFramerateError) as error:
            run._applyDeinterlaceFilters(run.ffmpegQos)
        assert str(error.value) == ("No deinterlace filter available for the given framerate combination. "
                                    f"{message}. Consider using the -fps flag to force a frame rate manually.")


class TestNormalizeChains:
    """vmaf._normalizeChains: deinterlace or fps first, then scale, on both chains."""

    def test_deinterlace_before_scale(self, probes):
        run = make_vmaf(probes, main=stream(width=720, height=576), main_frames=INTERLACED)
        assert run._normalizeChains(run.ffmpegQos) == 30.0
        assert filters(run.ffmpegQos) == (chain(0, YADIF_FRAME, 'scale=1920:1080:flags=bicubic'), [])

    def test_fps_before_scale(self, probes):
        run = make_vmaf(probes, ref=stream(width=1280, height=720))
        assert run._normalizeChains(run.ffmpegQos) == 30.0
        assert filters(run.ffmpegQos) == (chain(0, 'fps=fps=30.0'),
                                          chain(1, 'fps=fps=30.0', 'scale=1920:1080:flags=bicubic'))

    def test_manual_fps(self, probes):
        run = make_vmaf(probes, manual_fps=25.0, main=stream(width=720, height=576), main_frames=INTERLACED)
        assert run._normalizeChains(run.ffmpegQos) == 25.0
        assert filters(run.ffmpegQos) == (chain(0, 'fps=fps=25.0', 'scale=1920:1080:flags=bicubic'),
                                          chain(1, 'fps=fps=25.0'))
        assert [c for c in probes.calls if c[0] == 'frames'] == []

    def test_inverted(self, probes):
        run = make_vmaf(probes, main=stream(width=720, height=576), main_frames=INTERLACED)
        qos = FFmpegQos('ref.mp4', 'main.mp4')
        qos.invertedSrc = True
        run._normalizeChains(qos)
        assert filters(qos) == ([], chain(1, YADIF_FRAME, 'scale=1920:1080:flags=bicubic'))


class TestPixelFormat:
    """vmaf._measurementPixFmt and _applyPixelFormat: the chroma of the reference, 10 bits with v1."""

    @pytest.mark.parametrize("ref_pix_fmt, kwargs, expected", [
        ('yuv420p', {}, 'yuv420p10le'),
        ('yuv422p10le', {}, 'yuv422p10le'),
        ('yuv444p', {}, 'yuv444p10le'),
        ('yuv420p12le', {}, 'yuv420p12le'),
        ('nv12', {}, 'yuv420p10le'),
        ('yuv420p', {'vmaf_v0': True}, 'yuv420p'),
        ('yuv422p10le', {'vmaf_v0': True}, 'yuv422p10le'),
        ('yuv420p', {'vmaf_versions': ('1', '0.6')}, 'yuv420p10le'),
        ('yuv420p10le', {'bitdepth': '8'}, 'yuv420p'),
        ('yuv420p', {'bitdepth': '10', 'vmaf_v0': True}, 'yuv420p10le'),
    ], ids=["420-v1", "422-10", "444-v1", "420-12", "nv12", "420-v0", "422-10-v0", "both", "bitdepth-8", "bitdepth-10"])
    def test_measurement_pix_fmt(self, probes, ref_pix_fmt, kwargs, expected):
        run = make_vmaf(probes, ref=stream(pix_fmt=ref_pix_fmt), **kwargs)
        assert run._measurementPixFmt() == expected

    def test_unknown_reference_format_warns(self, probes, caplog):
        run = make_vmaf(probes, ref=stream(pix_fmt='rgb24'))
        with caplog.at_level('WARNING'):
            assert run._measurementPixFmt() == 'yuv420p10le'
        assert "Unrecognized reference pixel format rgb24; measuring as 4:2:0" in caplog.text

    def test_format_filter_first(self, probes):
        run = make_vmaf(probes, ref=stream(pix_fmt='yuv422p10le', width=1280, height=720))
        run._applyPixelFormat()
        run._normalizeChains(run.ffmpegQos)
        assert run.pix_fmt == 'yuv422p10le'
        assert filters(run.ffmpegQos) == (chain(0, 'format=yuv422p10le', 'fps=fps=30.0'),
                                          chain(1, 'fps=fps=30.0', 'scale=1920:1080:flags=bicubic'))

    def test_no_filter_when_equal(self, probes):
        run = make_vmaf(probes, vmaf_v0=True)
        run._applyPixelFormat()
        assert (run.pix_fmt, filters(run.ffmpegQos)) == ('yuv420p', ([], []))


class TestDynamicRange:
    """vmaf._resolveDynamicRange, _colorConversion and _applyColorParams: SDR, PQ, mixed and the errors."""

    def test_sdr_by_default(self, probes, hdr_build):
        run = make_vmaf(probes)
        assert (run.dynamic_range, run.color) == ('sdr', None)
        run._applyColorParams()
        assert filters(run.ffmpegQos) == ([], [])

    def test_pq_inputs(self, probes, hdr_build):
        run = make_vmaf(probes, main=stream(pix_fmt='yuv420p10le', **PQ), ref=stream(pix_fmt='yuv420p10le', **PQ))
        assert run.dynamic_range == 'hdr'
        assert run.color == {'reference': {'range': 'tv', 'primaries': 'bt2020', 'trc': 'smpte2084', 'matrix': 'bt2020nc'},
                             'distorted': {'range': 'tv', 'primaries': 'bt2020', 'trc': 'smpte2084', 'matrix': 'bt2020nc'}}
        assert (run._colorConversion('distorted'), run._colorConversion('reference')) == (None, None)
        run._applyColorParams()
        run._applyPixelFormat()
        tags = 'setparams=range=tv:color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc'
        assert filters(run.ffmpegQos) == (chain(0, tags), chain(1, tags))

    def test_mixed_inputs(self, probes, hdr_build, caplog):
        with caplog.at_level('WARNING'):
            run = make_vmaf(probes, main=stream(**BT709), ref=stream(pix_fmt='yuv420p10le', **PQ))
        assert "Mixed inputs: the reference video is PQ and the distorted video is not" in caplog.text
        assert run._colorConversion('distorted') == ('bt2020nc', 'tv')
        assert run._colorConversion('reference') is None
        run._applyColorParams()
        run._applyPixelFormat()
        assert filters(run.ffmpegQos)[0] == chain(
            0, 'setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709',
            'format=yuv420p10le:color_spaces=bt2020nc:color_ranges=tv')

    def test_overrides_and_presets(self, probes, hdr_build):
        run = make_vmaf(probes, main=stream(pix_fmt='yuv420p10le'), ref=stream(pix_fmt='yuv420p10le', **PQ),
                        distorted_color='hdr10')
        assert run.dynamic_range == 'hdr' and run.color['distorted']['trc'] == 'smpte2084'
        run = make_vmaf(probes, main=stream(pix_fmt='yuv420p10le', **PQ),
                        ref=stream(pix_fmt='yuv420p10le', color_range='tv', color_primaries='bt2020', color_space='bt2020nc'),
                        reference_color='trc=smpte2084')
        assert run.color['reference'] == {'range': 'tv', 'primaries': 'bt2020', 'trc': 'smpte2084', 'matrix': 'bt2020nc'}

    def test_forced(self, probes, hdr_build):
        run = make_vmaf(probes, main=stream(**BT709), ref=stream(**BT709), dynamic_range='hdr')
        assert run.dynamic_range == 'hdr'
        run = make_vmaf(probes, main=stream(pix_fmt='yuv420p10le', **PQ), ref=stream(pix_fmt='yuv420p10le', **PQ),
                        dynamic_range='sdr')
        assert (run.dynamic_range, run.color) == ('sdr', None)

    @pytest.mark.parametrize("main, kwargs, message", [
        (stream(color_transfer='arib-std-b67'), {}, r"is HLG \(arib-std-b67\): HDR-VMAF only measures PQ"),
        (stream(color_transfer='smpte2084'), {}, r"does not report its color range.*-distorted_color range=tv"),
        (stream(**dict(PQ, color_space='fcc')), {}, r"has color matrix fcc, which the HDR-VMAF models do not support"),
        (stream(**PQ), {'vmaf_v0': True}, r"VMAF 0.6 has no HDR models"),
        (stream(**PQ), {'vmaf_versions': ('1', '0.6')}, r"VMAF 0.6 has no HDR models"),
    ], ids=["hlg", "missing", "unsupported", "v0", "both-versions"])
    def test_rejected(self, probes, hdr_build, main, kwargs, message):
        with pytest.raises(UnsupportedModelConfigError, match=message):
            make_vmaf(probes, main=main, ref=stream(pix_fmt='yuv420p10le', **PQ), **kwargs)

    def test_build_without_hdr(self, probes, hdr_build):
        hdr_build(False)
        with pytest.raises(UnsupportedModelConfigError, match=r"This FFmpeg build cannot compute the HDR-VMAF models"):
            make_vmaf(probes, main=stream(**PQ), ref=stream(**PQ))
        assert make_vmaf(probes, main=stream(**PQ), ref=stream(**PQ), dynamic_range='sdr').dynamic_range == 'sdr'


@pytest.fixture
def psnr_passes(monkeypatch):
    """FFmpegQos.getPsnr replaced by a fake: PSNR per REF trim start from `peaks`, else 20.0;
    `passes` lists the (main, ref, main filters, ref filters, ffmpeg threads) of each pass"""
    passes = []
    peaks = {}

    def getPsnr(qos):
        passes.append((qos.main.videoSrc, qos.ref.videoSrc, list(qos.main.filtersList), list(qos.ref.filtersList),
                       qos._ffmpeg_threads))
        start = float(re.search(r'trim=start=([0-9.]+):duration=0.5', ';'.join(qos.ref.filtersList)).group(1))
        for offset, psnr in peaks.items():
            if start == pytest.approx(offset):
                return psnr
        return 20.0
    monkeypatch.setattr(FFmpegQos, 'getPsnr', getPsnr)
    return SimpleNamespace(passes=passes, peaks=peaks)


def trim_starts(passes):
    return sorted(float(re.search(r'trim=start=([0-9.]+):duration=0.5', ';'.join(p[3])).group(1)) for p in passes)


class TestSync:
    """vmaf._computePsnrAtOffset and syncOffset: one PSNR pass per frame of the sync window, filtered like the run."""

    def test_pass(self, probes, psnr_passes):
        run = make_vmaf(probes, ref=stream(width=1280, height=720))
        assert run._computePsnrAtOffset(1.5, reverse=False) == (1.5, 20.0)
        assert psnr_passes.passes == [(
            'main.mp4', 'ref.mp4',
            chain(0, 'fps=fps=30.0', 'trim=start=0:duration=0.5, setpts=PTS-STARTPTS'),
            chain(1, 'trim=start=0.5', 'fps=fps=30.0', 'scale=1920:1080:flags=bicubic',
                  'trim=start=1.5:duration=0.5, setpts=PTS-STARTPTS'),
            1,
        )]

    def test_pass_reverse(self, probes, psnr_passes):
        run = make_vmaf(probes, ref=stream(width=1280, height=720))
        run._computePsnrAtOffset(1.5, reverse=True)
        assert psnr_passes.passes == [(
            'ref.mp4', 'main.mp4',
            chain(0, 'fps=fps=30.0', 'scale=1920:1080:flags=bicubic', 'trim=start=0:duration=0.5, setpts=PTS-STARTPTS'),
            chain(1, 'trim=start=0.5', 'fps=fps=30.0', 'trim=start=1.5:duration=0.5, setpts=PTS-STARTPTS'),
            1,
        )]

    def test_pass_interlaced_reverse(self, probes, psnr_passes):
        """The searched interlaced input is deinterlaced before its trim, so the offset lands on a field."""
        run = make_vmaf(probes, main=stream(r_frame_rate='15/1'), main_frames=INTERLACED)
        run._computePsnrAtOffset(0.5, reverse=True)
        assert psnr_passes.passes[0][2:4] == (
            chain(0, 'trim=start=0:duration=0.5, setpts=PTS-STARTPTS'),
            chain(1, YADIF_FIELD, 'trim=start=0.5:duration=0.5, setpts=PTS-STARTPTS'))

    def test_pass_manual_fps(self, probes, psnr_passes):
        run = make_vmaf(probes, manual_fps=25.0)
        run._computePsnrAtOffset(0.0, reverse=False)
        assert psnr_passes.passes[0][2:4] == (
            chain(0, 'fps=fps=25.0', 'trim=start=0:duration=0.5, setpts=PTS-STARTPTS'),
            chain(1, 'fps=fps=25.0', 'trim=start=0.0:duration=0.5, setpts=PTS-STARTPTS'))

    def test_best_offset(self, probes, psnr_passes):
        run = make_vmaf(probes, threads=2)
        psnr_passes.peaks.update({33 / 30: 41.5, 31 / 30: 30.0})
        offset, psnr = run.syncOffset(syncWindow=0.2, start=1)
        assert (offset, psnr) == (pytest.approx(1.1), 41.5)
        assert run.offset == offset
        assert trim_starts(psnr_passes.passes) == pytest.approx([30 / 30, 31 / 30, 32 / 30, 33 / 30, 34 / 30, 35 / 30])

    def test_tie_picks_lowest_offset(self, probes, psnr_passes):
        run = make_vmaf(probes, threads=3)
        assert run.syncOffset(syncWindow=0.2, start=0) == [0.0, 20.0]

    def test_reverse(self, probes, psnr_passes):
        run = make_vmaf(probes, threads=2)
        psnr_passes.peaks.update({2 / 30: 35.0})
        assert run.syncOffset(syncWindow=0.1, start=0, reverse=True) == [pytest.approx(-2 / 30), 35.0]
        assert run.offset == pytest.approx(-2 / 30)
        assert {p[:2] for p in psnr_passes.passes} == {('ref.mp4', 'main.mp4')}

    def test_window_past_end(self, probes, psnr_passes):
        run = make_vmaf(probes, main=stream(duration='10.000000'), ref=stream(duration='12.000000'))
        with pytest.raises(ValueError, match=r"The sync window ends at 13 s, but the reference video lasts 12 s"):
            run.syncOffset(syncWindow=5, start=8)
        with pytest.raises(ValueError, match=r"the distorted video lasts 10 s"):
            run.syncOffset(syncWindow=5, start=8, reverse=True)
        assert psnr_passes.passes == []

    def test_frames_probed_once(self, probes, psnr_passes):
        run = make_vmaf(probes, threads=2)
        run.syncOffset(syncWindow=0.2, start=0)
        assert sorted(c for c in probes.calls if c[0] == 'frames') == [('frames', 'main.mp4'), ('frames', 'ref.mp4')]

    def test_manual_fps_skips_frames_probe(self, probes, psnr_passes):
        run = make_vmaf(probes, threads=2, manual_fps=25.0)
        run.syncOffset(syncWindow=0.2, start=0)
        assert [c for c in probes.calls if c[0] == 'frames'] == []

    def test_failed_pass(self, probes, monkeypatch):
        def getPsnr(qos):
            raise subprocess.CalledProcessError(1, ['ffmpeg'])
        monkeypatch.setattr(FFmpegQos, 'getPsnr', getPsnr)
        run = make_vmaf(probes, threads=2)
        with pytest.raises(subprocess.CalledProcessError):
            run.syncOffset(syncWindow=0.2, start=0)


class TestSetOffset:
    """vmaf.setOffset: trim filters on the delayed input, cut to the duration both inputs share."""

    def make(self, probes):
        return make_vmaf(probes, main=stream(duration='10.000000'), ref=stream(duration='12.000000'))

    def test_positive(self, probes):
        run = self.make(probes)
        run.setOffset(1.5)
        assert run.offset == 1.5
        assert filters(run.ffmpegQos) == (chain(0, 'trim=start=0:duration=10.0, setpts=PTS-STARTPTS'),
                                          chain(1, 'trim=start=1.5:duration=10.0, setpts=PTS-STARTPTS'))

    def test_positive_ref_shorter(self, probes):
        run = self.make(probes)
        run.setOffset(4.0)
        assert filters(run.ffmpegQos) == (chain(0, 'trim=start=0:duration=8.0, setpts=PTS-STARTPTS'),
                                          chain(1, 'trim=start=4.0:duration=8.0, setpts=PTS-STARTPTS'))

    def test_negative(self, probes):
        run = self.make(probes)
        run.offset = -2.5
        run.setOffset()
        assert filters(run.ffmpegQos) == (chain(0, 'trim=start=2.5:duration=7.5, setpts=PTS-STARTPTS'),
                                          chain(1, 'trim=start=0:duration=7.5, setpts=PTS-STARTPTS'))

    def test_zero(self, probes):
        run = self.make(probes)
        run.setOffset(0)
        assert filters(run.ffmpegQos) == ([], [])
        assert run._syncTrims() is None


class TestFeatureString:
    """vmaf._build_feature_string: psnr always, cambi with -cambi_heatmap at the CAMBI accepted size."""

    def test_psnr(self, probes):
        assert make_vmaf(probes)._build_feature_string() == 'name=psnr'

    def test_cambi(self, probes):
        run = make_vmaf(probes, main=stream(width=1280, height=720), cambi_heatmap=True)
        assert run._build_feature_string() == (
            'name=psnr|name=cambi' + BS * 2 + ':full_ref=true' + BS * 2 + ':enc_width=1280' + BS * 2
            + ':enc_height=720' + BS * 2 + ':src_width=1920' + BS * 2 + ':src_height=1080')

    def test_cambi_enc_size_and_guard(self, probes):
        run = make_vmaf(probes, cambi_heatmap=True, enc_size='256x144')
        assert ':enc_width=267' + BS * 2 + ':enc_height=150' in run._build_feature_string()


@pytest.fixture
def vmaf_run(monkeypatch):
    """FFmpegQos.getVmaf replaced by a fake; `calls` lists (qos, keyword arguments) per run"""
    calls = []

    def getVmaf(qos, **kwargs):
        calls.append((qos, kwargs))
        qos.vmafpath = kwargs.get('log_path') or 'main_vmaf.json'
        return 'process'
    monkeypatch.setattr(FFmpegQos, 'getVmaf', getVmaf)
    return calls


class TestGetVmaf:
    """vmaf.getVmaf: filter order, model selection and the arguments of FFmpegQos.getVmaf."""

    def test_v1(self, probes, vmaf_run):
        run = make_vmaf(probes, main=stream(width=1280, height=720, pix_fmt='yuv420p10le', duration='10.000000'),
                        ref=stream(duration='10.000000'), subsample=2, threads=4, print_progress=True, end_sync=True)
        run.offset = 1.0
        assert run.getVmaf() == 'process'
        assert filters(run.ffmpegQos) == (
            chain(0, 'fps=fps=30.0', 'scale=1920:1080:flags=bicubic', 'trim=start=0:duration=9.0, setpts=PTS-STARTPTS'),
            chain(1, 'format=yuv420p10le', 'fps=fps=30.0', 'trim=start=1.0:duration=9.0, setpts=PTS-STARTPTS'))
        cambi = {'cambi.enc_width': '1280', 'cambi.enc_height': '720', 'cambi.enc_bitdepth': '10'}
        assert [(m.name, os.path.basename(m.path), m.params) for m in run.models] == [
            ('vmaf_v1_hd', 'vmaf_v1.0.16_3d0h.json', cambi), ('vmaf_v1_hd_phone', 'vmaf_v1.0.16_5d0h.json', cambi)]
        assert vmaf_run == [(run.ffmpegQos, {
            'log_path': None, 'models': run.models, 'subsample': 2, 'output_fmt': 'json', 'threads': 4,
            'print_progress': True, 'end_sync': True, 'features': 'name=psnr', 'cambi_heatmap': False,
            'heatmap_path': None})]
        assert (run.pix_fmt, run.output_fps, run.hfr_active, run.frames_scored) == ('yuv420p10le', 30.0, False, None)

    def test_v0(self, probes, vmaf_run):
        run = make_vmaf(probes, vmaf_v0=True, output_fmt='xml', cambi_heatmap=True)
        run.getVmaf()
        assert filters(run.ffmpegQos) == (chain(0, 'fps=fps=30.0'), chain(1, 'fps=fps=30.0'))
        assert [(m.name, m.version, m.params) for m in run.models] == [
            ('vmaf_hd', 'vmaf_v0.6.1', {}), ('vmaf_hd_neg', 'vmaf_v0.6.1neg', {}),
            ('vmaf_hd_phone', 'vmaf_v0.6.1', {'enable_transform': 'true'})]
        kwargs = vmaf_run[0][1]
        assert (kwargs['output_fmt'], kwargs['cambi_heatmap'], kwargs['heatmap_path']) == ('xml', True, 'main_cambi_heatmap')
        assert kwargs['features'].startswith('name=psnr|name=cambi')
        assert run.pix_fmt == 'yuv420p'

    @pytest.mark.parametrize("rate, kwargs, names", [
        ('60/1', {}, ['vmaf_v1_hd_hfr', 'vmaf_v1_hd_phone_hfr']),
        ('60/1', {'disable_hfr': True}, ['vmaf_v1_hd', 'vmaf_v1_hd_phone']),
        ('30/1', {}, ['vmaf_v1_hd', 'vmaf_v1_hd_phone']),
        ('30/1', {'hfr': 'on'}, ['vmaf_v1_hd_hfr', 'vmaf_v1_hd_phone_hfr']),
    ], ids=["hfr", "hfr-disabled", "standard", "forced"])
    def test_hfr(self, probes, vmaf_run, rate, kwargs, names):
        run = make_vmaf(probes, main=stream(r_frame_rate=rate), ref=stream(r_frame_rate=rate), **kwargs)
        run.getVmaf()
        assert [m.name for m in run.models] == names
        assert run.hfr_active is names[0].endswith('_hfr')

    def test_4k(self, probes, vmaf_run):
        run = make_vmaf(probes, model='4K')
        run.getVmaf()
        assert [m.name for m in run.models] == ['vmaf_v1_4k', 'vmaf_v1_4k_3h']
        assert run.ffmpegQos.main.filtersList[:3] == chain(0, 'format=yuv420p10le', 'fps=fps=30.0',
                                                           'scale=3840:2160:flags=bicubic')

    def test_views_and_versions(self, probes, vmaf_run):
        run = make_vmaf(probes, views=['phone'], vmaf_versions=('1', '0.6'))
        run.getVmaf()
        assert [m.name for m in run.models] == ['vmaf_v1_hd_phone', 'vmaf_hd', 'vmaf_hd_neg', 'vmaf_hd_phone']

    def test_hdr(self, probes, vmaf_run, hdr_build):
        run = make_vmaf(probes, main=stream(pix_fmt='yuv420p10le', **PQ), ref=stream(pix_fmt='yuv420p10le', **PQ))
        run.getVmaf()
        tags = 'setparams=range=tv:color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc'
        assert filters(run.ffmpegQos) == (chain(0, tags, 'fps=fps=30.0'), chain(1, tags, 'fps=fps=30.0'))
        assert [m.name for m in run.models] == ['hdr_vmaf_v1_hd', 'hdr_vmaf_v1_hd_phone']
        assert run.models[0].params['cambi.enc_bitdepth'] == '10'

    def test_manual_fps(self, probes, vmaf_run):
        run = make_vmaf(probes, manual_fps=25.0)
        run.getVmaf()
        assert filters(run.ffmpegQos) == (chain(0, 'format=yuv420p10le', 'fps=fps=25.0'),
                                          chain(1, 'format=yuv420p10le', 'fps=fps=25.0'))

    def test_second_run_starts_clean(self, probes, vmaf_run):
        run = make_vmaf(probes)
        run.getVmaf()
        first = filters(run.ffmpegQos)
        run.ffmpegQos.psnrFilter = ['PSNR']
        run.getVmaf()
        assert filters(run.ffmpegQos) == first
        assert run.ffmpegQos.psnrFilter == []

    def test_auto_sync(self, probes, vmaf_run, monkeypatch):
        run = make_vmaf(probes)
        synced = []
        monkeypatch.setattr(run, 'syncOffset', lambda: synced.append(True))
        run.getVmaf(autoSync=True)
        assert synced == [True]


class TestApplyOffsetFilters:
    """vmaf._applyOffsetFilters: the trim filters of setOffset in any FFmpegQos."""

    def test_given_qos(self, probes):
        run = make_vmaf(probes, main=stream(duration='10.000000'), ref=stream(duration='12.000000'))
        run.offset = -2.5
        qos = FFmpegQos('main.mp4', 'ref.mp4')
        run._applyOffsetFilters(qos)
        assert filters(qos) == (chain(0, 'trim=start=2.5:duration=7.5, setpts=PTS-STARTPTS'),
                                chain(1, 'trim=start=0:duration=7.5, setpts=PTS-STARTPTS'))
        assert filters(run.ffmpegQos) == ([], [])


@pytest.fixture
def first_frames(monkeypatch):
    """FFmpegQos.getFirstFrameTimestamps replaced by a fake returning the given anchors; records the probe chains"""
    def install(main=(0, Fraction(1, 30)), ref=(0, Fraction(1, 30))):
        calls = []

        def getFirstFrameTimestamps(qos):
            calls.append((list(qos.main.filtersList), list(qos.ref.filtersList), qos._commitInputs()))
            return main, ref
        monkeypatch.setattr(FFmpegQos, 'getFirstFrameTimestamps', getFirstFrameTimestamps)
        return calls
    return install


class TestRange:
    """vmaf._rangeBounds, _applyRange, _rangeLogPath and _trimRangeLog."""

    @pytest.mark.parametrize("start, count, bounds, suffix", [
        (10, 100, (8, 112), '_f10-109'), (0, None, (0, None), '_f0-end'), (1, 5, (0, 8), '_f1-5'), (None, 3, (0, 5), '_f0-2'),
    ], ids=["10-100", "0-end", "1-5", "none-3"])
    def test_bounds_and_names(self, probes, start, count, bounds, suffix):
        run = make_vmaf(probes, start_frame=start, frame_count=count, output_fmt='xml')
        assert run._rangeBounds() == bounds
        assert run._rangeSuffix() == suffix
        assert run._rangeLogPath() == f'main_vmaf{suffix}.xml'
        assert run._cambiHeatmapPath() == f'main_cambi_heatmap{suffix}'

    def test_no_range(self, probes):
        run = make_vmaf(probes)
        assert (run._rangeSuffix(), run._rangeLogPath(), run._cambiHeatmapPath()) == ('', 'main_vmaf.json', 'main_cambi_heatmap')

    def test_apply_range_unsynced(self, probes, first_frames):
        calls = first_frames()
        run = make_vmaf(probes, main=stream(duration='10.000000'), ref=stream(duration='10.000000'),
                        start_frame=10, frame_count=100)
        run.ffmpegQos.clearFilters()
        run.output_fps = run._normalizeChains(run.ffmpegQos)
        run._applyRange()
        assert calls[0][0] == chain(0, 'fps=fps=30.0')
        assert calls[0][2][:4] == ['-copyts', '-start_at_zero', '-hwaccel', 'auto']
        assert '-ss' in calls[0][2]
        assert filters(run.ffmpegQos) == (
            chain(0, 'fps=fps=30.0', 'setpts=PTS-0', 'trim=start=0.25833333333333336:end=3.725, setpts=PTS-STARTPTS'),
            chain(1, 'fps=fps=30.0', 'setpts=PTS-0', 'trim=start=0.25833333333333336:end=3.725, setpts=PTS-STARTPTS'))
        assert run.ffmpegQos.main.seekOptions == ['-ss', '0.000000']

    def test_apply_range_synced(self, probes, first_frames):
        first_frames(main=(0, Fraction(1, 30)), ref=(45, Fraction(1, 30)))
        run = make_vmaf(probes, main=stream(duration='10.000000'), ref=stream(duration='12.000000'),
                        start_frame=100, frame_count=50)
        run.offset = 1.5
        run.ffmpegQos.clearFilters()
        run.output_fps = run._normalizeChains(run.ffmpegQos)
        run._applyRange()
        assert filters(run.ffmpegQos) == (
            chain(0, 'fps=fps=30.0', 'trim=start=0', 'setpts=PTS-0', 'trim=end=10.0',
                  'trim=start=3.2583333333333333:end=5.058333333333334, setpts=PTS-STARTPTS'),
            chain(1, 'fps=fps=30.0', 'trim=start=1.5', 'setpts=PTS-45', 'trim=end=10.0',
                  'trim=start=3.2583333333333333:end=5.058333333333334, setpts=PTS-STARTPTS'))
        assert run.ffmpegQos.main.seekOptions == ['-ss', '2.266667']
        assert run.ffmpegQos.ref.seekOptions == ['-ss', '3.766667']

    def test_apply_range_rejects_different_starts(self, probes, first_frames):
        first_frames(main=(0, Fraction(1, 30)), ref=(15, Fraction(1, 30)))
        run = make_vmaf(probes, start_frame=10)
        run.ffmpegQos.clearFilters()
        run.output_fps = run._normalizeChains(run.ffmpegQos)
        with pytest.raises(UnsupportedRangeError, match=r"start at different timestamps \(0 s and 0.5 s\)"):
            run._applyRange()

    def test_trim_range_log(self, probes, tmp_path):
        log = str(tmp_path / 'main_vmaf_f1-2.json')
        shutil.copy(os.path.join(DATA, 'vmaf_v1_hd.json'), log)
        run = make_vmaf(probes, main=stream(duration='0.200000'), ref=stream(duration='0.200000'),
                        start_frame=1, frame_count=2)
        run.output_fps = 25.0
        assert run._trimRangeLog(log) == 2
        assert [n for n, _ in read_frames(log, 'json')] == [1, 2]

    def test_trim_range_log_after_last_frame(self, probes, tmp_path):
        log = str(tmp_path / 'main_vmaf_f9-end.json')
        shutil.copy(os.path.join(DATA, 'vmaf_v1_hd.json'), log)
        Vmaf.trim_log(log, 'json', 0, 1, 0)   # a run from frame 7 that measured one frame
        run = make_vmaf(probes, main=stream(duration='0.200000'), ref=stream(duration='0.200000'), start_frame=9)
        run.output_fps = 25.0
        with pytest.raises(UnsupportedRangeError, match=r"The range starts at frame 9, after the last measured frame \(7\)"):
            run._trimRangeLog(log)

    def test_trim_range_log_missing_frames(self, probes, tmp_path):
        log = str(tmp_path / 'main_vmaf_f0-99.json')
        shutil.copy(os.path.join(DATA, 'vmaf_v1_hd.json'), log)
        run = make_vmaf(probes, main=stream(duration='10.000000'), ref=stream(duration='10.000000'),
                        start_frame=0, frame_count=100)
        run.output_fps = 25.0
        with pytest.raises(UnsupportedRangeError, match=r"measured 5 frames from frame 0 instead of 102"):
            run._trimRangeLog(log)


class TestJobs:
    """vmaf._planJobs, _jobRange and _computeJobs with a fake run per part."""

    @pytest.mark.parametrize("duration, kwargs, expected", [
        ('10.000000', {'jobs': 4}, [(0, 74), (74, 74), (148, 74), (222, None)]),
        ('10.000000', {'jobs': 2, 'start_frame': 50, 'frame_count': 100}, [(50, 50), (100, 50)]),
        ('0.200000', {'jobs': 4}, [(None, None)]),
    ], ids=["four", "range", "too-short"])
    def test_plan(self, probes, duration, kwargs, expected):
        run = make_vmaf(probes, main=stream(duration=duration), ref=stream(duration=duration), **kwargs)
        assert run._planJobs() == expected
        assert run.output_fps == 30.0
        assert filters(run.ffmpegQos) == ([], [])

    def test_plan_with_offset(self, probes):
        run = make_vmaf(probes, main=stream(duration='10.000000'), ref=stream(duration='12.000000'), jobs=2)
        run.offset = 2.0
        assert run._planJobs() == [(0, 149), (149, None)]

    def test_job_range(self, probes):
        run = make_vmaf(probes, jobs=2, threads=0)
        part = run._jobRange(10, 20, 3, 2)
        assert (part.jobs, part._jobPart, part.start_frame, part.frame_count, part.threads, part.print_progress) == (
            1, True, 10, 20, 3, False)
        assert part.ffmpegQos is not run.ffmpegQos and part.ffmpegQos._ffmpeg_threads == 2
        assert part.main is run.main and part._rangeSuffix() == '_f10-29_part'

    def _fake_parts(self, monkeypatch, tmp_path, failing=None):
        """getVmaf of a part writes a cut of the captured log; the shared run keeps the real getVmaf"""
        original = vmaf.getVmaf
        cancelled = []

        def getVmaf(self, autoSync=False):
            if self.jobs > 1:
                return original(self, autoSync)
            if failing is not None and self.start_frame == failing:
                raise subprocess.CalledProcessError(1, ['ffmpeg'])
            log = self._rangeLogPath()
            shutil.copy(os.path.join(DATA, 'vmaf_v1_hd.json'), log)
            self.ffmpegQos.vmafpath = log
            self.frames_scored = self.frame_count if self.frame_count else 5 - self.start_frame
            Vmaf.trim_log(log, 'json', self.start_frame, self.frames_scored, 0)
            self.pix_fmt, self.output_fps, self.features = 'yuv420p10le', 25.0, 'name=psnr'
            return None
        monkeypatch.setattr(vmaf, 'getVmaf', getVmaf)
        monkeypatch.setattr(vmaf, '_planJobs', lambda self: [(0, 2), (2, None)])
        monkeypatch.setattr(FFmpegQos, 'cancel', lambda qos: cancelled.append(qos))
        return cancelled

    def test_compute_jobs_joins(self, probes, monkeypatch, tmp_path):
        self._fake_parts(monkeypatch, tmp_path)
        src = str(tmp_path / 'main.mp4')
        probes.answers[src] = probes.answers['main.mp4'] = {'stream': stream(duration='0.200000'), 'frames': PROGRESSIVE, 'format': MP4_FORMAT}
        probes.answers['ref.mp4'] = {'stream': stream(duration='0.200000'), 'frames': PROGRESSIVE, 'format': MP4_FORMAT}
        run = vmaf(src, 'ref.mp4', 'json', jobs=2, start_frame=0, frame_count=5)
        assert run.getVmaf() is None
        assert run.ffmpegQos.vmafpath == str(tmp_path / 'main_vmaf_f0-4.json')
        assert [n for n, _ in read_frames(run.ffmpegQos.vmafpath, 'json')] == [0, 1, 2, 3, 4]
        assert (run.frames_scored, run.pix_fmt, run.output_fps) == (5, 'yuv420p10le', 25.0)
        assert [m.name for m in run.models] == ['vmaf_v1_hd', 'vmaf_v1_hd_phone']
        assert sorted(os.listdir(tmp_path)) == ['main_vmaf_f0-4.json']

    def test_compute_jobs_failure(self, probes, monkeypatch, tmp_path):
        cancelled = self._fake_parts(monkeypatch, tmp_path, failing=2)
        src = str(tmp_path / 'main.mp4')
        probes.answers[src] = {'stream': stream(duration='0.200000'), 'frames': PROGRESSIVE, 'format': MP4_FORMAT}
        probes.answers['ref.mp4'] = {'stream': stream(duration='0.200000'), 'frames': PROGRESSIVE, 'format': MP4_FORMAT}
        run = vmaf(src, 'ref.mp4', 'json', jobs=2)
        with pytest.raises(subprocess.CalledProcessError):
            run.getVmaf()
        assert len(cancelled) == 2
        assert os.listdir(tmp_path) == []


@pytest.fixture
def frames_run(monkeypatch):
    """FFmpegQos.getFrames replaced by a fake returning `written`; returns the calls as (qos, frame numbers, folder)"""
    def install(written):
        calls = []

        def getFrames(qos, frame_numbers, folder):
            calls.append((qos, frame_numbers, folder))
            return written
        monkeypatch.setattr(FFmpegQos, 'getFrames', getFrames)
        return calls
    return install


class TestGetFrames:
    """vmaf.getFrames: the frames chain without scale and format filters, the frame times, and the HDR conversion."""

    def test_chain_and_times(self, probes, frames_run):
        calls = frames_run([(207, '/w/a.tif', 6.9), (209, '/w/b.tif', 6.966667)])
        run = make_vmaf(probes, main=stream(width=1280, height=720, duration='10.000000'),
                        ref=stream(duration='12.000000'))
        run.offset = -1.0
        assert run.getFrames([207, 209], '/w') == [(207, '/w/a.tif', pytest.approx(7.9)),
                                                   (209, '/w/b.tif', pytest.approx(7.966667))]
        qos, numbers, folder = calls[0]
        assert (numbers, folder) == ([207, 209], '/w')
        assert filters(qos) == (chain(0, 'fps=fps=30.0', 'trim=start=1.0:duration=9.0, setpts=PTS-STARTPTS'),
                                chain(1, 'fps=fps=30.0', 'trim=start=0:duration=9.0, setpts=PTS-STARTPTS'))

    def test_positive_offset_and_manual_fps(self, probes, frames_run):
        calls = frames_run([(25, '/w/a.tif', 1.0)])
        run = make_vmaf(probes, main=stream(duration='10.000000'), ref=stream(duration='12.000000'),
                        manual_fps=25.0)
        run.offset = 1.5
        assert run.getFrames([25], '/w') == [(25, '/w/a.tif', 1.0)]
        assert filters(calls[0][0]) == (
            chain(0, 'fps=fps=25.0', 'trim=start=0:duration=10.0, setpts=PTS-STARTPTS'),
            chain(1, 'fps=fps=25.0', 'trim=start=1.5:duration=10.0, setpts=PTS-STARTPTS'))

    def test_hdr_conversion(self, probes, frames_run, hdr_build):
        calls = frames_run([(1, '/w/a.tif', 0.04)])
        run = make_vmaf(probes, main=stream(width=1280, height=720, pix_fmt='yuv420p10le', **PQ),
                        ref=stream(pix_fmt='yuv420p10le', **PQ))
        run.getFrames([1], '/w')
        assert filters(calls[0][0])[0] == chain(
            0, 'fps=fps=30.0', 'setparams=range=tv:color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc',
            'scale=1280:720:out_primaries=bt709:out_transfer=bt709:out_color_matrix=bt709:out_range=tv')
