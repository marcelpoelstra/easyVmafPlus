"""Tests for Vmaf.py: the stream info and duration of video, and the scale, deinterlace, frame rate,
sync and trim filters and the VMAF run of vmaf, with ffprobe and ffmpeg replaced by fakes.
The stream, format and frames info in tests/data are ffprobe outputs."""

import json
import logging
import os
import subprocess
from types import SimpleNamespace

import pytest

from easyVmafPlus import Vmaf
from easyVmafPlus.FFmpeg import FFmpegQos
from easyVmafPlus.Vmaf import FeatureConfig, UnsupportedFramerateError, getFrameRate, video, vmaf

DATA = os.path.join(os.path.dirname(__file__), 'data')
BS = "\\"


def _json(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


STREAM = _json('ffprobe_streams_mp4.json')['streams'][0]          # h264 1920x1080, 30/1, 2.066667 s
MKV_STREAM = _json('ffprobe_streams_mkv.json')['streams'][0]      # the same stream in Matroska: no duration
MKV_FORMAT = _json('ffprobe_format_mkv.json')['format']
PROGRESSIVE = _json('ffprobe_frames_progressive.json')['frames']  # 3 frames, interlaced_frame 0
INTERLACED = _json('ffprobe_frames_interlaced.json')['frames']    # 3 frames, interlaced_frame 1
PACKETS = _json('ffprobe_packets_mp4.json')['packets']


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


def make_vmaf(probes, main=STREAM, ref=STREAM, main_frames=PROGRESSIVE, ref_frames=PROGRESSIVE, **kwargs):
    """A vmaf for main.mp4 and ref.mp4 with the given stream and frames info"""
    probes.answers['main.mp4'] = {'stream': main, 'frames': main_frames}
    probes.answers['ref.mp4'] = {'stream': ref, 'frames': ref_frames}
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


def test_unsupported_framerate_error_is_value_error():
    assert issubclass(UnsupportedFramerateError, ValueError)


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
    """vmaf.__init__: a video and an FFmpegQos for MAIN and REF, and the target resolution of the model."""

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
        assert (run.offset, run.models) == (0, [])
        assert probes.calls == [('stream', 'main.mp4'), ('stream', 'ref.mp4')]


class TestScale:
    """vmaf._applyScaleFilters and _autoScale: inputs not at the target resolution are scaled to it."""

    def test_at_target(self, probes):
        run = make_vmaf(probes)
        run._autoScale()
        assert filters(run.ffmpegQos) == ([], [])

    def test_ref_scaled(self, probes):
        run = make_vmaf(probes, ref=stream(width=1280, height=720))
        run._autoScale()
        assert filters(run.ffmpegQos) == ([], ['[1:v]scale=1920:1080:flags=bicubic[input1_0]'])

    def test_main_scaled(self, probes):
        run = make_vmaf(probes, main=stream(width=720, height=576))
        run._autoScale()
        assert filters(run.ffmpegQos) == (['[0:v]scale=1920:1080:flags=bicubic[input0_0]'], [])

    def test_4k(self, probes):
        run = make_vmaf(probes, model='4K')
        run._autoScale()
        assert filters(run.ffmpegQos) == (['[0:v]scale=3840:2160:flags=bicubic[input0_0]'],
                                          ['[1:v]scale=3840:2160:flags=bicubic[input1_0]'])

    def test_inverted(self, probes):
        run = make_vmaf(probes, ref=stream(width=1280, height=720))
        qos = FFmpegQos('ref.mp4', 'main.mp4')
        qos.invertedSrc = True
        run._applyScaleFilters(qos)
        assert filters(qos) == (['[0:v]scale=1920:1080:flags=bicubic[input0_0]'], [])

    def test_second_call_warns(self, probes, caplog):
        run = make_vmaf(probes)
        run._autoScale()
        with caplog.at_level(logging.WARNING, logger='easyVmafPlus.Vmaf'):
            run._autoScale()
        assert caplog.messages == ["_autoScale() called again without clearFilters(); "
                                   "the filter chains can hold duplicate filters"]


YADIF_FRAME = 'yadif=0:-1:0'
YADIF_FIELD = 'yadif=1:-1:0'


def chain(input_id, *filter_bodies):
    """Filter strings as inputFFmpeg labels them: [<id>:v]f0[input<id>_0], [input<id>_0]f1[input<id>_1], ..."""
    result = []
    for n, body in enumerate(filter_bodies):
        source = f'{input_id}:v' if n == 0 else f'input{input_id}_{n - 1}'
        result.append(f'[{source}]{body}[input{input_id}_{n}]')
    return result


class TestDeinterlace:
    """vmaf._applyDeinterlaceFilters: the frame rate and deinterlace table of Vmaf.py."""

    @pytest.mark.parametrize("ref_rate, ref_interlaced, main_rate, main_interlaced, main_filters, ref_filters", [
        ('25/1', False, '50/1', False, ['fps=fps=25.0'], []),
        ('50/1', False, '25/1', False, [], ['fps=fps=25.0']),
        ('30/1', False, '30/1', False, ['fps=fps=30.0'], ['fps=fps=30.0']),
        ('30000/1001', False, '30000/1001', False, ['fps=fps=29.97003'], ['fps=fps=29.97003']),
        ('50/1', True, '50/1', True, ['fps=fps=50.0'], ['fps=fps=50.0']),
        ('60/1', True, '30/1', False, [], [YADIF_FRAME]),
        ('60000/1001', True, '30/1', False, [], [YADIF_FRAME, 'fps=fps=30.0']),
        ('30/1', True, '30/1', False, [], [YADIF_FRAME]),
        ('25/1', True, '50/1', False, [], [YADIF_FIELD]),
        ('50/1', False, '25/1', True, [YADIF_FIELD, 'fps=fps=25.0'], []),
        ('30/1', False, '30/1', True, [YADIF_FRAME], []),
        ('30/1', False, '60/1', True, [YADIF_FIELD], []),
    ], ids=["progressive-ref-lower", "progressive-main-lower", "progressive-equal", "progressive-29.97",
            "both-interlaced", "ref-interlaced-2x", "ref-interlaced-2x-59.94", "ref-interlaced-1x",
            "ref-interlaced-half", "main-interlaced-2x", "main-interlaced-1x", "main-interlaced-half"])
    def test_rule(self, probes, ref_rate, ref_interlaced, main_rate, main_interlaced, main_filters, ref_filters):
        run = make_vmaf(probes, main=stream(r_frame_rate=main_rate), ref=stream(r_frame_rate=ref_rate),
                        main_frames=INTERLACED if main_interlaced else PROGRESSIVE,
                        ref_frames=INTERLACED if ref_interlaced else PROGRESSIVE)
        run._autoDeinterlace()
        assert filters(run.ffmpegQos) == (chain(0, *main_filters), chain(1, *ref_filters))

    @pytest.mark.parametrize("ref_rate, ref_interlaced, main_rate, main_interlaced, main_filters, ref_filters", [
        ('60/1', True, '30/1', False, [YADIF_FRAME], []),
        ('25/1', True, '50/1', False, [YADIF_FIELD], []),
        ('30/1', False, '30/1', True, [], [YADIF_FRAME]),
        ('30/1', False, '60/1', True, [], [YADIF_FIELD]),
    ], ids=["ref-interlaced-2x", "ref-interlaced-half", "main-interlaced-1x", "main-interlaced-half"])
    def test_inverted(self, probes, ref_rate, ref_interlaced, main_rate, main_interlaced, main_filters, ref_filters):
        run = make_vmaf(probes, main=stream(r_frame_rate=main_rate), ref=stream(r_frame_rate=ref_rate),
                        main_frames=INTERLACED if main_interlaced else PROGRESSIVE,
                        ref_frames=INTERLACED if ref_interlaced else PROGRESSIVE)
        qos = FFmpegQos('ref.mp4', 'main.mp4')
        qos.invertedSrc = True
        run._applyDeinterlaceFilters(qos)
        assert filters(qos) == (chain(0, *main_filters), chain(1, *ref_filters))

    @pytest.mark.parametrize("ref_rate, ref_interlaced, main_rate, main_interlaced, message", [
        ('25/1', True, '30/1', False, "ref=25.0fps (interlaced=True), main=30.0fps (interlaced=False)"),
        ('25/1', False, '30/1', True, "ref=25.0fps (interlaced=False), main=30.0fps (interlaced=True)"),
    ], ids=["ref-interlaced", "main-interlaced"])
    def test_unsupported(self, probes, ref_rate, ref_interlaced, main_rate, main_interlaced, message):
        run = make_vmaf(probes, main=stream(r_frame_rate=main_rate), ref=stream(r_frame_rate=ref_rate),
                        main_frames=INTERLACED if main_interlaced else PROGRESSIVE,
                        ref_frames=INTERLACED if ref_interlaced else PROGRESSIVE)
        with pytest.raises(UnsupportedFramerateError) as error:
            run._autoDeinterlace()
        assert str(error.value) == ("No deinterlace filter available for the given framerate combination. "
                                    f"{message}. Consider using the -fps flag to force a frame rate manually.")

    def test_force_fps(self, probes):
        run = make_vmaf(probes, manual_fps=25.0)
        run._forceFps()
        assert filters(run.ffmpegQos) == (chain(0, 'fps=fps=25.0'), chain(1, 'fps=fps=25.0'))


@pytest.fixture
def psnr_passes(monkeypatch):
    """FFmpegQos.getPsnr replaced by a fake: PSNR per REF trim start from `peaks`, else 20.0;
    `passes` lists the (main, ref, main filters, ref filters) of each pass"""
    passes = []
    peaks = {}

    def getPsnr(qos):
        passes.append((qos.main.videoSrc, qos.ref.videoSrc, list(qos.main.filtersList), list(qos.ref.filtersList)))
        start = float(qos.ref.filtersList[0].split('trim=start=')[1].split(':')[0])
        for offset, psnr in peaks.items():
            if start == pytest.approx(offset):
                return psnr
        return 20.0
    monkeypatch.setattr(FFmpegQos, 'getPsnr', getPsnr)
    return SimpleNamespace(passes=passes, peaks=peaks)


class TestSync:
    """vmaf._computePsnrAtOffset and syncOffset: one PSNR pass per frame of the sync window."""

    def test_pass(self, probes, psnr_passes):
        run = make_vmaf(probes, ref=stream(width=1280, height=720))
        assert run._computePsnrAtOffset(1.5, reverse=False) == (1.5, 20.0)
        assert psnr_passes.passes == [(
            'main.mp4', 'ref.mp4',
            chain(0, 'trim=start=0:duration=0.5, setpts=PTS-STARTPTS', 'fps=fps=30.0'),
            chain(1, 'trim=start=1.5:duration=0.5, setpts=PTS-STARTPTS', 'scale=1920:1080:flags=bicubic',
                  'fps=fps=30.0'),
        )]

    def test_pass_reverse(self, probes, psnr_passes):
        run = make_vmaf(probes, ref=stream(width=1280, height=720))
        run._computePsnrAtOffset(1.5, reverse=True)
        assert psnr_passes.passes == [(
            'ref.mp4', 'main.mp4',
            chain(0, 'trim=start=0:duration=0.5, setpts=PTS-STARTPTS', 'scale=1920:1080:flags=bicubic',
                  'fps=fps=30.0'),
            chain(1, 'trim=start=1.5:duration=0.5, setpts=PTS-STARTPTS', 'fps=fps=30.0'),
        )]

    def test_pass_manual_fps(self, probes, psnr_passes):
        run = make_vmaf(probes, manual_fps=25.0)
        run._computePsnrAtOffset(0.0, reverse=False)
        assert psnr_passes.passes[0][2:] == (
            chain(0, 'trim=start=0:duration=0.5, setpts=PTS-STARTPTS', 'fps=fps=25.0'),
            chain(1, 'trim=start=0.0:duration=0.5, setpts=PTS-STARTPTS', 'fps=fps=25.0'))

    def test_best_offset(self, probes, psnr_passes):
        run = make_vmaf(probes, threads=2)
        psnr_passes.peaks.update({33 / 30: 41.5, 31 / 30: 30.0})
        offset, psnr = run.syncOffset(syncWindow=0.2, start=1)
        assert (offset, psnr) == (pytest.approx(1.1), 41.5)
        assert run.offset == offset
        starts = sorted(float(p[3][0].split('trim=start=')[1].split(':')[0]) for p in psnr_passes.passes)
        assert starts == pytest.approx([30 / 30, 31 / 30, 32 / 30, 33 / 30, 34 / 30, 35 / 30])

    def test_tie_picks_lowest_offset(self, probes, psnr_passes):
        run = make_vmaf(probes, threads=3)
        assert run.syncOffset(syncWindow=0.2, start=0) == [0.0, 20.0]

    def test_reverse(self, probes, psnr_passes):
        run = make_vmaf(probes, threads=2)
        psnr_passes.peaks.update({2 / 30: 35.0})
        assert run.syncOffset(syncWindow=0.1, start=0, reverse=True) == [pytest.approx(-2 / 30), 35.0]
        assert run.offset == pytest.approx(-2 / 30)
        assert {p[:2] for p in psnr_passes.passes} == {('ref.mp4', 'main.mp4')}

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


class TestFeatureString:
    """vmaf._build_feature_string: psnr always, cambi with -cambi_heatmap."""

    def test_psnr(self, probes):
        assert make_vmaf(probes)._build_feature_string() == 'name=psnr'

    def test_cambi(self, probes):
        run = make_vmaf(probes, main=stream(width=1280, height=720), cambi_heatmap=True)
        assert run._build_feature_string() == (
            'name=psnr|name=cambi' + BS * 2 + ':full_ref=true' + BS * 2 + ':enc_width=1280' + BS * 2
            + ':enc_height=720' + BS * 2 + ':src_width=1920' + BS * 2 + ':src_height=1080')


@pytest.fixture
def vmaf_run(monkeypatch):
    """FFmpegQos.getVmaf replaced by a fake; `calls` lists (qos, keyword arguments) per run"""
    calls = []

    def getVmaf(qos, **kwargs):
        calls.append((qos, kwargs))
        return 'process'
    monkeypatch.setattr(FFmpegQos, 'getVmaf', getVmaf)
    return calls


class TestGetVmaf:
    """vmaf.getVmaf: filter order, model selection and the arguments of FFmpegQos.getVmaf."""

    def test_v1(self, probes, vmaf_run):
        run = make_vmaf(probes, main=stream(width=1280, height=720, bits_per_raw_sample='10', duration='10.000000'),
                        ref=stream(duration='10.000000'), subsample=2, threads=4, print_progress=True, end_sync=True)
        run.offset = 1.0
        assert run.getVmaf() == 'process'
        assert filters(run.ffmpegQos) == (
            chain(0, 'scale=1920:1080:flags=bicubic', 'fps=fps=30.0', 'trim=start=0:duration=9.0, setpts=PTS-STARTPTS',
                  'format=yuv420p10le'),
            chain(1, 'fps=fps=30.0', 'trim=start=1.0:duration=9.0, setpts=PTS-STARTPTS', 'format=yuv420p10le'))
        cambi = {'cambi.enc_width': '1280', 'cambi.enc_height': '720', 'cambi.enc_bitdepth': '10'}
        assert [(m.name, os.path.basename(m.path), m.params) for m in run.models] == [
            ('vmaf_v1_hd', 'vmaf_v1.0.16_3d0h.json', cambi), ('vmaf_v1_hd_phone', 'vmaf_v1.0.16_5d0h.json', cambi)]
        assert vmaf_run == [(run.ffmpegQos, {
            'models': run.models, 'subsample': 2, 'output_fmt': 'json', 'threads': 4, 'print_progress': True,
            'end_sync': True, 'features': 'name=psnr', 'cambi_heatmap': False})]

    def test_v0(self, probes, vmaf_run):
        run = make_vmaf(probes, vmaf_v0=True, output_fmt='xml', cambi_heatmap=True)
        run.getVmaf()
        assert filters(run.ffmpegQos) == (chain(0, 'fps=fps=30.0'), chain(1, 'fps=fps=30.0'))
        assert [(m.name, m.version, m.params) for m in run.models] == [
            ('vmaf_hd', 'vmaf_v0.6.1', {}), ('vmaf_hd_neg', 'vmaf_v0.6.1neg', {}),
            ('vmaf_hd_phone', 'vmaf_v0.6.1', {'enable_transform': 'true'})]
        kwargs = vmaf_run[0][1]
        assert (kwargs['output_fmt'], kwargs['cambi_heatmap']) == ('xml', True)
        assert kwargs['features'].startswith('name=psnr|name=cambi')

    @pytest.mark.parametrize("rate, disable_hfr, names", [
        ('60/1', False, ['vmaf_v1_hd_hfr', 'vmaf_v1_hd_phone_hfr']),
        ('60/1', True, ['vmaf_v1_hd', 'vmaf_v1_hd_phone']),
        ('30/1', False, ['vmaf_v1_hd', 'vmaf_v1_hd_phone']),
    ], ids=["hfr", "hfr-disabled", "standard"])
    def test_hfr(self, probes, vmaf_run, rate, disable_hfr, names):
        run = make_vmaf(probes, main=stream(r_frame_rate=rate), ref=stream(r_frame_rate=rate),
                        disable_hfr=disable_hfr)
        run.getVmaf()
        assert [m.name for m in run.models] == names

    def test_4k(self, probes, vmaf_run):
        run = make_vmaf(probes, model='4K')
        run.getVmaf()
        assert [m.name for m in run.models] == ['vmaf_v1_4k', 'vmaf_v1_4k_3h']
        assert run.ffmpegQos.main.filtersList[0] == '[0:v]scale=3840:2160:flags=bicubic[input0_0]'

    def test_manual_fps(self, probes, vmaf_run):
        run = make_vmaf(probes, manual_fps=25.0)
        run.getVmaf()
        assert filters(run.ffmpegQos) == (chain(0, 'fps=fps=25.0', 'format=yuv420p10le'),
                                          chain(1, 'fps=fps=25.0', 'format=yuv420p10le'))

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
