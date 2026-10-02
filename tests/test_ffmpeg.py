"""Tests for FFmpeg.py: the ffprobe and ffmpeg command lines, the filter chains of inputFFmpeg,
the PSNR and VMAF runs of FFmpegQos and the checks of check_ffmpeg, with subprocess replaced
by fakes. The files in tests/data are outputs of ffprobe and ffmpeg 9.0.2 with libvmaf 3.2.1."""

import logging
import os
import subprocess
from types import SimpleNamespace

import pytest

from easyVmafPlus import FFmpeg
from easyVmafPlus.FFmpeg import FFmpegQos, FFprobe, ModelConfig, check_ffmpeg, inputFFmpeg, v1_model_path

DATA = os.path.join(os.path.dirname(__file__), 'data')
BS = "\\"
# Lines of `ffmpeg -hide_banner -filters` of FFmpeg 9.0.2
LIBVMAF_LINE = ' .. libvmaf           VV->V      Calculate the VMAF between two video streams.\n'
PSNR_LINE = ' TS psnr              VV->V      Calculate the PSNR between two video streams.\n'
VERSION_LINE = 'ffmpeg version 9.0.2 Copyright (c) 2000-2026 the FFmpeg developers\n'


def _data(name):
    with open(os.path.join(DATA, name), 'rb') as f:
        return f.read()


@pytest.fixture(autouse=True)
def executables(monkeypatch):
    """Fixed ffmpeg and ffprobe names, so the command lines do not depend on the PATH"""
    monkeypatch.setattr(FFmpegQos, '_executable', 'ffmpeg')
    monkeypatch.setattr(FFprobe, '_executable', 'ffprobe')


@pytest.fixture
def ffprobe_output(monkeypatch):
    """subprocess.check_output returning a captured ffprobe output; returns the calls made"""
    def install(name):
        calls = []

        def check_output(cmd, shell):
            calls.append((cmd, shell))
            return _data(name)
        monkeypatch.setattr(subprocess, 'check_output', check_output)
        return calls
    return install


@pytest.fixture
def popen(monkeypatch):
    """subprocess.Popen replaced by a fake that exits with returncode; returns the fakes created"""
    def install(returncode=0):
        created = []

        class FakePopen:
            def __init__(self, cmd, stdout, shell):
                self.call = (cmd, stdout, shell)
                self.returncode = None
                created.append(self)

            def communicate(self):
                self.returncode = returncode
                return b'', None
        monkeypatch.setattr(subprocess, 'Popen', FakePopen)
        return created
    return install


class TestFFprobe:
    """FFprobe: the ffprobe command line of each probe and the parsed JSON."""

    @staticmethod
    def command(option, loglevel='quiet', src='in.mp4'):
        return ['ffprobe', '-hide_banner', '-loglevel', loglevel, '-print_format', 'json', option,
                '-select_streams', 'v', '-i', src, '-read_intervals', '%+5']

    def test_stream_info(self, ffprobe_output):
        calls = ffprobe_output('ffprobe_streams_mp4.json')
        probe = FFprobe('in.mp4')
        info = probe.getStreamInfo()
        assert (info['codec_name'], info['width'], info['height'], info['r_frame_rate']) == ('h264', 1920, 1080, '30/1')
        assert probe.streamInfo is info
        assert calls == [(self.command('-show_streams'), False)]

    def test_frames_info(self, ffprobe_output):
        calls = ffprobe_output('ffprobe_frames_interlaced.json')
        probe = FFprobe('in.mp4')
        frames = probe.getFramesInfo()
        assert [(f['interlaced_frame'], f['pkt_size']) for f in frames] == [(1, '13111'), (1, '6127'), (1, '8462')]
        assert probe.framesInfo is frames
        assert calls == [(self.command('-show_frames'), False)]

    def test_packets_info(self, ffprobe_output):
        calls = ffprobe_output('ffprobe_packets_mp4.json')
        probe = FFprobe('in.mp4')
        packets = probe.getPacketsInfo()
        assert [(p['pts'], p['size']) for p in packets] == [(0, '628899'), (2048, '64707'), (1024, '5619')]
        assert probe.packetsInfo is packets
        assert calls == [(self.command('-show_packets'), False)]

    def test_format_info(self, ffprobe_output):
        calls = ffprobe_output('ffprobe_format_mkv.json')
        probe = FFprobe('in.mp4')
        info = probe.getFormatInfo()
        assert (info['duration'], info['start_time']) == ('2.066000', '0.000000')
        assert probe.formatInfo is info
        assert calls == [(self.command('-show_format'), False)]

    @pytest.mark.parametrize("loglevel, ffprobe_loglevel", [('info', 'quiet'), ('error', 'quiet'), ('verbose', 'verbose')])
    def test_loglevel(self, ffprobe_output, loglevel, ffprobe_loglevel):
        calls = ffprobe_output('ffprobe_streams_mp4.json')
        FFprobe('in.mp4', loglevel=loglevel).getStreamInfo()
        assert calls[0][0] == self.command('-show_streams', loglevel=ffprobe_loglevel)

    def test_path_is_one_argument(self, ffprobe_output):
        calls = ffprobe_output('ffprobe_streams_mp4.json')
        FFprobe("/videos/it's a [test]; clip.mp4").getStreamInfo()
        assert calls[0][0] == self.command('-show_streams', src="/videos/it's a [test]; clip.mp4")


class TestInputFFmpeg:
    """inputFFmpeg: one filter per call, labelled input<id>_<n> and chained to the previous one."""

    def test_initial_state(self):
        stream = inputFFmpeg('ref.mp4', input_id=1)
        assert (stream.name, stream.id, stream.videoSrc, stream.filtersList, stream.lastOutputID) == (
            'input1_', 1, 'ref.mp4', [], '1:v')

    @pytest.mark.parametrize("method, args, expected", [
        ('setScaleFilter', (1920, 1080), '[0:v]scale=1920:1080:flags=bicubic[input0_0]'),
        ('setScaleFilter', (3840, 2160, 'lanczos'), '[0:v]scale=3840:2160:flags=lanczos[input0_0]'),
        ('setOffsetFilter', (1.5,), '[0:v]setpts=PTS+1.5/TB[input0_0]'),
        ('setDeintFrameFilter', (), '[0:v]yadif=0:-1:0[input0_0]'),
        ('setDeintFieldFilter', (), '[0:v]yadif=1:-1:0[input0_0]'),
        ('setTrimFilter', (1.5, 8.5), '[0:v]trim=start=1.5:duration=8.5, setpts=PTS-STARTPTS[input0_0]'),
        ('setFpsFilter', (29.97003,), '[0:v]fps=fps=29.97003[input0_0]'),
        ('setFormatFilter', ('yuv420p10le',), '[0:v]format=yuv420p10le[input0_0]'),
    ], ids=["scale", "scale-algo", "offset", "deint-frame", "deint-field", "trim", "fps", "format"])
    def test_filter(self, method, args, expected):
        stream = inputFFmpeg('main.mp4', input_id=0)
        getattr(stream, method)(*args)
        assert stream.filtersList == [expected]
        assert stream.lastOutputID == 'input0_0'

    def test_chain(self):
        stream = inputFFmpeg('ref.mp4', input_id=1)
        stream.setScaleFilter(1920, 1080)
        stream.setDeintFrameFilter()
        stream.setFormatFilter('yuv420p10le')
        assert stream.filtersList == ['[1:v]scale=1920:1080:flags=bicubic[input1_0]',
                                      '[input1_0]yadif=0:-1:0[input1_1]',
                                      '[input1_1]format=yuv420p10le[input1_2]']
        assert stream.lastOutputID == 'input1_2'

    def test_clear_filters(self):
        stream = inputFFmpeg('ref.mp4', input_id=1)
        stream.setFpsFilter(25)
        stream.clearFilters()
        assert (stream.filtersList, stream.lastOutputID) == ([], '1:v')
        stream.setFpsFilter(25)
        assert stream.filtersList == ['[1:v]fps=fps=25[input1_0]']


class TestFFmpegQosCommand:
    """FFmpegQos: the ffmpeg command line, with hardware accelerated decoding on both inputs."""

    def test_command_line(self):
        qos = FFmpegQos('main.mp4', 'ref.mp4')
        qos.main.setScaleFilter(1920, 1080)
        qos.ref.setFpsFilter(25)
        qos.psnrFilter = ['[input0_0][input1_0]psnr']
        qos._commit()
        assert qos._cmd == ['ffmpeg', '-y', '-hide_banner', '-stats', '-loglevel', 'info',
                            '-hwaccel', 'auto', '-i', 'main.mp4', '-hwaccel', 'auto', '-i', 'ref.mp4',
                            '-map', '0:v', '-map', '1:v',
                            '-lavfi', '[0:v]scale=1920:1080:flags=bicubic[input0_0];'
                                      '[1:v]fps=fps=25[input1_0];[input0_0][input1_0]psnr',
                            '-f', 'null', '-']

    def test_filter_order(self):
        qos = FFmpegQos('main.mp4', 'ref.mp4')
        qos.main.setFpsFilter(25)
        qos.ref.setFpsFilter(25)
        qos.psnrFilter = ['PSNR']
        qos.vmafFilter = ['VMAF']
        assert qos._commitFilters() == ['-lavfi', '[0:v]fps=fps=25[input0_0];[1:v]fps=fps=25[input1_0];PSNR;VMAF']

    def test_loglevel_and_paths(self):
        qos = FFmpegQos("/v/it's [1].mp4", '/v/ref; 2.mp4', loglevel='verbose')
        qos._commit()
        assert qos._cmd[:6] == ['ffmpeg', '-y', '-hide_banner', '-stats', '-loglevel', 'verbose']
        assert qos._cmd[6:12] == ['-hwaccel', 'auto', '-i', "/v/it's [1].mp4", '-hwaccel', 'auto']
        assert qos._cmd[12:14] == ['-i', '/v/ref; 2.mp4']

    def test_inputs_keep_their_roles(self):
        qos = FFmpegQos('main.mp4', 'ref.mp4')
        assert (qos.main.videoSrc, qos.main.id, qos.ref.videoSrc, qos.ref.id) == ('main.mp4', 0, 'ref.mp4', 1)
        assert (qos.invertedSrc, qos.vmafpath, qos.vmaf_cambi_heatmap_path) == (False, None, None)

    def test_clear_filters(self):
        qos = FFmpegQos('main.mp4', 'ref.mp4')
        qos.psnrFilter = ['PSNR']
        qos.vmafFilter = ['VMAF']
        qos.clearFilters()
        assert (qos.psnrFilter, qos.vmafFilter) == ([], [])


class TestGetPsnr:
    """FFmpegQos.getPsnr: the psnr filter on the last outputs and the average from the ffmpeg output."""

    def test_average(self, monkeypatch):
        calls = []

        def check_output(cmd, stderr, shell):
            calls.append((cmd, stderr, shell))
            return _data('ffmpeg_psnr_output.txt')
        monkeypatch.setattr(subprocess, 'check_output', check_output)
        qos = FFmpegQos('main.mp4', 'ref.mp4')
        qos.ref.setTrimFilter(1.5, 0.5)
        # "PSNR y:29.794730 u:27.095862 v:23.559389 average:27.548382" in the captured output
        assert qos.getPsnr() == 27.548382
        assert qos.psnrFilter == ['[0:v][input1_0]psnr']
        assert calls == [(qos._cmd, subprocess.STDOUT, False)]
        assert qos._cmd[-4:] == ['[1:v]trim=start=1.5:duration=0.5, setpts=PTS-STARTPTS[input1_0];[0:v][input1_0]psnr',
                                 '-f', 'null', '-']

    def test_ffmpeg_error(self, monkeypatch):
        def check_output(cmd, stderr, shell):
            raise subprocess.CalledProcessError(1, cmd)
        monkeypatch.setattr(subprocess, 'check_output', check_output)
        with pytest.raises(subprocess.CalledProcessError):
            FFmpegQos('main.mp4', 'ref.mp4').getPsnr()


class TestGetVmaf:
    """FFmpegQos.getVmaf: the libvmaf filter options, the output paths and the ffmpeg run."""

    MODEL = ModelConfig('vmaf_v1_hd', 'VMAF v1 HD', path='/m/a.json', params={'cambi.enc_width': '1920'})

    def test_libvmaf_filter(self, popen):
        created = popen()
        qos = FFmpegQos('/v/dist.mp4', '/v/ref.mp4')
        qos.main.setFormatFilter('yuv420p10le')
        qos.ref.setFormatFilter('yuv420p10le')
        process = qos.getVmaf(models=[self.MODEL], subsample=2, output_fmt='xml', threads=3, end_sync=True,
                              features='name=psnr')
        assert qos.vmafFilter == [
            '[input0_0][input1_0]libvmaf=log_fmt=xml:model=path=/m/a.json'
            + BS * 2 + ':name=vmaf_v1_hd' + BS * 2 + ':cambi.enc_width=1920'
            + ':n_subsample=2:log_path=/v/dist_vmaf.xml:n_threads=3:shortest=1:feature=name=psnr']
        assert (qos.vmafpath, qos.vmaf_cambi_heatmap_path) == ('/v/dist_vmaf.xml', '/v/dist_cambi_heatmap')
        assert process is created[0]
        assert created[0].call == (qos._cmd, subprocess.PIPE, False)
        assert qos._cmd[-4:-3] == [';'.join(qos.main.filtersList + qos.ref.filtersList + qos.vmafFilter)]

    @pytest.mark.parametrize("output_fmt, log_fmt", [('json', 'json'), ('xml', 'xml'), ('csv', 'csv'), ('txt', 'json')])
    def test_log_format(self, popen, output_fmt, log_fmt):
        popen()
        qos = FFmpegQos('/v/dist.mp4', '/v/ref.mp4')
        qos.getVmaf(models=[self.MODEL], output_fmt=output_fmt, threads=1)
        assert qos.vmafpath == f'/v/dist_vmaf.{log_fmt}'
        assert qos.vmafFilter[0].startswith(f'[0:v][1:v]libvmaf=log_fmt={log_fmt}:model=')
        assert f':log_path=/v/dist_vmaf.{log_fmt}:' in qos.vmafFilter[0]

    def test_log_path_given_and_escaped(self, popen):
        popen()
        qos = FFmpegQos('/v/dist.mp4', '/v/ref.mp4')
        qos.getVmaf(log_path='/out/a:b.json', models=[self.MODEL], threads=1)
        assert qos.vmafpath == '/out/a:b.json'
        assert ':log_path=/out/a' + BS * 2 + ':b.json:n_threads=1:' in qos.vmafFilter[0]

    def test_threads_default_to_cpu_count(self, popen, monkeypatch):
        popen()
        monkeypatch.setattr(os, 'cpu_count', lambda: 6)
        qos = FFmpegQos('/v/dist.mp4', '/v/ref.mp4')
        qos.getVmaf(models=[self.MODEL])
        assert ':n_threads=6:shortest=0' in qos.vmafFilter[0]

    def test_default_models(self, popen):
        popen()
        qos = FFmpegQos('/v/dist.mp4', '/v/ref.mp4')
        qos.getVmaf(threads=1)
        assert ':model=version=vmaf_v0.6.1' + BS * 2 + ':name=vmaf_hd|' in qos.vmafFilter[0]

    def test_cambi_heatmap(self, popen):
        popen()
        qos = FFmpegQos('/v/a:b.mp4', '/v/ref.mp4')
        qos.getVmaf(models=[self.MODEL], threads=1, features='name=psnr|name=cambi', cambi_heatmap=True)
        assert qos.vmafFilter[0].endswith(
            ':feature=name=psnr|name=cambi' + BS * 2 + ':heatmaps_path=/v/a' + BS * 6 + ':b_cambi_heatmap')
        assert ':log_path=/v/a' + BS * 2 + ':b_vmaf.json:' in qos.vmafFilter[0]

    def test_cambi_heatmap_needs_features(self, popen):
        popen()
        qos = FFmpegQos('/v/dist.mp4', '/v/ref.mp4')
        qos.getVmaf(models=[self.MODEL], threads=1, cambi_heatmap=True)
        assert 'heatmaps_path' not in qos.vmafFilter[0]
        assert qos.vmafFilter[0].endswith(':shortest=0')

    def test_ffmpeg_error(self, popen):
        popen(returncode=1)
        qos = FFmpegQos('/v/dist.mp4', '/v/ref.mp4')
        with pytest.raises(subprocess.CalledProcessError) as error:
            qos.getVmaf(models=[self.MODEL], threads=1)
        assert (error.value.returncode, error.value.cmd) == (1, qos._cmd)

    def test_progress(self, monkeypatch, caplog):
        created = []

        class FakeProgress:
            def __init__(self, cmd):
                self.cmd = cmd
                self.stderr = '\n'.join(f'line {i}' for i in range(12))
                created.append(self)

            def run_command_with_progress(self):
                yield 50.0
                yield 100.0
        monkeypatch.setattr(FFmpeg, 'FfmpegProgress', FakeProgress)
        qos = FFmpegQos('/v/dist.mp4', '/v/ref.mp4')
        with caplog.at_level(logging.INFO, logger='easyVmafPlus.FFmpeg'):
            process = qos.getVmaf(models=[self.MODEL], threads=1, print_progress=True)
        assert process is created[0]
        assert created[0].cmd == qos._cmd
        assert [r.getMessage() for r in caplog.records] == ['progress = 50.0% - line 3', 'progress = 100.0% - line 3']


@pytest.fixture
def ffmpeg_run(monkeypatch):
    """subprocess.run of check_ffmpeg answering per command; returns the commands run"""
    def install(version=VERSION_LINE, filters=LIBVMAF_LINE + PSNR_LINE, builtin_stderr='', v1_returncode=0):
        calls = []

        def run(cmd, capture_output, text):
            calls.append(cmd)
            if cmd[1:] == ['-version']:
                return SimpleNamespace(stdout=version, stderr='', returncode=0)
            if cmd[1:] == ['-hide_banner', '-filters']:
                return SimpleNamespace(stdout=filters, stderr='', returncode=0)
            if 'nullsrc=s=64x64:r=1:d=0.1' in cmd:
                return SimpleNamespace(stdout='', stderr=builtin_stderr, returncode=1 if builtin_stderr else 0)
            return SimpleNamespace(stdout='', stderr='', returncode=v1_returncode)
        monkeypatch.setattr(subprocess, 'run', run)
        return calls
    return install


class TestCheckFfmpeg:
    """check_ffmpeg: the binaries, the libvmaf filter, the built-in model and the commands it runs."""

    def test_all_available(self, ffmpeg_run):
        calls = ffmpeg_run()
        assert check_ffmpeg() == {'version_str': '9.0', 'meets_minimum': True, 'libvmaf': True,
                                  'builtin_models': True, 'v1_models': True}
        v1_model = FFmpegQos._escape_filter_value(v1_model_path('3d0h'), option_levels=2)
        assert calls == [
            ['ffmpeg', '-version'],
            ['ffprobe', '-version'],
            ['ffmpeg', '-hide_banner', '-filters'],
            ['ffmpeg', '-hide_banner', '-loglevel', 'error',
             '-f', 'lavfi', '-i', 'nullsrc=s=64x64:r=1:d=0.1', '-f', 'lavfi', '-i', 'nullsrc=s=64x64:r=1:d=0.1',
             '-lavfi', f'libvmaf=model=version=vmaf_v0.6.1:log_fmt=json:log_path={os.devnull}', '-f', 'null', '-'],
            ['ffmpeg', '-hide_banner', '-loglevel', 'error',
             '-f', 'lavfi', '-i', 'nullsrc=s=320x240:r=25:d=0.2', '-f', 'lavfi', '-i', 'nullsrc=s=320x240:r=25:d=0.2',
             '-lavfi', f'libvmaf=model=path={v1_model}:log_fmt=json:log_path={os.devnull}', '-f', 'null', '-'],
        ]

    def test_no_libvmaf(self, ffmpeg_run):
        ffmpeg_run(filters=PSNR_LINE)
        assert check_ffmpeg()['libvmaf'] is False

    def test_builtin_model_missing(self, ffmpeg_run):
        # vf_libvmaf.c: "could not load libvmaf model with version: %s"
        ffmpeg_run(builtin_stderr='could not load libvmaf model with version: vmaf_v0.6.1\n')
        assert check_ffmpeg()['builtin_models'] is False

    def test_other_probe_error(self, ffmpeg_run):
        ffmpeg_run(builtin_stderr='Error initializing filters\n')
        assert check_ffmpeg()['builtin_models'] is True

    def test_ffmpeg_not_found(self, monkeypatch):
        monkeypatch.setattr(FFmpegQos, '_executable', None)
        with pytest.raises(RuntimeError, match=r"^ffmpeg not found on PATH\. Install FFmpeg >= 9\.0 built with "
                                               r"--enable-libvmaf, or point the FFMPEG environment variable to it\.$"):
            check_ffmpeg()

    def test_ffprobe_not_found(self, monkeypatch):
        monkeypatch.setattr(FFprobe, '_executable', None)
        with pytest.raises(RuntimeError, match=r"^ffprobe not found on PATH\. Install FFmpeg >= 9\.0, "
                                               r"or point the FFPROBE environment variable to it\.$"):
            check_ffmpeg()

    def test_cannot_run(self, monkeypatch):
        def run(cmd, capture_output, text):
            raise PermissionError(13, 'Permission denied', cmd[0])
        monkeypatch.setattr(subprocess, 'run', run)
        with pytest.raises(RuntimeError, match=r"^cannot run 'ffmpeg': Permission denied$"):
            check_ffmpeg()
