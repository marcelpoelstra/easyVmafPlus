"""Tests for FFmpeg.py: the ffprobe and ffmpeg command lines, the filter chains of inputFFmpeg,
the PSNR and VMAF runs of FFmpegQos and the checks of check_ffmpeg, with subprocess replaced
by fakes. The files in tests/data are outputs of ffprobe and ffmpeg 9.0.2 with libvmaf 3.2.1."""

import json
import logging
import os
import subprocess
from types import SimpleNamespace

import pytest

from easyVmafPlus import FFmpeg
from easyVmafPlus.FFmpeg import (FFmpegQos, FFprobe, ModelConfig, check_ffmpeg, create_unique_dir,
                                 create_unique_file, inputFFmpeg, known_scores, read_vmaf_log, v1_model_path)

DATA = os.path.join(os.path.dirname(__file__), 'data')
# Per-frame scores of tests/data/vmaf_v1_hd.* and tests/data/vmaf_v0_hd.json
V1_HD = {
    'vmaf_v1_hd': [53.299798, 54.369612, 52.443107, 52.641412, 53.13457],
    'vmaf_v1_hd_phone': [54.006148, 53.949253, 53.893185, 53.967701, 53.712772],
}
V0_HD = {
    'vmaf_hd': [39.066204, 41.277708, 41.096402, 41.249924, 40.860752],
    'vmaf_hd_neg': [37.677686, 39.919671, 39.724437, 39.896747, 39.557843],
    'vmaf_hd_phone': [58.388002, 60.952835, 60.745159, 60.92104, 60.47454],
}
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
    """FFmpegQos.getVmaf: the libvmaf filter options, the reserved output paths and the ffmpeg run."""

    MODEL = ModelConfig('vmaf_v1_hd', 'VMAF v1 HD', path='/m/a.json', params={'cambi.enc_width': '1920'})

    @staticmethod
    def qos(tmp_path, main='dist.mp4'):
        return FFmpegQos(str(tmp_path / main), str(tmp_path / 'ref.mp4'))

    def test_libvmaf_filter(self, popen, tmp_path):
        created = popen()
        qos = self.qos(tmp_path)
        qos.main.setFormatFilter('yuv420p10le')
        qos.ref.setFormatFilter('yuv420p10le')
        process = qos.getVmaf(models=[self.MODEL], subsample=2, output_fmt='xml', threads=3, end_sync=True,
                              features='name=psnr')
        assert qos.vmafFilter == [
            '[input0_0][input1_0]libvmaf=log_fmt=xml:model=path=/m/a.json'
            + BS * 2 + ':name=vmaf_v1_hd' + BS * 2 + ':cambi.enc_width=1920'
            + f':n_subsample=2:log_path={tmp_path}/dist_vmaf.xml:n_threads=3:shortest=1:feature=name=psnr']
        assert (qos.vmafpath, qos.vmaf_cambi_heatmap_path) == (str(tmp_path / 'dist_vmaf.xml'), None)
        assert os.path.getsize(qos.vmafpath) == 0
        assert process is created[0]
        assert created[0].call == (qos._cmd, subprocess.PIPE, False)
        assert qos._cmd[-4:-3] == [';'.join(qos.main.filtersList + qos.ref.filtersList + qos.vmafFilter)]

    @pytest.mark.parametrize("output_fmt, log_fmt", [('json', 'json'), ('xml', 'xml'), ('csv', 'csv'), ('txt', 'json')])
    def test_log_format(self, popen, tmp_path, output_fmt, log_fmt):
        popen()
        qos = self.qos(tmp_path)
        qos.getVmaf(models=[self.MODEL], output_fmt=output_fmt, threads=1)
        assert qos.vmafpath == str(tmp_path / f'dist_vmaf.{log_fmt}')
        assert qos.vmafFilter[0].startswith(f'[0:v][1:v]libvmaf=log_fmt={log_fmt}:model=')
        assert f':log_path={tmp_path}/dist_vmaf.{log_fmt}:' in qos.vmafFilter[0]

    def test_log_path_given_and_escaped(self, popen, tmp_path):
        popen()
        qos = self.qos(tmp_path)
        qos.getVmaf(log_path=str(tmp_path / 'a:b.json'), models=[self.MODEL], threads=1)
        assert qos.vmafpath == str(tmp_path / 'a:b.json')
        assert f':log_path={tmp_path}/a' + BS * 2 + ':b.json:n_threads=1:' in qos.vmafFilter[0]

    def test_existing_outputs_are_kept(self, popen, tmp_path):
        popen()
        (tmp_path / 'dist_vmaf.json').write_text('first run')
        (tmp_path / 'dist_cambi_heatmap').mkdir()
        qos = self.qos(tmp_path)
        qos.getVmaf(models=[self.MODEL], threads=1, features='name=psnr|name=cambi', cambi_heatmap=True)
        assert (qos.vmafpath, qos.vmaf_cambi_heatmap_path) == (
            str(tmp_path / 'dist_vmaf_2.json'), str(tmp_path / 'dist_cambi_heatmap_2'))
        assert (tmp_path / 'dist_vmaf.json').read_text() == 'first run'
        assert f':log_path={tmp_path}/dist_vmaf_2.json:' in qos.vmafFilter[0]
        assert qos.vmafFilter[0].endswith(f':heatmaps_path={tmp_path}/dist_cambi_heatmap_2')

    def test_threads_default_to_cpu_count(self, popen, monkeypatch, tmp_path):
        popen()
        monkeypatch.setattr(os, 'cpu_count', lambda: 6)
        qos = self.qos(tmp_path)
        qos.getVmaf(models=[self.MODEL])
        assert ':n_threads=6:shortest=0' in qos.vmafFilter[0]

    def test_default_models(self, popen, tmp_path):
        popen()
        qos = self.qos(tmp_path)
        qos.getVmaf(threads=1)
        assert ':model=version=vmaf_v0.6.1' + BS * 2 + ':name=vmaf_hd|' in qos.vmafFilter[0]

    def test_cambi_heatmap(self, popen, tmp_path):
        popen()
        qos = self.qos(tmp_path, main='a:b.mp4')
        qos.getVmaf(models=[self.MODEL], threads=1, features='name=psnr|name=cambi', cambi_heatmap=True)
        assert qos.vmafFilter[0].endswith(
            ':feature=name=psnr|name=cambi' + BS * 2 + f':heatmaps_path={tmp_path}/a' + BS * 6 + ':b_cambi_heatmap')
        assert f':log_path={tmp_path}/a' + BS * 2 + ':b_vmaf.json:' in qos.vmafFilter[0]
        assert os.path.isdir(qos.vmaf_cambi_heatmap_path)

    def test_cambi_heatmap_needs_features(self, popen, tmp_path):
        popen()
        qos = self.qos(tmp_path)
        qos.getVmaf(models=[self.MODEL], threads=1, cambi_heatmap=True)
        assert 'heatmaps_path' not in qos.vmafFilter[0]
        assert qos.vmafFilter[0].endswith(':shortest=0')
        assert (qos.vmaf_cambi_heatmap_path, (tmp_path / 'dist_cambi_heatmap').exists()) == (None, False)

    def test_ffmpeg_error(self, popen, tmp_path):
        popen(returncode=1)
        qos = self.qos(tmp_path)
        with pytest.raises(subprocess.CalledProcessError) as error:
            qos.getVmaf(models=[self.MODEL], threads=1, features='name=psnr|name=cambi', cambi_heatmap=True)
        assert (error.value.returncode, error.value.cmd) == (1, qos._cmd)
        assert os.listdir(tmp_path) == []

    def test_error_keeps_written_log(self, monkeypatch, tmp_path):
        qos = self.qos(tmp_path)

        class FakePopen:
            def __init__(self, cmd, stdout, shell):
                self.returncode = None

            def communicate(self):
                with open(qos.vmafpath, 'w') as log_file:
                    log_file.write('partial')
                self.returncode = 1
                return b'', None
        monkeypatch.setattr(subprocess, 'Popen', FakePopen)
        with pytest.raises(subprocess.CalledProcessError):
            qos.getVmaf(models=[self.MODEL], threads=1)
        assert (tmp_path / 'dist_vmaf.json').read_text() == 'partial'

    def test_reservation_error_removes_log(self, monkeypatch, tmp_path):
        def create_unique_dir(path):
            raise PermissionError(13, 'Permission denied', path)
        monkeypatch.setattr(FFmpeg, 'create_unique_dir', create_unique_dir)
        with pytest.raises(PermissionError):
            self.qos(tmp_path).getVmaf(models=[self.MODEL], threads=1, features='name=psnr|name=cambi',
                                       cambi_heatmap=True)
        assert os.listdir(tmp_path) == []

    def test_interrupt_removes_empty_outputs(self, monkeypatch, tmp_path):
        class FakePopen:
            def __init__(self, cmd, stdout, shell):
                self.returncode = None

            def communicate(self):
                raise SystemExit(0)
        monkeypatch.setattr(subprocess, 'Popen', FakePopen)
        with pytest.raises(SystemExit):
            self.qos(tmp_path).getVmaf(models=[self.MODEL], threads=1, features='name=psnr|name=cambi',
                                       cambi_heatmap=True)
        assert os.listdir(tmp_path) == []

    def test_progress(self, monkeypatch, caplog, tmp_path):
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
        qos = self.qos(tmp_path)
        with caplog.at_level(logging.INFO, logger='easyVmafPlus.FFmpeg'):
            process = qos.getVmaf(models=[self.MODEL], threads=1, print_progress=True)
        assert process is created[0]
        assert created[0].cmd == qos._cmd
        assert [r.getMessage() for r in caplog.records] == ['progress = 50.0% - line 3', 'progress = 100.0% - line 3']


@pytest.fixture
def frames_popen(monkeypatch):
    """subprocess.Popen of FFmpegQos.getFrames replaced by a fake that writes `lines` on stderr and exits
    with returncode; returns the command and the content of the graph file"""
    def install(lines, returncode=0):
        seen = {}

        class FakePopen:
            def __init__(self, cmd, stdout, stderr, text, encoding, errors, shell):
                seen['cmd'] = cmd
                seen['call'] = (stdout, stderr, text, encoding, errors, shell)
                with open(cmd[cmd.index('-/filter_complex') + 1]) as graph_file:
                    seen['graph'] = graph_file.read()
                self.stderr = iter(lines)
                self.returncode = None

            def wait(self):
                self.returncode = returncode
                return returncode
        monkeypatch.setattr(subprocess, 'Popen', FakePopen)
        return seen
    return install


SHOWINFO = '[showinfo@frames{branch} @ 0x1] n:   {n} pts:    {pts} pts_time:{t}    duration:1\n'
OUTPUT = ['-fps_mode', 'passthrough', '-c:v', 'tiff', '-compression_algo', 'lzw', '-pix_fmt', 'rgb24']


class TestGetFrames:
    """FFmpegQos.getFrames: the frames pass command, the graph file and the reading of its stderr."""

    def test_command_and_times(self, frames_popen, caplog, capsys, tmp_path):
        seen = frames_popen([
            "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'dist.mp4':\n",
            SHOWINFO.format(branch=0, n=0, pts=0, t='0'),
            SHOWINFO.format(branch=0, n=1, pts=207, t='6.9'),
            '[showinfo@frames0 @ 0x1]   color_range:tv\n',
            'frame=2\n', 'out_time=00:00:06.933333\n', 'progress=continue\n',
            SHOWINFO.format(branch=0, n=2, pts=209, t='6.966667'),
            'out_time=00:00:07.000000\n', 'progress=end\n'])
        qos = FFmpegQos('dist.mp4', 'ref.mp4')
        qos.main.setFpsFilter(30.0)
        with caplog.at_level(logging.INFO, logger='easyVmafPlus.FFmpeg'):
            written = qos.getFrames([209, 207], str(tmp_path))
        # the first file of the branch is frame 0, which no output may lack
        assert written == [(207, str(tmp_path / '00000_000002.tif'), 6.9),
                           (209, str(tmp_path / '00000_000003.tif'), 6.966667)]
        graph_path = seen['cmd'][seen['cmd'].index('-/filter_complex') + 1]
        assert seen['cmd'] == ['ffmpeg', '-y', '-hide_banner', '-nostats', '-loglevel', 'info', '-progress', 'pipe:2',
                               '-stats_period', '10', '-hwaccel', 'auto', '-i', 'dist.mp4',
                               '-/filter_complex', graph_path, '-map', '[frames0]', *OUTPUT,
                               str(tmp_path / '00000_%06d.tif')]
        assert seen['call'] == (subprocess.DEVNULL, subprocess.PIPE, True, 'utf-8', 'replace', False)
        assert seen['graph'] == ('[0:v]fps=fps=30.0[input0_0];[input0_0]setpts=PTS-STARTPTS,split=1[select0];'
                                 '[select0]select=eq(n' + BS + ',0)+eq(n' + BS + ',207)+eq(n' + BS + ',209),'
                                 'showinfo@frames0[frames0]')
        assert not os.path.exists(graph_path)
        assert [r.getMessage() for r in caplog.records] == [
            'Low frames pass: 00:00:06.933333 of the distorted input',
            'Low frames pass: 00:00:07.000000 of the distorted input']
        assert capsys.readouterr().err == "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'dist.mp4':\n"

    def test_branches_of_99_frames(self, frames_popen, tmp_path):
        seen = frames_popen([])
        assert FFmpegQos('dist.mp4', 'ref.mp4').getFrames(list(range(1, 251)), str(tmp_path)) == []
        head, *branches = seen['graph'].split(';')
        assert head == '[0:v]setpts=PTS-STARTPTS,split=3[select0][select1][select2]'
        # 99 frames plus frame 0, 99 plus frame 0, 52 plus frame 0
        assert [branch.count('eq(n') for branch in branches] == [100, 100, 53]
        assert [seen['cmd'][index + 1] for index, arg in enumerate(seen['cmd']) if arg == '-map'] == [
            '[frames0]', '[frames1]', '[frames2]']
        assert seen['cmd'][-1] == str(tmp_path / '00002_%06d.tif')

    def test_frames_beyond_the_end(self, frames_popen, tmp_path):
        frames_popen([SHOWINFO.format(branch=0, n=0, pts=0, t='0'), SHOWINFO.format(branch=0, n=1, pts=5, t='0.2')])
        assert FFmpegQos('dist.mp4', 'ref.mp4').getFrames([5, 900], str(tmp_path)) == [
            (5, str(tmp_path / '00000_000002.tif'), 0.2)]

    def test_frame_zero_requested(self, frames_popen, tmp_path):
        seen = frames_popen([SHOWINFO.format(branch=0, n=0, pts=0, t='0')])
        assert FFmpegQos('dist.mp4', 'ref.mp4').getFrames([0], str(tmp_path)) == [
            (0, str(tmp_path / '00000_000001.tif'), 0.0)]
        assert seen['graph'] == ('[0:v]setpts=PTS-STARTPTS,split=1[select0];'
                                 '[select0]select=eq(n' + BS + ',0),showinfo@frames0[frames0]')

    def test_ffmpeg_error(self, frames_popen, tmp_path):
        seen = frames_popen(['[tiff @ 0x1] error\n'], returncode=1)
        with pytest.raises(subprocess.CalledProcessError) as error:
            FFmpegQos('dist.mp4', 'ref.mp4').getFrames([5], str(tmp_path))
        assert (error.value.returncode, error.value.cmd) == (1, seen['cmd'])
        assert not os.path.exists(seen['cmd'][seen['cmd'].index('-/filter_complex') + 1])


class TestUniqueNames:
    """create_unique_file and create_unique_dir: a taken name gets _2, _3 and so on; nothing is replaced."""

    def test_file(self, tmp_path):
        paths = []
        for content in (b'first', b'second', b'third'):
            unique_file, path = create_unique_file(str(tmp_path / 'a_vmaf.json'))
            with unique_file:
                unique_file.write(content)
            paths.append(path)
        assert paths == [str(tmp_path / 'a_vmaf.json'), str(tmp_path / 'a_vmaf_2.json'), str(tmp_path / 'a_vmaf_3.json')]
        assert [open(path, 'rb').read() for path in paths] == [b'first', b'second', b'third']

    def test_dir(self, tmp_path):
        (tmp_path / 'a_cambi_heatmap').mkdir()
        (tmp_path / 'a_cambi_heatmap' / 'keep').write_text('keep')
        assert create_unique_dir(str(tmp_path / 'a_cambi_heatmap')) == str(tmp_path / 'a_cambi_heatmap_2')
        assert create_unique_dir(str(tmp_path / 'a_cambi_heatmap')) == str(tmp_path / 'a_cambi_heatmap_3')
        assert (tmp_path / 'a_cambi_heatmap' / 'keep').read_text() == 'keep'


class TestKnownScores:
    """known_scores: the score names of VMAF_MODELS with the v1 HFR names, their labels and score caps."""

    def test_names_labels_caps(self):
        scores = known_scores()
        assert list(scores) == ['vmaf_hd', 'vmaf_hd_neg', 'vmaf_hd_phone', 'vmaf_4k',
                                'vmaf_v1_hd', 'vmaf_v1_hd_hfr', 'vmaf_v1_hd_phone', 'vmaf_v1_hd_phone_hfr',
                                'vmaf_v1_4k', 'vmaf_v1_4k_hfr', 'vmaf_v1_4k_3h', 'vmaf_v1_4k_3h_hfr']
        assert scores == {
            'vmaf_hd': ('VMAF HD', 100.0), 'vmaf_hd_neg': ('VMAF Neg', 100.0),
            'vmaf_hd_phone': ('VMAF Phone', 100.0), 'vmaf_4k': ('VMAF 4K', 100.0),
            'vmaf_v1_hd': ('VMAF v1 HD', 100.0), 'vmaf_v1_hd_hfr': ('VMAF v1 HD HFR', 100.0),
            'vmaf_v1_hd_phone': ('VMAF v1 Phone', 100.0), 'vmaf_v1_hd_phone_hfr': ('VMAF v1 Phone HFR', 100.0),
            'vmaf_v1_4k': ('VMAF v1 4K', 100.0), 'vmaf_v1_4k_hfr': ('VMAF v1 4K HFR', 100.0),
            'vmaf_v1_4k_3h': ('VMAF v1 4K 3H', 110.0), 'vmaf_v1_4k_3h_hfr': ('VMAF v1 4K 3H HFR', 110.0)}


class TestReadVmafLog:
    """read_vmaf_log: frame numbers, scores, means and harmonic means of a json, xml or csv VMAF log."""

    @pytest.mark.parametrize("log_fmt", ['json', 'xml', 'csv'])
    def test_v1_hd(self, log_fmt):
        path = os.path.join(DATA, f'vmaf_v1_hd.{log_fmt}')
        log = read_vmaf_log(path)
        assert (log.path, log.frame_numbers) == (path, [0, 1, 2, 3, 4])
        assert list(log.scores) == ['vmaf_v1_hd', 'vmaf_v1_hd_phone']
        assert log.scores == {name: pytest.approx(values) for name, values in V1_HD.items()}
        # the pooled values of the json and xml logs; computed for the csv log, which has none
        assert log.means == {'vmaf_v1_hd': pytest.approx(53.1777, abs=1e-6),
                             'vmaf_v1_hd_phone': pytest.approx(53.905812, abs=1e-6)}
        assert log.harmonic_means == {'vmaf_v1_hd': pytest.approx(53.169416, abs=1e-6),
                                      'vmaf_v1_hd_phone': pytest.approx(53.905618, abs=1e-6)}

    def test_v0_hd(self):
        log = read_vmaf_log(os.path.join(DATA, 'vmaf_v0_hd.json'))
        assert log.scores == {name: pytest.approx(values) for name, values in V0_HD.items()}
        assert (log.means['vmaf_hd'], log.harmonic_means['vmaf_hd']) == (40.710198, 40.692994)

    def test_subsampled_frame_numbers(self, tmp_path):
        path = tmp_path / 'a_vmaf.json'
        path.write_text(json.dumps({'frames': [{'frameNum': n, 'metrics': {'vmaf_v1_hd': 50.0 + n}} for n in (0, 3, 6)]}))
        log = read_vmaf_log(str(path))
        assert (log.frame_numbers, log.scores) == ([0, 3, 6], {'vmaf_v1_hd': [50.0, 53.0, 56.0]})

    @pytest.mark.parametrize("name, content, message", [
        ('a_vmaf.txt', '', "unknown format 'txt'"),
        ('a_vmaf.json', '{"frames": []}', 'holds no frames'),
        ('a_vmaf.json', '{"frames": [{"frameNum": 0, "metrics": {"psnr_y": 30.0}}]}', 'holds no VMAF score'),
        ('a_vmaf.json', '{"version": "3.2.0"}', 'cannot be read'),
        ('a_vmaf.json', '[1, 2]', 'cannot be read'),
        ('a_vmaf.xml', '<VMAF>', 'cannot be read'),
        ('a_vmaf.json', '{"frames": [{"frameNum": 0, "metrics": {"vmaf_v1_hd": 50.0}}], '
                        '"pooled_metrics": {"vmaf_v1_hd": {"mean": 50.0}}}', 'cannot be read'),
        ('a_vmaf.json', '{"frames": [{"frameNum": 0, "metrics": {"vmaf_v1_hd": 50.0}}], "pooled_metrics": null}',
         'cannot be read'),
    ], ids=["unknown-format", "no-frames", "no-vmaf-score", "no-frames-key", "not-an-object", "broken-xml",
            "pooled-without-harmonic-mean", "pooled-null"])
    def test_errors(self, tmp_path, name, content, message):
        (tmp_path / name).write_text(content)
        with pytest.raises(ValueError, match=message):
            read_vmaf_log(str(tmp_path / name))

    def test_missing_file(self, tmp_path):
        with pytest.raises(OSError):
            read_vmaf_log(str(tmp_path / 'none_vmaf.json'))


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
