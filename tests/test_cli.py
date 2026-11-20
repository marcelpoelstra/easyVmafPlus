"""Tests for cli.py: the result record, the arguments, the startup check, the summary file and the run per
distorted file, with check_ffmpeg and vmaf replaced by fakes. tests/data/vmaf_v1_hd.* and
tests/data/vmaf_v0_hd.json are logs of libvmaf 3.2.1 over five frames."""

import json
import logging
import math
import os
import shutil
import subprocess
import sys
from statistics import mean
from types import SimpleNamespace

import pytest
from PIL import Image

from easyVmafPlus import cli
from easyVmafPlus.FFmpeg import create_unique_file, select_models
from easyVmafPlus.Vmaf import UnsupportedFramerateError, UnsupportedModelConfigError, UnsupportedRangeError

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
CHECK_OK = {'version_str': '9.0', 'meets_minimum': True, 'libvmaf': True, 'builtin_models': True, 'v1_models': True,
            'hdr_vmaf': True}
HDR10 = {'range': 'tv', 'primaries': 'bt2020', 'trc': 'smpte2084', 'matrix': 'bt2020nc'}
DEFAULT_KWARGS = {
    'loglevel': 'info', 'subsample': 1, 'model': 'HD', 'output_fmt': 'json', 'threads': 0, 'print_progress': False,
    'end_sync': False, 'manual_fps': 0, 'cambi_heatmap': False, 'vmaf_versions': ('1',), 'views': None, 'hfr': 'auto',
    'bitdepth': 'auto', 'enc_size': None, 'enc_bitdepth': None, 'model_options': [], 'dynamic_range': 'auto',
    'reference_color': None, 'distorted_color': None, 'start_frame': None, 'frame_count': None, 'jobs': 1}


class TestBuildResult:
    """cli._build_result: the record of one distorted file."""

    def test_sync(self):
        assert cli._build_result('d.mp4', 'r.mp4', 1.5000004, 40.0321214, 'HD') == {
            'distorted': 'd.mp4', 'reference': 'r.mp4', 'sync': {'offset': 1.5, 'psnr': 40.032121}}

    def test_without_sync(self):
        assert cli._build_result('d.mp4', 'r.mp4', None, None, 'HD')['sync'] == {'offset': 0.0, 'psnr': None}

    @pytest.mark.parametrize("psnr, status", [(math.inf, 'positive_infinity'), (-math.inf, 'negative_infinity'),
                                              (math.nan, 'nan')], ids=["inf", "-inf", "nan"])
    def test_non_finite_psnr(self, psnr, status):
        result = cli._build_result('d.mp4', 'r.mp4', 0.5, psnr, 'HD')
        assert result['sync'] == {'offset': 0.5, 'psnr': None, 'psnr_status': status}
        json.dumps(result, allow_nan=False)

    def test_vmaf(self):
        result = cli._build_result('d.mp4', 'r.mp4', 0, None, '4K',
                                   vmaf_scores={'vmaf_v1_4k': 87.2551875058, 'vmaf_v1_4k_3h': 97.4560392},
                                   vmaf_output_file='d_vmaf.json', cambi_heatmap_path='d_cambi_heatmap',
                                   pix_fmt='yuv420p10le', hfr=False, dynamic_range='sdr')
        assert result['vmaf'] == {'model': '4K', 'pix_fmt': 'yuv420p10le', 'hfr': False, 'dynamic_range': 'sdr',
                                  'vmaf_v1_4k': 87.255188, 'vmaf_v1_4k_3h': 97.456039,
                                  'output_file': 'd_vmaf.json', 'cambi_heatmap_path': 'd_cambi_heatmap'}

    def test_vmaf_without_paths(self):
        result = cli._build_result('d.mp4', 'r.mp4', 0, None, 'HD', vmaf_scores={'vmaf_hd': 90.0})
        assert result['vmaf'] == {'model': 'HD', 'vmaf_hd': 90.0}

    def test_hdr_and_range(self):
        color = {'distorted': dict(HDR10), 'reference': dict(HDR10, primaries='bt709')}
        result = cli._build_result('d.mp4', 'r.mp4', 0, None, 'HD', vmaf_scores={'hdr_vmaf_v1_hd': 80.0},
                                   dynamic_range='hdr', color=color,
                                   frame_range={'start_frame': 100, 'frame_count': 50, 'frames_scored': 50})
        assert list(result['vmaf']) == ['model', 'dynamic_range', 'color', 'hdr_vmaf_v1_hd', 'range']
        assert list(result['vmaf']['color']) == ['reference', 'distorted']
        assert result['vmaf']['color']['reference']['primaries'] == 'bt709'
        assert result['vmaf']['range'] == {'start_frame': 100, 'frame_count': 50, 'frames_scored': 50}


class TestSummary:
    """cli._summary_path and _write_summary: the summary file next to the log, never replaced."""

    @pytest.mark.parametrize("distorted, vmafpath, expected", [
        ('a/b.mp4', None, 'a/b_sync_summary.json'),
        ('a/b.mp4', 'a/b_vmaf.json', 'a/b_vmaf_summary.json'),
        ('a/b.mp4', 'a/b_vmaf_f0-99.xml', 'a/b_vmaf_f0-99_summary.json'),
        ('a/b.mp4', 'a/b_vmaf_2.csv', 'a/b_vmaf_2_summary.json'),
    ], ids=["sync", "json", "range", "second"])
    def test_path(self, distorted, vmafpath, expected):
        assert cli._summary_path(distorted, vmafpath) == expected

    def test_write(self, tmp_path):
        record = {'distorted': 'd.mp4', 'sync': {'offset': 1.5, 'psnr': None}}
        path = str(tmp_path / 'd_vmaf_summary.json')
        assert cli._write_summary(record, path) == path
        assert cli._write_summary(record, path) == str(tmp_path / 'd_vmaf_summary_2.json')
        with open(path) as summary:
            text = summary.read()
        assert json.loads(text) == record
        assert text == json.dumps(record, indent=2) + '\n'

    def test_strict_json(self, tmp_path):
        with pytest.raises(ValueError):
            cli._write_summary({'x': math.nan}, str(tmp_path / 's.json'))
        assert os.listdir(tmp_path) == []


class TestArguments:
    """cli.get_args and MyParser: flags, defaults and argument errors."""

    @staticmethod
    def parse(monkeypatch, *argv):
        monkeypatch.setattr(sys, 'argv', ['easyVmafPlus', *argv])
        return cli.get_args()

    DEFAULTS = {
        'd': 'd.mp4', 'r': 'r.mp4', 'fps': 0, 'sw': 0, 'ss': 0, 'reverse': False, 'sync_only': False,
        'sync_offset': None, 'endsync': False, 'start_frame': None, 'frame_count': None, 'model': 'HD',
        'vmaf_version': None, 'vmaf_v0': False, 'view': None, 'disable_hfr': False, 'force_hfr': False,
        'bitdepth': 'auto', 'dynamic_range': 'auto', 'reference_color': None, 'distorted_color': None,
        'enc_size': None, 'enc_bitdepth': None, 'model_option': [], 'output_fmt': 'json', 'cambi_heatmap': False,
        'plot': False, 'low_frames': False, 'json': False, 'verbose': False, 'progress': False, 'threads': 0,
        'jobs': 1, 'n': 1, 'vmaf_versions': ('1',), 'hfr': 'auto'}

    def test_defaults(self, monkeypatch):
        assert vars(self.parse(monkeypatch, '-d', 'd.mp4', '-r', 'r.mp4')) == self.DEFAULTS

    def test_values(self, monkeypatch):
        args = self.parse(monkeypatch, '-d', 'd*.mp4', '-r', 'r.mp4', '-sw', '2', '-ss', '1.5', '-fps', '25',
                          '-reverse', '-model', '4K', '-vmaf_v0', '-disable_hfr', '-threads', '8',
                          '-verbose', '-progress', '-endsync', '-output_fmt', 'xml', '-cambi_heatmap', '-plot',
                          '-low_frames', '-json', '-bitdepth', '10', '-dynamic_range', 'SDR',
                          '-start_frame', '10', '-frame_count', '100', '-jobs', '4')
        assert vars(args) == dict(
            self.DEFAULTS, d='d*.mp4', sw=2.0, ss=1.5, fps=25.0, reverse=True, model='4K', vmaf_v0=True,
            disable_hfr=True, threads=8, verbose=True, progress=True, endsync=True, output_fmt='xml',
            cambi_heatmap=True, plot=True, low_frames=True, json=True, bitdepth='10',
            dynamic_range='sdr', start_frame=10, frame_count=100, jobs=4, vmaf_versions=('0.6',), hfr='off')

    def test_sync_only_values(self, monkeypatch):
        args = self.parse(monkeypatch, '-d', 'd.mp4', '-r', 'r.mp4', '-sw', '2', '-sync_only', '-subsample', '3')
        assert (args.sw, args.sync_only, args.n) == (2.0, True, 3)

    def test_v1_values(self, monkeypatch):
        args = self.parse(monkeypatch, '-d', 'd.mp4', '-r', 'r.mp4', '-vmaf_version', '1', '0.6', '-view', 'PHONE',
                          '3h', '-force_hfr', '-enc_size', '1280x720', '-enc_bitdepth', '10', '-model_option',
                          'cambi.topk=0.5', '-model_option', 'CAMBI.full_ref=true', '-sync_offset', '-1.5',
                          '-reference_color', 'hdr10', '-distorted_color', 'trc=smpte2084')
        assert vars(args) == dict(
            self.DEFAULTS, vmaf_version=['1', '0.6'], vmaf_versions=('1', '0.6'), view=['phone', '3h'], force_hfr=True,
            hfr='on', enc_size=(1280, 720), enc_bitdepth=10, model_option=['cambi.topk=0.5', 'cambi.full_ref=true'],
            sync_offset=-1.5, reference_color='hdr10', distorted_color='trc=smpte2084')
        args = self.parse(monkeypatch, '-d', 'd.mp4', '-r', 'r.mp4', '-dynamic_range', 'HDR', '-vmaf_version', '1')
        assert (args.dynamic_range, args.vmaf_versions) == ('hdr', ('1',))

    @pytest.mark.parametrize("argv, message", [
        (['-sync_only'], 'error: -sync_only requires -sw greater than 0\n'),
        ([], 'error: the following arguments are required: -r\n'),
        (['-sw', 'abc'], "error: argument -sw: invalid float value: 'abc'\n"),
        (['-sync_offset', '1', '-sw', '1'], 'error: -sync_offset and -sw are mutually exclusive: -sync_offset is a manual offset, -sw searches for one\n'),
        (['-sync_offset', '1', '-ss', '1'], 'error: -sync_offset and -ss are mutually exclusive: -ss without -sw is the manual offset\n'),
        (['-sync_offset', 'inf'], 'error: -sync_offset must be finite\n'),
        (['-vmaf_v0', '-vmaf_version', '1'], 'error: -vmaf_v0 and -vmaf_version are mutually exclusive: -vmaf_v0 equals -vmaf_version 0.6\n'),
        (['-disable_hfr', '-force_hfr'], 'error: -disable_hfr and -force_hfr are mutually exclusive\n'),
        (['-subsample', '0'], 'error: -subsample must be an integer of at least 1\n'),
        (['-threads', '-1'], 'error: -threads must be an integer greater than or equal to zero\n'),
        (['-sync_only', '-sw', '1', '-start_frame', '0'], 'error: -start_frame and -frame_count measure VMAF; -sync_only does not\n'),
        (['-sync_only', '-sw', '1', '-jobs', '2'], 'error: -jobs splits the VMAF calculation; -sync_only does not compute VMAF\n'),
        (['-view', '3h', '-vmaf_v0'], 'error: -view: only for VMAF version 1 models, add 1 to -vmaf_version\n'),
        (['-force_hfr', '-vmaf_v0'], 'error: -force_hfr: only for VMAF version 1 models, add 1 to -vmaf_version\n'),
        (['-model', 'X'], "error: Invalid VMAF model: 'X'. Supported: HD, 4K\n"),
        (['-view', '1.5h'], "error: View '1.5h' is not available for -model HD. Supported: 3h, 5h\n"),
        (['-dynamic_range', 'hdr', '-vmaf_v0'], 'error: -dynamic_range hdr: VMAF 0.6 has no HDR models, remove 0.6 from -vmaf_version\n'),
        (['-reference_color', 'foo'], "error: Invalid -reference_color 'foo': expected hdr10, bt709 or key=value[,key=value] with the keys range, primaries, trc, matrix\n"),
        (['-reference_color', 'hdr10', '-dynamic_range', 'sdr'], 'error: -reference_color: only for the HDR models, not with -dynamic_range sdr\n'),
        (['-model_option', 'bad'], "error: Invalid -model_option 'bad': expected feature.option=value\n"),
        (['-enc_size', '12x'], "error: argument -enc_size: expected WxH, e.g. 1280x720, not '12x'\n"),
        (['-start_frame', '-1'], 'error: -start_frame must be an integer of at least 0, not -1\n'),
        (['-frame_count', '0'], 'error: -frame_count must be an integer of at least 1, not 0\n'),
        (['-start_frame', '0', '-subsample', '2'], 'error: -subsample: not supported with a frame range (-start_frame, -frame_count)\n'),
        (['-jobs', '0'], 'error: -jobs must be an integer of at least 1, not 0\n'),
        (['-jobs', '2', '-subsample', '2'], 'error: -jobs above 1 measures frame ranges in parallel, which do not support -subsample above 1\n'),
    ], ids=["sync-only-without-sw", "missing-reference", "invalid-number", "offset-sw", "offset-ss", "offset-inf",
            "v0-version", "hfr-both", "subsample", "threads", "sync-only-range", "sync-only-jobs", "view-v0",
            "force-hfr-v0", "model", "view-display", "hdr-v0", "color-value", "color-sdr", "model-option", "enc-size",
            "start-frame", "frame-count", "range-subsample", "jobs", "jobs-subsample"])
    def test_error(self, monkeypatch, capsys, argv, message):
        base = ['-d', 'd.mp4'] + ([] if argv == [] else ['-r', 'r.mp4'])
        with pytest.raises(SystemExit) as exit_info:
            self.parse(monkeypatch, *base, *argv)
        out, err = capsys.readouterr()
        assert (exit_info.value.code, err) == (2, message)
        assert out.startswith('usage: easyVmafPlus')

    def test_no_arguments(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exit_info:
            self.parse(monkeypatch)
        out, err = capsys.readouterr()
        assert (exit_info.value.code, out) == (1, '')
        assert err.startswith('usage: easyVmafPlus')

    def test_help_groups(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exit_info:
            self.parse(monkeypatch, '-h')
        out = capsys.readouterr().out
        assert exit_info.value.code == 0
        for group in ('input:', 'synchronization:', 'frame range:', 'models:', 'VMAF v1 parameters:', 'output:', 'execution:'):
            assert f'\n{group}\n' in out


def test_sigint_handler(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.handler(2, None)
    assert exit_info.value.code == 130
    assert capsys.readouterr().err == 'SIGINT or CTRL-C detected. Calculation interrupted.\n'


@pytest.fixture
def cli_run(monkeypatch, tmp_path, capsys):
    """cli.main against an empty ref.mp4 in tmp_path, with check_ffmpeg returning `check` (or raising it)
    and vmaf replaced by a fake. The fake returns `sync` from syncOffset and, in getVmaf, copies the
    libvmaf log of tests/data next to the distorted file, or raises `run_error`. `init_error` is raised
    when the fake is created."""
    reference = str(tmp_path / 'ref.mp4')
    open(reference, 'wb').close()

    def run(*argv, check=CHECK_OK, sync=(1.5, 40.032121), run_error=None, init_error=None, files=('dist.mp4',)):
        for name in files:
            open(tmp_path / name, 'wb').close()
        runs = []

        class FakeVmaf:
            def __init__(self, mainSrc, refSrc, **kwargs):
                self.mainSrc, self.refSrc, self.kwargs = mainSrc, refSrc, kwargs
                self.main = SimpleNamespace(videoSrc=mainSrc)
                self.offset = 0
                self.models = []
                self.calls = []
                self.frames = []
                self.ffmpegQos = SimpleNamespace(vmafpath=None, vmaf_cambi_heatmap_path=None)
                self.pix_fmt = None
                self.hfr_active = False
                self.dynamic_range = 'hdr' if kwargs['dynamic_range'] == 'hdr' else 'sdr'
                self.color = {'reference': dict(HDR10), 'distorted': dict(HDR10)} if self.dynamic_range == 'hdr' else None
                self.start_frame, self.frame_count = kwargs['start_frame'], kwargs['frame_count']
                self.frames_scored = None
                runs.append(self)
                if init_error is not None:
                    raise init_error

            def syncOffset(self, syncWindow, start, reverse):
                self.calls.append(('syncOffset', syncWindow, start, reverse))
                self.offset = -sync[0] if reverse else sync[0]
                return [self.offset, sync[1]]

            def getVmaf(self):
                self.calls.append(('getVmaf', self.offset))
                if run_error is not None:
                    raise run_error
                log_fmt = self.kwargs['output_fmt']
                generation = 'v0' if '1' not in self.kwargs['vmaf_versions'] else 'v1'
                # the captured logs hold the SDR score names, also in the HDR output tests
                self.models = select_models('HD', ('0.6',) if generation == 'v0' else ('1',))
                self.pix_fmt = 'yuv420p' if generation == 'v0' else 'yuv420p10le'
                base = os.path.splitext(self.mainSrc)[0]
                suffix = '' if self.start_frame is None and self.frame_count is None else f'_f{self.start_frame or 0}-end'
                # the VMAF log is reserved as FFmpegQos.getVmaf reserves it
                log_file, self.ffmpegQos.vmafpath = create_unique_file(f'{base}_vmaf{suffix}.{log_fmt}')
                log_file.close()
                self.ffmpegQos.vmaf_cambi_heatmap_path = f'{base}_cambi_heatmap'
                shutil.copy(os.path.join(DATA, f'vmaf_{generation}_hd.{log_fmt}'), self.ffmpegQos.vmafpath)
                if suffix:
                    self.frames_scored = 5

            def getFrames(self, frame_numbers, folder):
                self.frames.append(list(frame_numbers))
                written = []
                for number in sorted(frame_numbers):
                    path = os.path.join(folder, f'{number}.tif')
                    Image.new('RGB', (64, 48), (255, 0, 0)).save(path, format='TIFF')
                    written.append((number, path, number / 30))
                return written

        def check_ffmpeg():
            if isinstance(check, Exception):
                raise check
            return dict(check)

        monkeypatch.setattr(cli, 'signal', lambda *args: None)
        monkeypatch.setattr(cli.logging, 'basicConfig', lambda **kwargs: None)
        monkeypatch.setattr(cli, 'check_ffmpeg', check_ffmpeg)
        monkeypatch.setattr(cli, 'vmaf', FakeVmaf)
        monkeypatch.setattr(sys, 'argv', ['easyVmafPlus', *argv])
        try:
            cli.main()
            code = None
        except SystemExit as exit_info:
            code = exit_info.code
        out, err = capsys.readouterr()
        return SimpleNamespace(code=code, out=out, err=err, runs=runs)

    return SimpleNamespace(run=run, dir=tmp_path, reference=reference)


def _means(scores):
    return {name: mean(values) for name, values in scores.items()}


def _summary(path):
    with open(path) as summary:
        return json.load(summary)


class TestMain:
    """cli.main: the startup check, the inputs, the sync, the summary file and the output per distorted file."""

    def test_text_output(self, cli_run):
        distorted = str(cli_run.dir / 'dist.mp4')
        result = cli_run.run('-d', distorted, '-r', cli_run.reference, '-ss', '1.5')
        assert result.code is None
        assert result.runs[0].calls == [('getVmaf', 1.5)]
        means = _means(V1_HD)
        lines = result.out.splitlines()
        for line in (f'Results: {distorted}', 'offset:  1.5  | psnr:  None', 'pix_fmt: yuv420p10le | HFR: off',
                     'dynamic range: sdr',
                     f"VMAF v1 HD:  {means['vmaf_v1_hd']}", f"VMAF v1 Phone:  {means['vmaf_v1_hd_phone']}",
                     f"VMAF output file path:  {cli_run.dir / 'dist_vmaf.json'}",
                     f"Summary file path:  {cli_run.dir / 'dist_vmaf_summary.json'}"):
            assert line in lines
        assert not any(line.startswith(('CAMBI Heatmap', 'color', 'range:')) for line in lines)

    def test_summary_file(self, cli_run):
        distorted = str(cli_run.dir / 'dist.mp4')
        result = cli_run.run('-d', distorted, '-r', cli_run.reference, '-ss', '1.5', '-json')
        assert _summary(cli_run.dir / 'dist_vmaf_summary.json') == json.loads(result.out)

    def test_vmaf_arguments(self, cli_run):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, '-subsample', '2',
                             '-fps', '-25', '-threads', '3', '-model', '4K', '-progress', '-endsync',
                             '-cambi_heatmap', '-output_fmt', 'csv', '-verbose', '-disable_hfr', '-view', '1.5h',
                             '-enc_size', '1280x720', '-enc_bitdepth', '8', '-model_option', 'cambi.topk=0.5',
                             '-bitdepth', '8', '-dynamic_range', 'sdr')
        run = result.runs[0]
        assert (run.mainSrc, run.refSrc) == (str(cli_run.dir / 'dist.mp4'), cli_run.reference)
        assert run.kwargs == dict(
            DEFAULT_KWARGS, loglevel='verbose', subsample=2, model='4K', output_fmt='csv', threads=3,
            print_progress=True, end_sync=True, manual_fps=25.0, cambi_heatmap=True, hfr='off', views=['1.5h'],
            enc_size=(1280, 720), enc_bitdepth=8, model_options=['cambi.topk=0.5'], bitdepth='8', dynamic_range='sdr')
        assert f"CAMBI Heatmap output path:  {cli_run.dir / 'dist_cambi_heatmap'}" in result.out.splitlines()

    def test_default_arguments(self, cli_run):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference)
        assert result.runs[0].kwargs == DEFAULT_KWARGS
        assert result.runs[0].calls == [('getVmaf', 0)]

    def test_manual_offset_reverse(self, cli_run):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, '-ss', '2', '-reverse')
        assert result.runs[0].calls == [('getVmaf', -2.0)]

    @pytest.mark.parametrize("value, offset", [('1.5', 1.5), ('-1', -1.0), ('-0', 0.0)], ids=["positive", "negative", "zero"])
    def test_sync_offset(self, cli_run, value, offset):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, '-sync_offset', value, '-json')
        assert result.runs[0].calls == [('getVmaf', offset)]
        assert json.loads(result.out)['sync'] == {'offset': offset, 'psnr': None}

    @pytest.mark.parametrize("reverse, offset", [(False, 1.5), (True, -1.5)], ids=["default", "reverse"])
    def test_sync(self, cli_run, reverse, offset):
        argv = ['-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, '-sw', '1', '-ss', '0.5']
        result = cli_run.run(*argv + (['-reverse'] if reverse else []))
        assert result.runs[0].calls == [('syncOffset', 1.0, 0.5, reverse), ('getVmaf', offset)]
        assert f'offset:  {offset}  | psnr:  40.032121' in result.out.splitlines()

    def test_sync_only(self, cli_run):
        distorted = str(cli_run.dir / 'dist.mp4')
        result = cli_run.run('-d', distorted, '-r', cli_run.reference, '-sw', '1', '-sync_only')
        assert result.runs[0].calls == [('syncOffset', 1.0, 0.0, False)]
        summary = cli_run.dir / 'dist_sync_summary.json'
        assert result.out == f'Results: {distorted} | offset: 1.5 | psnr: 40.032121\nSummary file path: {summary}\n'
        assert _summary(summary) == {'distorted': distorted, 'reference': cli_run.reference,
                                     'sync': {'offset': 1.5, 'psnr': 40.032121}}

    def test_sync_only_json(self, cli_run):
        distorted = str(cli_run.dir / 'dist.mp4')
        result = cli_run.run('-d', distorted, '-r', cli_run.reference, '-sw', '1', '-sync_only', '-json')
        assert json.loads(result.out) == {'distorted': distorted, 'reference': cli_run.reference,
                                          'sync': {'offset': 1.5, 'psnr': 40.032121}}

    @pytest.mark.parametrize("log_fmt", ['json', 'xml', 'csv'])
    def test_json(self, cli_run, log_fmt):
        distorted = str(cli_run.dir / 'dist.mp4')
        result = cli_run.run('-d', distorted, '-r', cli_run.reference, '-ss', '1.5', '-json', '-output_fmt', log_fmt)
        means = _means(V1_HD)
        assert json.loads(result.out) == {
            'distorted': distorted, 'reference': cli_run.reference, 'sync': {'offset': 1.5, 'psnr': None},
            'vmaf': {'model': 'HD', 'pix_fmt': 'yuv420p10le', 'hfr': False, 'dynamic_range': 'sdr',
                     'vmaf_v1_hd': round(means['vmaf_v1_hd'], 6),
                     'vmaf_v1_hd_phone': round(means['vmaf_v1_hd_phone'], 6),
                     'output_file': str(cli_run.dir / f'dist_vmaf.{log_fmt}')}}
        assert _summary(cli_run.dir / 'dist_vmaf_summary.json') == json.loads(result.out)

    def test_json_v0_with_cambi_heatmap(self, cli_run):
        distorted = str(cli_run.dir / 'dist.mp4')
        result = cli_run.run('-d', distorted, '-r', cli_run.reference, '-json', '-vmaf_v0', '-cambi_heatmap')
        means = _means(V0_HD)
        assert json.loads(result.out)['vmaf'] == {
            'model': 'HD', 'pix_fmt': 'yuv420p', 'hfr': False, 'dynamic_range': 'sdr',
            'vmaf_hd': round(means['vmaf_hd'], 6), 'vmaf_hd_neg': round(means['vmaf_hd_neg'], 6),
            'vmaf_hd_phone': round(means['vmaf_hd_phone'], 6),
            'output_file': str(cli_run.dir / 'dist_vmaf.json'),
            'cambi_heatmap_path': str(cli_run.dir / 'dist_cambi_heatmap')}

    def test_hdr_output(self, cli_run):
        distorted = str(cli_run.dir / 'dist.mp4')
        result = cli_run.run('-d', distorted, '-r', cli_run.reference, '-dynamic_range', 'hdr', '-reference_color',
                             'hdr10', '-distorted_color', 'hdr10')
        lines = result.out.splitlines()
        assert 'dynamic range: hdr' in lines
        assert ('color (range/primaries/trc/matrix): reference tv/bt2020/smpte2084/bt2020nc | '
                'distorted tv/bt2020/smpte2084/bt2020nc') in lines
        assert _summary(cli_run.dir / 'dist_vmaf_summary.json')['vmaf']['color'] == {
            'reference': HDR10, 'distorted': HDR10}

    def test_range_output(self, cli_run):
        distorted = str(cli_run.dir / 'dist.mp4')
        result = cli_run.run('-d', distorted, '-r', cli_run.reference, '-start_frame', '100')
        lines = result.out.splitlines()
        assert 'range: frames 100-104 (5 frames)' in lines
        assert f"VMAF output file path:  {cli_run.dir / 'dist_vmaf_f100-end.json'}" in lines
        assert f"Summary file path:  {cli_run.dir / 'dist_vmaf_f100-end_summary.json'}" in lines
        assert _summary(cli_run.dir / 'dist_vmaf_f100-end_summary.json')['vmaf']['range'] == {
            'start_frame': 100, 'frame_count': None, 'frames_scored': 5}

    def test_v0_text_labels(self, cli_run):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, '-vmaf_v0')
        means = _means(V0_HD)
        lines = result.out.splitlines()
        for label, name in (('VMAF HD', 'vmaf_hd'), ('VMAF Neg', 'vmaf_hd_neg'), ('VMAF Phone', 'vmaf_hd_phone')):
            assert f'{label}:  {means[name]}' in lines

    def test_plots(self, cli_run):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, '-plot', '-json')
        vmaf_block = json.loads(result.out)['vmaf']
        assert (vmaf_block['plot_path'], vmaf_block['histo_path']) == (
            str(cli_run.dir / 'dist_vmaf_plot.png'), str(cli_run.dir / 'dist_vmaf_histo.png'))
        assert 'low_frames_path' not in vmaf_block
        assert not (cli_run.dir / 'ref_vmaf_combined_plot.png').exists()

    def test_plots_text_output(self, cli_run):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, '-plot')
        lines = result.out.splitlines()
        assert f"Plot output path:  {cli_run.dir / 'dist_vmaf_plot.png'}" in lines
        assert f"Percentile plot output path:  {cli_run.dir / 'dist_vmaf_histo.png'}" in lines

    def test_low_frames(self, cli_run):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, '-low_frames', '-json')
        folder = cli_run.dir / 'dist_vmaf_lowframes'
        assert json.loads(result.out)['vmaf']['low_frames_path'] == str(folder)
        assert os.listdir(folder) == ['dist_vmaf_VMAF52_frame002.tif']
        assert result.runs[0].frames == [[2]]

    def test_low_frames_text_output(self, cli_run):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, '-low_frames')
        assert f"Low frames output path:  {cli_run.dir / 'dist_vmaf_lowframes'}" in result.out.splitlines()

    def test_combined_plots(self, cli_run, caplog):
        with caplog.at_level(logging.INFO, logger='easyVmafPlus.cli'):
            result = cli_run.run('-d', str(cli_run.dir / 'dist_*.mp4'), '-r', cli_run.reference, '-plot', '-json',
                                 files=('dist_a.mp4', 'dist_b.mp4'))
        assert len(result.out.splitlines()) == 2
        for name in ('ref_vmaf_combined_plot.png', 'ref_vmaf_combined_histo.png'):
            assert (cli_run.dir / name).exists()
        assert f"Combined plot output path: {cli_run.dir / 'ref_vmaf_combined_plot.png'}" in caplog.messages

    def test_combined_plots_text_output(self, cli_run):
        result = cli_run.run('-d', str(cli_run.dir / 'dist_*.mp4'), '-r', cli_run.reference, '-plot',
                             files=('dist_a.mp4', 'dist_b.mp4'))
        lines = result.out.splitlines()
        assert f"Combined plot output path:  {cli_run.dir / 'ref_vmaf_combined_plot.png'}" in lines
        assert f"Combined percentile plot output path:  {cli_run.dir / 'ref_vmaf_combined_histo.png'}" in lines

    def test_reference_among_distorted_files(self, cli_run):
        cli_run.run('-d', str(cli_run.dir / '*.mp4'), '-r', cli_run.reference, '-plot')
        assert {'ref_vmaf_plot.png', 'ref_vmaf_histo.png', 'ref_vmaf_combined_plot.png',
                'ref_vmaf_combined_histo.png'} <= set(os.listdir(cli_run.dir))

    def test_second_run_keeps_first(self, cli_run):
        argv = ('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, '-plot', '-low_frames')
        cli_run.run(*argv)
        first = {name: (cli_run.dir / name).stat().st_mtime_ns
                 for name in ('dist_vmaf.json', 'dist_vmaf_summary.json', 'dist_vmaf_plot.png', 'dist_vmaf_histo.png')}
        result = cli_run.run(*argv)
        lines = result.out.splitlines()
        assert f"VMAF output file path:  {cli_run.dir / 'dist_vmaf_2.json'}" in lines
        assert f"Summary file path:  {cli_run.dir / 'dist_vmaf_2_summary.json'}" in lines
        for name in ('dist_vmaf_2_plot.png', 'dist_vmaf_2_histo.png', 'dist_vmaf_2_lowframes', 'dist_vmaf_2_summary.json'):
            assert (cli_run.dir / name).exists()
        assert {name: (cli_run.dir / name).stat().st_mtime_ns for name in first} == first

    def test_every_matching_file(self, cli_run):
        result = cli_run.run('-d', str(cli_run.dir / 'dist_*.mp4'), '-r', cli_run.reference, '-json',
                             files=('dist_a.mp4', 'dist_b.mp4'))
        expected = {str(cli_run.dir / 'dist_a.mp4'), str(cli_run.dir / 'dist_b.mp4')}
        assert {run.mainSrc for run in result.runs} == expected
        assert {json.loads(line)['distorted'] for line in result.out.splitlines()} == expected

    def test_home_directory_pattern(self, cli_run, monkeypatch):
        monkeypatch.setenv('HOME', str(cli_run.dir))
        result = cli_run.run('-d', '~/dist.mp4', '-r', cli_run.reference)
        assert result.runs[0].mainSrc == str(cli_run.dir / 'dist.mp4')

    def test_unsupported_output_format(self, cli_run, caplog):
        with caplog.at_level(logging.WARNING, logger='easyVmafPlus.cli'):
            result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, '-output_fmt', 'yuv')
        assert "output_fmt 'yuv' not supported, using json" in caplog.messages
        assert result.runs[0].kwargs['output_fmt'] == 'json'

    @pytest.mark.parametrize("check, message", [
        (CHECK_OK, "FFmpeg 9.0 detected. Built-in models: available. VMAF v1 models: available. HDR-VMAF: available."),
        (dict(CHECK_OK, hdr_vmaf=False), "FFmpeg 9.0 detected. Built-in models: available. VMAF v1 models: available. HDR-VMAF: not available."),
    ], ids=["hdr", "no-hdr"])
    def test_startup_log(self, cli_run, caplog, check, message):
        with caplog.at_level(logging.INFO, logger='easyVmafPlus.cli'):
            cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, check=check)
        assert message in caplog.messages

    @pytest.mark.parametrize("argv, check, message", [
        ([], RuntimeError("ffmpeg not found on PATH. Install FFmpeg >= 9.0 built with --enable-libvmaf, "
                          "or point the FFMPEG environment variable to it."),
         "ffmpeg not found on PATH. Install FFmpeg >= 9.0 built with --enable-libvmaf, "
         "or point the FFMPEG environment variable to it."),
        ([], dict(CHECK_OK, version_str='8.1', meets_minimum=False),
         "FFmpeg 8.1 detected. easyVmafPlus requires FFmpeg >= 9.0 built with --enable-libvmaf."),
        ([], dict(CHECK_OK, libvmaf=False),
         "FFmpeg 9.0 has no libvmaf filter. Build FFmpeg with --enable-libvmaf."),
        (['-vmaf_v0'], dict(CHECK_OK, builtin_models=False),
         "FFmpeg 9.0 is installed but libvmaf built-in models are not available. "
         "Rebuild libvmaf with '-Dbuilt_in_models=true' and recompile FFmpeg."),
        (['-vmaf_version', '1', '0.6'], dict(CHECK_OK, builtin_models=False),
         "FFmpeg 9.0 is installed but libvmaf built-in models are not available. "
         "Rebuild libvmaf with '-Dbuilt_in_models=true' and recompile FFmpeg."),
        ([], dict(CHECK_OK, v1_models=False),
         "FFmpeg 9.0 cannot run the VMAF v1 models. easyVmafPlus needs libvmaf 3.2.1 or later. "
         "The v0.6.1 models run with -vmaf_v0."),
    ], ids=["not-found", "too-old", "no-libvmaf", "no-builtin-models", "no-builtin-models-both", "no-v1-models"])
    def test_startup_check_fails(self, cli_run, argv, check, message):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, *argv, check=check)
        assert (result.code, result.err, result.out, result.runs) == (1, f'[easyVmafPlus] ERROR: {message}\n', '', [])

    @pytest.mark.parametrize("argv, check", [
        (['-vmaf_v0'], dict(CHECK_OK, v1_models=False)),
        ([], dict(CHECK_OK, builtin_models=False)),
        ([], dict(CHECK_OK, hdr_vmaf=False)),
    ], ids=["v0-without-v1-models", "v1-without-builtin-models", "sdr-without-hdr"])
    def test_startup_check_per_generation(self, cli_run, argv, check):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, *argv, check=check)
        assert result.code is None
        assert len(result.runs) == 1

    def test_reference_not_found(self, cli_run):
        missing = str(cli_run.dir / 'missing.mp4')
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', missing)
        assert (result.code, result.err, result.runs) == (
            1, f'[easyVmafPlus] ERROR: Reference Video file not found: {missing}\n', [])

    def test_no_distorted_file(self, cli_run):
        pattern = str(cli_run.dir / 'none_*.mp4')
        result = cli_run.run('-d', pattern, '-r', cli_run.reference)
        assert (result.code, result.err, result.runs) == (
            1, f'[easyVmafPlus] ERROR: Distorted Video files not found with the given pattern/name: {pattern}\n', [])

    @pytest.mark.parametrize("errors", [
        {'run_error': UnsupportedFramerateError('No deinterlace filter available for the given framerate combination.')},
        {'init_error': UnsupportedModelConfigError('The distorted video d.mp4 is HLG (arib-std-b67)')},
        {'init_error': UnsupportedRangeError('A frame range needs inputs that seek to an exact frame')},
        {'run_error': RuntimeError('ffmpeg exited with an error')},
        {'run_error': subprocess.CalledProcessError(1, ['ffmpeg'])},
        {'run_error': OSError('No space left on device')},
    ], ids=["unsupported-framerate", "model-config", "range", "progress-error", "ffmpeg-error", "os-error"])
    def test_run_error(self, cli_run, errors):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, **errors)
        error = next(iter(errors.values()))
        assert (result.code, result.err, result.out) == (1, f'[easyVmafPlus] ERROR: {error}\n', '')
        assert not any(name.endswith('_summary.json') for name in os.listdir(cli_run.dir))
