"""Tests for cli.py: the -json result, the reading of the VMAF log, the arguments, the startup check
and the run per distorted file, with check_ffmpeg and vmaf replaced by fakes. tests/data/vmaf_v1_hd.*
and tests/data/vmaf_v0_hd.json are logs of libvmaf 3.2.1 over five frames."""

import json
import logging
import os
import shutil
import subprocess
import sys
from statistics import mean
from types import SimpleNamespace

import pytest

from easyVmafPlus import cli
from easyVmafPlus.FFmpeg import select_models
from easyVmafPlus.Vmaf import UnsupportedFramerateError

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
CHECK_OK = {'version_str': '9.0', 'meets_minimum': True, 'libvmaf': True, 'builtin_models': True, 'v1_models': True}


class TestBuildResult:
    """cli._build_result: the -json object of one distorted file."""

    def test_sync(self):
        assert cli._build_result('d.mp4', 'r.mp4', 1.5000004, 40.0321214, 'HD') == {
            'distorted': 'd.mp4', 'reference': 'r.mp4', 'sync': {'offset': 1.5, 'psnr': 40.032121}}

    def test_without_sync(self):
        assert cli._build_result('d.mp4', 'r.mp4', None, None, 'HD')['sync'] == {'offset': 0.0, 'psnr': None}

    def test_vmaf(self):
        result = cli._build_result('d.mp4', 'r.mp4', 0, None, '4K',
                                   vmaf_scores={'vmaf_v1_4k': 87.2551875058, 'vmaf_v1_4k_3h': 97.4560392},
                                   vmaf_output_file='d_vmaf.json', cambi_heatmap_path='d_cambi_heatmap')
        assert result['vmaf'] == {'model': '4K', 'vmaf_v1_4k': 87.255188, 'vmaf_v1_4k_3h': 97.456039,
                                  'output_file': 'd_vmaf.json', 'cambi_heatmap_path': 'd_cambi_heatmap'}

    def test_vmaf_without_paths(self):
        result = cli._build_result('d.mp4', 'r.mp4', 0, None, 'HD', vmaf_scores={'vmaf_hd': 90.0})
        assert result['vmaf'] == {'model': 'HD', 'vmaf_hd': 90.0}


class TestReadScores:
    """cli._read_scores: the per-frame scores of the given names from a json, xml or csv log."""

    @pytest.mark.parametrize("log_fmt", ['json', 'xml', 'csv'])
    def test_v1_hd(self, log_fmt):
        scores = cli._read_scores(os.path.join(DATA, f'vmaf_v1_hd.{log_fmt}'), log_fmt, list(V1_HD))
        assert scores == {name: pytest.approx(values) for name, values in V1_HD.items()}

    def test_v0_hd(self):
        scores = cli._read_scores(os.path.join(DATA, 'vmaf_v0_hd.json'), 'json', list(V0_HD))
        assert scores == {name: pytest.approx(values) for name, values in V0_HD.items()}

    def test_selected_names(self):
        scores = cli._read_scores(os.path.join(DATA, 'vmaf_v1_hd.csv'), 'csv', ['vmaf_v1_hd_phone'])
        assert scores == {'vmaf_v1_hd_phone': pytest.approx(V1_HD['vmaf_v1_hd_phone'])}

    def test_other_format_reads_json(self):
        scores = cli._read_scores(os.path.join(DATA, 'vmaf_v1_hd.json'), 'txt', ['vmaf_v1_hd'])
        assert scores == {'vmaf_v1_hd': pytest.approx(V1_HD['vmaf_v1_hd'])}


class TestArguments:
    """cli.get_args and MyParser: flags, defaults and argument errors."""

    @staticmethod
    def parse(monkeypatch, *argv):
        monkeypatch.setattr(sys, 'argv', ['easyVmafPlus', *argv])
        return cli.get_args()

    def test_defaults(self, monkeypatch):
        assert vars(self.parse(monkeypatch, '-d', 'd.mp4', '-r', 'r.mp4')) == {
            'd': 'd.mp4', 'r': 'r.mp4', 'sw': 0, 'ss': 0, 'fps': 0, 'n': 1, 'reverse': False, 'model': 'HD',
            'vmaf_v0': False, 'disable_hfr': False, 'threads': 0, 'verbose': False, 'progress': False,
            'endsync': False, 'output_fmt': 'json', 'cambi_heatmap': False, 'sync_only': False, 'json': False}

    def test_values(self, monkeypatch):
        args = self.parse(monkeypatch, '-d', 'd*.mp4', '-r', 'r.mp4', '-sw', '2', '-ss', '1.5', '-fps', '25',
                          '-subsample', '3', '-reverse', '-model', '4K', '-vmaf_v0', '-disable_hfr', '-threads', '8',
                          '-verbose', '-progress', '-endsync', '-output_fmt', 'xml', '-cambi_heatmap', '-sync_only',
                          '-json')
        assert vars(args) == {
            'd': 'd*.mp4', 'r': 'r.mp4', 'sw': 2.0, 'ss': 1.5, 'fps': 25.0, 'n': 3, 'reverse': True, 'model': '4K',
            'vmaf_v0': True, 'disable_hfr': True, 'threads': 8, 'verbose': True, 'progress': True,
            'endsync': True, 'output_fmt': 'xml', 'cambi_heatmap': True, 'sync_only': True, 'json': True}

    @pytest.mark.parametrize("argv, message", [
        (['-d', 'd.mp4', '-r', 'r.mp4', '-sync_only'], 'error: -sync_only requires -sw greater than 0\n'),
        (['-d', 'd.mp4'], 'error: the following arguments are required: -r\n'),
        (['-d', 'd.mp4', '-r', 'r.mp4', '-sw', 'abc'], "error: argument -sw: invalid float value: 'abc'\n"),
    ], ids=["sync-only-without-sw", "missing-reference", "invalid-number"])
    def test_error(self, monkeypatch, capsys, argv, message):
        with pytest.raises(SystemExit) as exit_info:
            self.parse(monkeypatch, *argv)
        out, err = capsys.readouterr()
        assert (exit_info.value.code, err) == (2, message)
        assert out.startswith('usage: easyVmafPlus')

    def test_no_arguments(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exit_info:
            self.parse(monkeypatch)
        out, err = capsys.readouterr()
        assert (exit_info.value.code, out) == (1, '')
        assert err.startswith('usage: easyVmafPlus')


def test_sigint_handler(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.handler(2, None)
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == 'SIGINT or CTRL-C detected. Exiting gracefully\n'


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
                self.offset = 0
                self.models = []
                self.calls = []
                self.ffmpegQos = SimpleNamespace(vmafpath=None, vmaf_cambi_heatmap_path=None)
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
                generation = 'v0' if self.kwargs['vmaf_v0'] else 'v1'
                self.models = select_models('HD', vmaf_v0=self.kwargs['vmaf_v0'])
                base = os.path.splitext(self.mainSrc)[0]
                self.ffmpegQos.vmafpath = f'{base}_vmaf.{log_fmt}'
                self.ffmpegQos.vmaf_cambi_heatmap_path = f'{base}_cambi_heatmap'
                shutil.copy(os.path.join(DATA, f'vmaf_{generation}_hd.{log_fmt}'), self.ffmpegQos.vmafpath)

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


class TestMain:
    """cli.main: the startup check, the inputs, the sync and the output per distorted file."""

    def test_text_output(self, cli_run):
        distorted = str(cli_run.dir / 'dist.mp4')
        result = cli_run.run('-d', distorted, '-r', cli_run.reference, '-ss', '1.5')
        assert result.code is None
        assert result.runs[0].calls == [('getVmaf', 1.5)]
        means = _means(V1_HD)
        lines = result.out.splitlines()
        for line in (f'Results: {distorted}', 'offset:  1.5  | psnr:  None',
                     f"VMAF v1 HD:  {means['vmaf_v1_hd']}", f"VMAF v1 Phone:  {means['vmaf_v1_hd_phone']}",
                     f"VMAF output file path:  {cli_run.dir / 'dist_vmaf.json'}"):
            assert line in lines
        assert not any(line.startswith('CAMBI Heatmap') for line in lines)

    def test_vmaf_arguments(self, cli_run):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, '-subsample', '-2',
                             '-fps', '-25', '-threads', '3', '-model', '4K', '-progress', '-endsync',
                             '-cambi_heatmap', '-output_fmt', 'csv', '-verbose', '-disable_hfr')
        run = result.runs[0]
        assert (run.mainSrc, run.refSrc) == (str(cli_run.dir / 'dist.mp4'), cli_run.reference)
        assert run.kwargs == {'loglevel': 'verbose', 'subsample': 2, 'model': '4K', 'output_fmt': 'csv', 'threads': 3,
                              'print_progress': True, 'end_sync': True, 'manual_fps': 25.0, 'cambi_heatmap': True,
                              'vmaf_v0': False, 'disable_hfr': True}
        assert f"CAMBI Heatmap output path:  {cli_run.dir / 'dist_cambi_heatmap'}" in result.out.splitlines()

    def test_default_arguments(self, cli_run):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference)
        assert result.runs[0].kwargs == {
            'loglevel': 'info', 'subsample': 1, 'model': 'HD', 'output_fmt': 'json', 'threads': 0,
            'print_progress': False, 'end_sync': False, 'manual_fps': 0, 'cambi_heatmap': False, 'vmaf_v0': False,
            'disable_hfr': False}
        assert result.runs[0].calls == [('getVmaf', 0)]

    def test_manual_offset_reverse(self, cli_run):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, '-ss', '2', '-reverse')
        assert result.runs[0].calls == [('getVmaf', -2.0)]

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
        assert result.out == f'Results: {distorted} | offset: 1.5 | psnr: 40.032121\n'

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
            'vmaf': {'model': 'HD', 'vmaf_v1_hd': round(means['vmaf_v1_hd'], 6),
                     'vmaf_v1_hd_phone': round(means['vmaf_v1_hd_phone'], 6),
                     'output_file': str(cli_run.dir / f'dist_vmaf.{log_fmt}')}}

    def test_json_v0_with_cambi_heatmap(self, cli_run):
        distorted = str(cli_run.dir / 'dist.mp4')
        result = cli_run.run('-d', distorted, '-r', cli_run.reference, '-json', '-vmaf_v0', '-cambi_heatmap')
        means = _means(V0_HD)
        assert json.loads(result.out)['vmaf'] == {
            'model': 'HD', 'vmaf_hd': round(means['vmaf_hd'], 6), 'vmaf_hd_neg': round(means['vmaf_hd_neg'], 6),
            'vmaf_hd_phone': round(means['vmaf_hd_phone'], 6),
            'output_file': str(cli_run.dir / 'dist_vmaf.json'),
            'cambi_heatmap_path': str(cli_run.dir / 'dist_cambi_heatmap')}

    def test_v0_text_labels(self, cli_run):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, '-vmaf_v0')
        means = _means(V0_HD)
        lines = result.out.splitlines()
        for label, name in (('VMAF HD', 'vmaf_hd'), ('VMAF Neg', 'vmaf_hd_neg'), ('VMAF Phone', 'vmaf_hd_phone')):
            assert f'{label}:  {means[name]}' in lines

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

    @pytest.mark.parametrize("argv, message", [
        ([], "FFmpeg 9.0 detected. VMAF v1 models: available."),
        (['-vmaf_v0'], "FFmpeg 9.0 detected. Built-in models: available."),
    ], ids=["v1", "v0"])
    def test_startup_log(self, cli_run, caplog, argv, message):
        with caplog.at_level(logging.INFO, logger='easyVmafPlus.cli'):
            cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, *argv)
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
        ([], dict(CHECK_OK, v1_models=False),
         "FFmpeg 9.0 cannot run the VMAF v1 models. easyVmafPlus needs libvmaf 3.2.1 or later. "
         "The v0.6.1 models run with -vmaf_v0."),
    ], ids=["not-found", "too-old", "no-libvmaf", "no-builtin-models", "no-v1-models"])
    def test_startup_check_fails(self, cli_run, argv, check, message):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, *argv, check=check)
        assert (result.code, result.err, result.out, result.runs) == (1, f'[easyVmafPlus] ERROR: {message}\n', '', [])

    @pytest.mark.parametrize("argv, check", [
        (['-vmaf_v0'], dict(CHECK_OK, v1_models=False)),
        ([], dict(CHECK_OK, builtin_models=False)),
    ], ids=["v0-without-v1-models", "v1-without-builtin-models"])
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
        {'init_error': ValueError("Invalid VMAF model: '8K'. Supported: HD, 4K")},
        {'run_error': RuntimeError('ffmpeg exited with an error')},
        {'run_error': subprocess.CalledProcessError(1, ['ffmpeg'])},
    ], ids=["unsupported-framerate", "invalid-model", "progress-error", "ffmpeg-error"])
    def test_run_error(self, cli_run, errors):
        result = cli_run.run('-d', str(cli_run.dir / 'dist.mp4'), '-r', cli_run.reference, **errors)
        error = next(iter(errors.values()))
        assert (result.code, result.err, result.out) == (1, f'[easyVmafPlus] ERROR: {error}\n', '')
