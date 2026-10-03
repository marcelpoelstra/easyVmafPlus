"""Tests for the VMAF model sets, the libvmaf model string, the HFR choice, the CAMBI
encode parameters, the startup check and the reading of the VMAF log."""

import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from easyVmafPlus import FFmpeg
from easyVmafPlus.config import ffmpeg as FFMPEG
from easyVmafPlus.FFmpeg import FFmpegQos, ModelConfig, check_ffmpeg, read_vmaf_log, select_models, v1_model_path
from easyVmafPlus.Vmaf import vmaf

BS = "\\"
V1_VARIANTS = ['3d0h', '5d0h', '1d5h_2160', '3d0h_2160']
PROBE_INPUT = 'nullsrc=s=320x240:r=25:d=0.2'
# The v1 phone model stops on 320x240 frames with "SpEED: image too small"; 640x480 passes
LOG_INPUT = 'nullsrc=s=640x480:r=25:d=0.2'


def _run_libvmaf(model_value, log_fmt='json', log_path=os.devnull, source=PROBE_INPUT):
    """ffmpeg's libvmaf filter with this model= value on two lavfi inputs; returns the exit code"""
    cmd = [FFMPEG, '-hide_banner', '-loglevel', 'error',
           '-f', 'lavfi', '-i', source,
           '-f', 'lavfi', '-i', source,
           '-lavfi', f'libvmaf=model={model_value}:log_fmt={log_fmt}'
                     f':log_path={FFmpegQos._escape_filter_value(log_path)}',
           '-f', 'null', '-']
    return subprocess.run(cmd, capture_output=True, text=True).returncode


def _v1_runs(model_path):
    """True when libvmaf runs the v1 model file at model_path"""
    model = ModelConfig('vmaf_v1_hd', 'VMAF v1 HD', path=model_path, params={'cambi.enc_bitdepth': '8'})
    return _run_libvmaf(FFmpegQos._build_model_string([model])) == 0


V1_AVAILABLE = FFMPEG is not None and _v1_runs(v1_model_path('3d0h'))
needs_v1 = pytest.mark.skipif(not V1_AVAILABLE, reason="needs ffmpeg with libvmaf 3.2.1 or later on PATH")


class TestSelectModels:
    """select_models: model generation, -model value and HFR variants."""

    def test_v1_hd(self):
        models = select_models('HD')
        assert [(m.name, m.label, os.path.basename(m.path), m.version) for m in models] == [
            ('vmaf_v1_hd', 'VMAF v1 HD', 'vmaf_v1.0.16_3d0h.json', None),
            ('vmaf_v1_hd_phone', 'VMAF v1 Phone', 'vmaf_v1.0.16_5d0h.json', None),
        ]

    def test_v1_4k(self):
        models = select_models('4K')
        assert [(m.name, m.label, os.path.basename(m.path)) for m in models] == [
            ('vmaf_v1_4k', 'VMAF v1 4K', 'vmaf_v1.0.16_1d5h_2160.json'),
            ('vmaf_v1_4k_3h', 'VMAF v1 4K 3H', 'vmaf_v1.0.16_3d0h_2160.json'),
        ]

    def test_v1_hfr(self):
        models = select_models('4K', hfr=True)
        assert [(m.name, m.label, os.path.basename(m.path)) for m in models] == [
            ('vmaf_v1_4k_hfr', 'VMAF v1 4K HFR', 'vmaf_v1.0.16_hfr_1d5h_2160.json'),
            ('vmaf_v1_4k_3h_hfr', 'VMAF v1 4K 3H HFR', 'vmaf_v1.0.16_hfr_3d0h_2160.json'),
        ]

    @pytest.mark.parametrize("hfr", [False, True], ids=["standard", "hfr-ignored"])
    def test_v0_hd(self, hfr):
        models = select_models('HD', vmaf_v0=True, hfr=hfr)
        assert [(m.name, m.label, m.version, m.path, m.params) for m in models] == [
            ('vmaf_hd', 'VMAF HD', 'vmaf_v0.6.1', None, {}),
            ('vmaf_hd_neg', 'VMAF Neg', 'vmaf_v0.6.1neg', None, {}),
            ('vmaf_hd_phone', 'VMAF Phone', 'vmaf_v0.6.1', None, {'enable_transform': 'true'}),
        ]

    def test_v0_4k(self):
        models = select_models('4K', vmaf_v0=True)
        assert [(m.name, m.label, m.version) for m in models] == [('vmaf_4k', 'VMAF 4K', 'vmaf_4k_v0.6.1')]

    @pytest.mark.parametrize("vmaf_v0", [False, True], ids=["v1", "v0"])
    def test_invalid_model(self, vmaf_v0):
        with pytest.raises(ValueError, match=r"Invalid VMAF model: '8K'\. Supported: HD, 4K"):
            select_models('8K', vmaf_v0=vmaf_v0)

    def test_params_are_copies(self):
        select_models('HD', vmaf_v0=True)[2].params['x'] = 'y'
        assert select_models('HD', vmaf_v0=True)[2].params == {'enable_transform': 'true'}

    @pytest.mark.parametrize("hfr", [False, True], ids=["standard", "hfr"])
    @pytest.mark.parametrize("variant", V1_VARIANTS)
    def test_bundled_model_file_exists(self, variant, hfr):
        assert os.path.isfile(v1_model_path(variant, hfr=hfr))


class TestBuildModelString:
    """FFmpegQos._build_model_string: built-in versions, escaped model file paths and parameters."""

    def test_v0_hd_unchanged(self):
        """The v0 HD model= value of 269b356."""
        expected = ('version=vmaf_v0.6.1' + BS * 2 + ':name=vmaf_hd'
                    + '|version=vmaf_v0.6.1neg' + BS * 2 + ':name=vmaf_hd_neg'
                    + '|version=vmaf_v0.6.1' + BS * 2 + ':name=vmaf_hd_phone' + BS * 2 + ':enable_transform=true')
        assert FFmpegQos._build_model_string(select_models('HD', vmaf_v0=True)) == expected

    def test_v0_4k_unchanged(self):
        """The v0 4K model= value of 269b356."""
        expected = 'version=vmaf_4k_v0.6.1' + BS * 2 + ':name=vmaf_4k'
        assert FFmpegQos._build_model_string(select_models('4K', vmaf_v0=True)) == expected

    def test_v1_escaped_path_and_parameters(self):
        model = ModelConfig('vmaf_v1_hd', 'VMAF v1 HD', path='/m/a:b.json',
                            params={'cambi.enc_width': '1280', 'cambi.enc_height': '720',
                                    'cambi.enc_bitdepth': '8'})
        expected = ('path=/m/a' + BS * 6 + ':b.json'
                    + BS * 2 + ':name=vmaf_v1_hd'
                    + BS * 2 + ':cambi.enc_width=1280'
                    + BS * 2 + ':cambi.enc_height=720'
                    + BS * 2 + ':cambi.enc_bitdepth=8')
        assert FFmpegQos._build_model_string([model]) == expected

    def test_models_joined(self):
        models = [ModelConfig('a', 'A', path='/m/a.json'), ModelConfig('b', 'B', path='/m/b.json')]
        expected = 'path=/m/a.json' + BS * 2 + ':name=a|path=/m/b.json' + BS * 2 + ':name=b'
        assert FFmpegQos._build_model_string(models) == expected


@needs_v1
class TestModelPathRoundTrip:
    """A bundled model file in a directory with filter-special characters still loads."""

    @pytest.mark.parametrize("dirname", ["it's", "a:b", "x,y;z", "[set]"],
                             ids=["quote", "colon", "comma-semicolon", "brackets"])
    def test_special_directory(self, tmp_path, dirname):
        directory = tmp_path / dirname
        directory.mkdir()
        model_file = directory / 'vmaf_v1.0.16_3d0h.json'
        shutil.copy(v1_model_path('3d0h'), model_file)
        assert _v1_runs(str(model_file))


def _vmaf_stub(ref_rate='25/1', main_rate='25/1', ref_interlaced=False, main_interlaced=False,
               manual_fps=0, vmaf_v0=False, disable_hfr=False, main_stream=None):
    """A vmaf object without ffprobe runs, holding only what the tested methods read"""
    stub = vmaf.__new__(vmaf)
    stub.manual_fps = manual_fps
    stub.vmaf_v0 = vmaf_v0
    stub.disable_hfr = disable_hfr
    stub.ref = SimpleNamespace(streamInfo={'r_frame_rate': ref_rate}, interlaced=ref_interlaced)
    stub.main = SimpleNamespace(streamInfo=main_stream or {'r_frame_rate': main_rate},
                                interlaced=main_interlaced)
    return stub


class TestComparedFrameRate:
    """vmaf._comparedFrameRate follows the frame rate rules of _applyDeinterlaceFilters and _forceFps."""

    @pytest.mark.parametrize(
        "ref_rate, main_rate, ref_interlaced, main_interlaced, manual_fps, expected",
        [
            ('50/1', '25/1', False, False, 0, 25),
            ('25/1', '50/1', False, False, 0, 25),
            ('60000/1001', '60000/1001', False, False, 0, 60000 / 1001),
            ('25/1', '50/1', True, False, 0, 50),
            ('50/1', '25/1', False, True, 0, 50),
            ('50/1', '50/1', True, True, 0, None),
            ('25/1', '25/1', False, False, 50, 50),
        ],
        ids=["progressive-main-lower", "progressive-ref-lower", "progressive-59.94",
             "ref-interlaced", "main-interlaced", "both-interlaced", "fps-flag"],
    )
    def test_rule(self, ref_rate, main_rate, ref_interlaced, main_interlaced, manual_fps, expected):
        result = _vmaf_stub(ref_rate, main_rate, ref_interlaced, main_interlaced, manual_fps)._comparedFrameRate()
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)


class TestUseHfr:
    """vmaf._useHfr: v1 runs at a compared frame rate of 50 or more, unless -disable_hfr."""

    @pytest.mark.parametrize("rate, expected", [('48/1', False), ('50/1', True), ('60000/1001', True), ('30/1', False)],
                             ids=["48", "50", "59.94", "30"])
    def test_threshold(self, rate, expected):
        assert _vmaf_stub(rate, rate)._useHfr() is expected

    def test_25i_reference_against_50p(self):
        assert _vmaf_stub('25/1', '50/1', ref_interlaced=True)._useHfr() is True

    def test_both_interlaced(self):
        assert _vmaf_stub('50/1', '50/1', ref_interlaced=True, main_interlaced=True)._useHfr() is False

    def test_disable_hfr(self):
        assert _vmaf_stub('60/1', '60/1', disable_hfr=True)._useHfr() is False

    @pytest.mark.parametrize("disable_hfr", [False, True], ids=["hfr-allowed", "hfr-disabled"])
    def test_vmaf_v0(self, disable_hfr):
        assert _vmaf_stub('60/1', '60/1', vmaf_v0=True, disable_hfr=disable_hfr)._useHfr() is False


class TestCambiEncodeParams:
    """vmaf._cambiEncodeParams: encode-side size and bit depth of MAIN for the v1 CAMBI feature."""

    def test_with_bit_depth(self):
        stub = _vmaf_stub(main_stream={'width': 1280, 'height': 720, 'bits_per_raw_sample': '10'})
        assert stub._cambiEncodeParams() == {
            'cambi.enc_width': '1280', 'cambi.enc_height': '720', 'cambi.enc_bitdepth': '10'}

    @pytest.mark.parametrize("stream", [
        {'width': 1920, 'height': 1080},
        {'width': 1920, 'height': 1080, 'bits_per_raw_sample': 'N/A'},
    ], ids=["missing", "not-a-number"])
    def test_without_bit_depth(self, stream):
        assert _vmaf_stub(main_stream=stream)._cambiEncodeParams() == {
            'cambi.enc_width': '1920', 'cambi.enc_height': '1080'}


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """check_ffmpeg against canned results: the version line and the exit code of the v1 probe"""
    def install(version_line, v1_returncode=0):
        def run(cmd, capture_output=True, text=True):
            if cmd[1:] == ['-version']:
                return SimpleNamespace(stdout=version_line + '\n', stderr='', returncode=0)
            if PROBE_INPUT in cmd:
                return SimpleNamespace(stdout='', stderr='', returncode=v1_returncode)
            return SimpleNamespace(stdout='', stderr='', returncode=0)
        monkeypatch.setattr(FFmpeg.FFmpegQos, '_executable', 'ffmpeg')
        monkeypatch.setattr(FFmpeg.FFprobe, '_executable', 'ffprobe')
        monkeypatch.setattr(FFmpeg.subprocess, 'run', run)
    return install


class TestCheckFfmpeg:
    """check_ffmpeg: the FFmpeg 9.0 minimum and the VMAF v1 probe."""

    @pytest.mark.parametrize("version_line, version_str, meets_minimum", [
        ("ffmpeg version 8.1 Copyright (c) 2000-2026 the FFmpeg developers", '8.1', False),
        ("ffmpeg version 9.0 Copyright (c) 2000-2026 the FFmpeg developers", '9.0', True),
        ("ffmpeg version 9.0.2 Copyright (c) 2000-2026 the FFmpeg developers", '9.0', True),
        ("ffmpeg version N-121234-gabcdef123 Copyright (c) 2000-2026 the FFmpeg developers", 'dev-build', True),
    ], ids=["8.1", "9.0", "9.0.2", "dev-build"])
    def test_minimum(self, fake_ffmpeg, version_line, version_str, meets_minimum):
        fake_ffmpeg(version_line)
        result = check_ffmpeg()
        assert (result['version_str'], result['meets_minimum']) == (version_str, meets_minimum)

    @pytest.mark.parametrize("returncode, v1_models", [(0, True), (234, False)], ids=["runs", "fails"])
    def test_v1_probe(self, fake_ffmpeg, returncode, v1_models):
        fake_ffmpeg("ffmpeg version 9.0.2 Copyright (c) 2000-2026 the FFmpeg developers", v1_returncode=returncode)
        assert check_ffmpeg()['v1_models'] is v1_models


@needs_v1
class TestReadVmafLog:
    """read_vmaf_log reads every score name of a libvmaf log in json, xml and csv."""

    @pytest.mark.parametrize("log_fmt", ['json', 'xml', 'csv'])
    def test_v1_hd_log(self, tmp_path, log_fmt):
        models = select_models('HD')
        log_path = str(tmp_path / f'log.{log_fmt}')
        assert _run_libvmaf(FFmpegQos._build_model_string(models), log_fmt, log_path, LOG_INPUT) == 0
        log = read_vmaf_log(log_path)
        assert list(log.scores) == ['vmaf_v1_hd', 'vmaf_v1_hd_phone']
        assert len(log.scores['vmaf_v1_hd']) == len(log.scores['vmaf_v1_hd_phone']) == len(log.frame_numbers) > 0
