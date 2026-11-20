"""Tests for the VMAF model sets, the libvmaf model string, the HFR choice, the model
overrides, the startup check and the reading of the VMAF log."""

import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from easyVmafPlus import FFmpeg
from easyVmafPlus.config import ffmpeg as FFMPEG
from easyVmafPlus.FFmpeg import (FFmpegQos, HDR_PROBE_CHAIN, HDR_VMAF_VERSION, ModelConfig, check_ffmpeg,
                                 known_scores, read_vmaf_log, select_models, v1_model_path)
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
    """select_models: model generation, -model value, versions, views, HDR and HFR variants."""

    def test_v1_hd(self):
        models = select_models('HD')
        assert [(m.name, m.label, os.path.basename(m.path), m.version, m.view) for m in models] == [
            ('vmaf_v1_hd', 'VMAF v1 HD', 'vmaf_v1.0.16_3d0h.json', None, '3h'),
            ('vmaf_v1_hd_phone', 'VMAF v1 Phone', 'vmaf_v1.0.16_5d0h.json', None, '5h'),
        ]

    def test_v1_4k(self):
        models = select_models('4K')
        assert [(m.name, m.label, os.path.basename(m.path), m.view) for m in models] == [
            ('vmaf_v1_4k', 'VMAF v1 4K', 'vmaf_v1.0.16_1d5h_2160.json', '1.5h'),
            ('vmaf_v1_4k_3h', 'VMAF v1 4K 3H', 'vmaf_v1.0.16_3d0h_2160.json', '3h'),
        ]

    def test_v1_hfr(self):
        models = select_models('4K', hfr=True)
        assert [(m.name, m.label, os.path.basename(m.path)) for m in models] == [
            ('vmaf_v1_4k_hfr', 'VMAF v1 4K HFR', 'vmaf_v1.0.16_hfr_1d5h_2160.json'),
            ('vmaf_v1_4k_3h_hfr', 'VMAF v1 4K 3H HFR', 'vmaf_v1.0.16_hfr_3d0h_2160.json'),
        ]

    @pytest.mark.parametrize("hfr", [False, True], ids=["standard", "hfr-ignored"])
    def test_v0_hd(self, hfr):
        models = select_models('HD', ('0.6',), hfr=hfr)
        assert [(m.name, m.label, m.version, m.path, m.params, m.view) for m in models] == [
            ('vmaf_hd', 'VMAF HD', 'vmaf_v0.6.1', None, {}, None),
            ('vmaf_hd_neg', 'VMAF Neg', 'vmaf_v0.6.1neg', None, {}, None),
            ('vmaf_hd_phone', 'VMAF Phone', 'vmaf_v0.6.1', None, {'enable_transform': 'true'}, None),
        ]

    def test_v0_4k(self):
        models = select_models('4K', ('0.6',))
        assert [(m.name, m.label, m.version) for m in models] == [('vmaf_4k', 'VMAF 4K', 'vmaf_4k_v0.6.1')]

    @pytest.mark.parametrize("views, expected", [
        (['phone'], ['vmaf_v1_hd_phone']),
        (['5h'], ['vmaf_v1_hd_phone']),
        (['3h'], ['vmaf_v1_hd']),
        (['3H', 'phone'], ['vmaf_v1_hd', 'vmaf_v1_hd_phone']),
    ], ids=["phone", "5h", "3h", "both-upper"])
    def test_views(self, views, expected):
        assert [m.name for m in select_models('HD', views=views)] == expected

    def test_views_4k(self):
        assert [m.name for m in select_models('4K', views=['1.5h'])] == ['vmaf_v1_4k']

    def test_views_ignored_for_v0(self):
        assert [m.name for m in select_models('HD', ('0.6',), views=['phone'])] == ['vmaf_hd', 'vmaf_hd_neg', 'vmaf_hd_phone']

    def test_view_not_available(self):
        with pytest.raises(ValueError, match=r"View '1.5h' is not available for -model HD\. Supported: 3h, 5h"):
            select_models('HD', views=['1.5h'])

    def test_both_versions(self):
        assert [m.name for m in select_models('HD', ('1', '0.6'))] == [
            'vmaf_v1_hd', 'vmaf_v1_hd_phone', 'vmaf_hd', 'vmaf_hd_neg', 'vmaf_hd_phone']

    def test_versions_deduplicated(self):
        assert [m.name for m in select_models('4K', ('0.6', '1', '0.6'))] == ['vmaf_4k', 'vmaf_v1_4k', 'vmaf_v1_4k_3h']

    def test_hdr(self):
        models = select_models('HD', dynamic_range='hdr')
        assert [(m.name, m.label, os.path.basename(m.path), m.view) for m in models] == [
            ('hdr_vmaf_v1_hd', 'HDR-VMAF v1 HD', 'hdr_vmaf_v1.1.8_3d0h.json', '3h'),
            ('hdr_vmaf_v1_hd_phone', 'HDR-VMAF v1 Phone', 'hdr_vmaf_v1.1.8_5d0h.json', '5h'),
        ]
        assert all(os.path.basename(os.path.dirname(m.path)) == 'hdr_vmaf_v1.1.8' for m in models)

    def test_hdr_hfr_4k(self):
        models = select_models('4K', hfr=True, dynamic_range='hdr')
        assert [(m.name, os.path.basename(m.path)) for m in models] == [
            ('hdr_vmaf_v1_4k_hfr', 'hdr_vmaf_v1.1.8_hfr_1d5h_2160.json'),
            ('hdr_vmaf_v1_4k_3h_hfr', 'hdr_vmaf_v1.1.8_hfr_3d0h_2160.json'),
        ]

    @pytest.mark.parametrize("args, message", [
        (('8K',), r"Invalid VMAF model: '8K'\. Supported: HD, 4K"),
        (('8K', ('0.6',)), r"Invalid VMAF model: '8K'\. Supported: HD, 4K"),
        (('HD', ('2',)), r"Invalid VMAF version: '2'\. Supported: 1, 0\.6"),
        (('HD', ()), r"At least one VMAF version is required"),
        (('HD', ('0.6',), None, False, 'hdr'), r"VMAF 0\.6 has no HDR models"),
        (('HD', ('1',), None, False, 'x'), r"Invalid dynamic range: 'x'"),
        (('HD', ('1',), []), r"At least one view is required for -model HD"),
    ], ids=["model-v1", "model-v0", "version", "no-version", "hdr-0.6", "dynamic-range", "no-view"])
    def test_errors(self, args, message):
        with pytest.raises(ValueError, match=message):
            select_models(*args)

    def test_params_are_copies(self):
        select_models('HD', ('0.6',))[2].params['x'] = 'y'
        assert select_models('HD', ('0.6',))[2].params == {'enable_transform': 'true'}

    @pytest.mark.parametrize("hfr", [False, True], ids=["standard", "hfr"])
    @pytest.mark.parametrize("stem", ['vmaf_v1.0.16', HDR_VMAF_VERSION], ids=["v1", "hdr"])
    @pytest.mark.parametrize("variant", V1_VARIANTS)
    def test_bundled_model_file_exists(self, variant, stem, hfr):
        path = v1_model_path(variant, hfr=hfr, stem=stem)
        assert os.path.isfile(path)
        assert os.path.basename(path) == f"{stem}{'_hfr' if hfr else ''}_{variant}.json"


class TestKnownScores:
    """known_scores: every score name with its label and cap, the HDR names included."""

    def test_hdr_names_and_caps(self):
        scores = known_scores()
        hdr = {name: value for name, value in scores.items() if name.startswith('hdr_')}
        assert hdr == {
            'hdr_vmaf_v1_hd': ('HDR-VMAF v1 HD', 100.0),
            'hdr_vmaf_v1_hd_hfr': ('HDR-VMAF v1 HD HFR', 100.0),
            'hdr_vmaf_v1_hd_phone': ('HDR-VMAF v1 Phone', 100.0),
            'hdr_vmaf_v1_hd_phone_hfr': ('HDR-VMAF v1 Phone HFR', 100.0),
            'hdr_vmaf_v1_4k': ('HDR-VMAF v1 4K', 100.0),
            'hdr_vmaf_v1_4k_hfr': ('HDR-VMAF v1 4K HFR', 100.0),
            'hdr_vmaf_v1_4k_3h': ('HDR-VMAF v1 4K 3H', 110.0),
            'hdr_vmaf_v1_4k_3h_hfr': ('HDR-VMAF v1 4K 3H HFR', 110.0),
        }

    def test_order_and_v1(self):
        names = list(known_scores())
        assert names[:4] == ['vmaf_hd', 'vmaf_hd_neg', 'vmaf_hd_phone', 'vmaf_4k']
        assert names[4:8] == ['vmaf_v1_hd', 'vmaf_v1_hd_hfr', 'vmaf_v1_hd_phone', 'vmaf_v1_hd_phone_hfr']
        assert known_scores()['vmaf_v1_4k_3h'] == ('VMAF v1 4K 3H', 110.0)


class TestBuildModelString:
    """FFmpegQos._build_model_string: built-in versions, escaped model file paths and parameters."""

    def test_v0_hd_unchanged(self):
        """The v0 HD model= value of 269b356."""
        expected = ('version=vmaf_v0.6.1' + BS * 2 + ':name=vmaf_hd'
                    + '|version=vmaf_v0.6.1neg' + BS * 2 + ':name=vmaf_hd_neg'
                    + '|version=vmaf_v0.6.1' + BS * 2 + ':name=vmaf_hd_phone' + BS * 2 + ':enable_transform=true')
        assert FFmpegQos._build_model_string(select_models('HD', ('0.6',))) == expected

    def test_v0_4k_unchanged(self):
        """The v0 4K model= value of 269b356."""
        expected = 'version=vmaf_4k_v0.6.1' + BS * 2 + ':name=vmaf_4k'
        assert FFmpegQos._build_model_string(select_models('4K', ('0.6',))) == expected

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


def _vmaf_stub(ref_interlaced=False, main_interlaced=False, manual_fps=0, vmaf_versions=('1',), hfr='auto',
               output_fps=None, main_stream=None, enc_size=None, enc_bitdepth=None, model_options=(),
               model='HD', dynamic_range='sdr', views=None):
    """A vmaf object without ffprobe runs, holding only what the tested methods read"""
    stub = vmaf.__new__(vmaf)
    stub.manual_fps = manual_fps
    stub.vmaf_versions = tuple(vmaf_versions)
    stub.hfr = hfr
    stub.output_fps = output_fps
    stub.model = model
    stub.views = views
    stub.dynamic_range = dynamic_range
    stub.enc_size = enc_size
    stub.enc_bitdepth = enc_bitdepth
    stub.model_options = tuple(model_options)
    stub.ref = SimpleNamespace(streamInfo={'r_frame_rate': '25/1', 'width': 1920, 'height': 1080}, interlaced=ref_interlaced)
    stub.main = SimpleNamespace(streamInfo=main_stream or {'r_frame_rate': '25/1', 'width': 1920, 'height': 1080, 'pix_fmt': 'yuv420p'},
                                interlaced=main_interlaced)
    return stub


class TestUseHfr:
    """vmaf._useHfr: v1 runs at an effective frame rate of 50 or more, -force_hfr and -disable_hfr."""

    @pytest.mark.parametrize("output_fps, expected", [(48, False), (50, True), (60000 / 1001, True), (30, False), (None, False)],
                             ids=["48", "50", "59.94", "30", "unknown"])
    def test_threshold(self, output_fps, expected):
        assert _vmaf_stub(output_fps=output_fps)._useHfr() is expected

    def test_force(self):
        assert _vmaf_stub(output_fps=25, hfr='on')._useHfr() is True

    def test_disable(self):
        assert _vmaf_stub(output_fps=60, hfr='off')._useHfr() is False

    def test_both_interlaced(self):
        assert _vmaf_stub(output_fps=50, ref_interlaced=True, main_interlaced=True)._useHfr() is False

    def test_both_interlaced_with_fps_flag(self):
        assert _vmaf_stub(output_fps=50, manual_fps=50, ref_interlaced=True, main_interlaced=True)._useHfr() is True

    @pytest.mark.parametrize("hfr", ['auto', 'on'], ids=["auto", "forced"])
    def test_vmaf_v0(self, hfr):
        assert _vmaf_stub(output_fps=60, vmaf_versions=('0.6',), hfr=hfr)._useHfr() is False


class TestModelOverrides:
    """vmaf._modelOverrides and _resolveModels: CAMBI encoding size and bit depth, -model_option."""

    def test_from_stream(self):
        stub = _vmaf_stub(main_stream={'width': 1280, 'height': 720, 'pix_fmt': 'yuv420p10le'})
        assert stub._modelOverrides() == {
            'cambi.enc_width': '1280', 'cambi.enc_height': '720', 'cambi.enc_bitdepth': '10'}

    @pytest.mark.parametrize("stream, bitdepth", [
        ({'width': 1920, 'height': 1080, 'pix_fmt': 'yuv420p'}, '8'),
        ({'width': 1920, 'height': 1080}, '8'),
        ({'width': 1920, 'height': 1080, 'pix_fmt': 'gbrp12le'}, '12'),
    ], ids=["8-bit", "missing", "12-bit"])
    def test_bit_depth_from_pix_fmt(self, stream, bitdepth):
        assert _vmaf_stub(main_stream=stream)._modelOverrides()['cambi.enc_bitdepth'] == bitdepth

    def test_flags(self):
        stub = _vmaf_stub(enc_size=(1280, 720), enc_bitdepth=10, model_options=('cambi.topk=0.5',))
        assert stub._modelOverrides() == {
            'cambi.enc_width': '1280', 'cambi.enc_height': '720', 'cambi.enc_bitdepth': '10', 'cambi.topk': '0.5'}

    def test_small_encode(self, caplog):
        stub = _vmaf_stub(main_stream={'width': 256, 'height': 144, 'pix_fmt': 'yuv420p'})
        with caplog.at_level('WARNING'):
            overrides = stub._modelOverrides()
        assert (overrides['cambi.enc_width'], overrides['cambi.enc_height']) == ('267', '150')
        assert "CAMBI does not accept a 256x144 encoding size; using 267x150" in caplog.text

    def test_resolve_models(self):
        stub = _vmaf_stub(output_fps=25, main_stream={'width': 1280, 'height': 720, 'pix_fmt': 'yuv420p'})
        models = stub._resolveModels()
        assert [m.name for m in models] == ['vmaf_v1_hd', 'vmaf_v1_hd_phone']
        assert all(m.params == {'cambi.enc_width': '1280', 'cambi.enc_height': '720', 'cambi.enc_bitdepth': '8'} for m in models)
        assert stub.hfr_active is False

    def test_resolve_models_both_versions(self):
        stub = _vmaf_stub(output_fps=50, vmaf_versions=('1', '0.6'))
        models = stub._resolveModels()
        assert [m.name for m in models] == ['vmaf_v1_hd_hfr', 'vmaf_v1_hd_phone_hfr', 'vmaf_hd', 'vmaf_hd_neg', 'vmaf_hd_phone']
        assert 'cambi.enc_width' in models[0].params and 'cambi.enc_width' not in models[2].params
        assert stub.hfr_active is True

    def test_resolve_models_hdr(self):
        stub = _vmaf_stub(output_fps=24, dynamic_range='hdr', main_stream={'width': 1920, 'height': 1080, 'pix_fmt': 'yuv420p10le'})
        models = stub._resolveModels()
        assert [m.name for m in models] == ['hdr_vmaf_v1_hd', 'hdr_vmaf_v1_hd_phone']
        assert models[0].params['cambi.enc_bitdepth'] == '10'

    def test_above_calibration_warning(self, caplog):
        stub = _vmaf_stub(output_fps=120)
        with caplog.at_level('WARNING'):
            stub._resolveModels()
        assert "120 fps is above the ~50/60 fps the HFR VMAF v1 models are calibrated for" in caplog.text


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """check_ffmpeg against canned results: the version line and the exit codes of the v1 and HDR probes"""
    commands = []

    def install(version_line, v1_returncode=0, hdr_returncode=0):
        def run(cmd, capture_output=True, text=True):
            commands.append(cmd)
            if cmd[1:] == ['-version']:
                return SimpleNamespace(stdout=version_line + '\n', stderr='', returncode=0)
            if PROBE_INPUT in cmd:
                hdr = HDR_VMAF_VERSION in cmd[cmd.index('-lavfi') + 1]
                return SimpleNamespace(stdout='', stderr='', returncode=hdr_returncode if hdr else v1_returncode)
            return SimpleNamespace(stdout='', stderr='', returncode=0)
        monkeypatch.setattr(FFmpeg.FFmpegQos, '_executable', 'ffmpeg')
        monkeypatch.setattr(FFmpeg.FFprobe, '_executable', 'ffprobe')
        monkeypatch.setattr(FFmpeg.subprocess, 'run', run)
        return commands
    return install


class TestCheckFfmpeg:
    """check_ffmpeg: the FFmpeg 9.0 minimum, the VMAF v1 probe and the HDR-VMAF probe."""

    @pytest.mark.parametrize("version_line, version_str, meets_minimum", [
        ("ffmpeg version 8.1 Copyright (c) 2000-2026 the FFmpeg developers", '8.1', False),
        ("ffmpeg version 9.0 Copyright (c) 2000-2026 the FFmpeg developers", '9.0', True),
        ("ffmpeg version 9.0.2 Copyright (c) 2000-2026 the FFmpeg developers", '9.0', True),
        ("ffmpeg version N-121234-gabcdef123 Copyright (c) 2000-2026 the FFmpeg developers", 'dev-build', True),
        ("ffmpeg version git-88c46fce27e3 Copyright (c) 2000-2026 the FFmpeg developers", 'dev-build', True),
    ], ids=["8.1", "9.0", "9.0.2", "dev-build", "git-commit"])
    def test_minimum(self, fake_ffmpeg, version_line, version_str, meets_minimum):
        fake_ffmpeg(version_line)
        result = check_ffmpeg()
        assert (result['version_str'], result['meets_minimum']) == (version_str, meets_minimum)

    @pytest.mark.parametrize("returncode, v1_models", [(0, True), (234, False)], ids=["runs", "fails"])
    def test_v1_probe(self, fake_ffmpeg, returncode, v1_models):
        fake_ffmpeg("ffmpeg version 9.0.2 Copyright (c) 2000-2026 the FFmpeg developers", v1_returncode=returncode)
        assert check_ffmpeg()['v1_models'] is v1_models

    @pytest.mark.parametrize("returncode, hdr_vmaf", [(0, True), (1, False)], ids=["runs", "fails"])
    def test_hdr_probe(self, fake_ffmpeg, returncode, hdr_vmaf):
        commands = fake_ffmpeg("ffmpeg version 9.0.2 Copyright (c) 2000-2026 the FFmpeg developers", hdr_returncode=returncode)
        assert check_ffmpeg()['hdr_vmaf'] is hdr_vmaf
        graphs = [cmd[cmd.index('-lavfi') + 1] for cmd in commands if PROBE_INPUT in cmd]
        hdr_graph = [graph for graph in graphs if HDR_VMAF_VERSION in graph]
        assert len(hdr_graph) == 1
        assert hdr_graph[0].startswith(f'[0:v]{HDR_PROBE_CHAIN}[dis];[1:v]{HDR_PROBE_CHAIN}[ref];[dis][ref]libvmaf=model=path=')
        assert 'hdr_vmaf_v1.1.8_3d0h.json' in hdr_graph[0]
        assert HDR_PROBE_CHAIN == ('setparams=range=tv:color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc,'
                                   'format=yuv420p10le')


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
