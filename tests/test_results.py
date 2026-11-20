"""Tests for results.py: reading, trimming and joining libvmaf logs of frame ranges in json, xml and csv,
and trimming and joining CAMBI heatmaps. tests/data/vmaf_v1_hd.* are logs of libvmaf 3.2.1 over five frames."""

import csv
import json
import os
import shutil
import xml.etree.ElementTree as ET

import pytest

from easyVmafPlus.FFmpeg import read_vmaf_log
from easyVmafPlus.results import (heatmap_files, join_heatmaps, join_logs, read_frames, trim_heatmaps, trim_log,
                                  _joined_fps, _pool)

DATA = os.path.join(os.path.dirname(__file__), 'data')
FORMATS = ['json', 'xml', 'csv']
V1_HD = [53.299798, 54.369612, 52.443107, 52.641412, 53.13457]


def _copy(tmp_path, log_fmt, name='log'):
    path = str(tmp_path / f'{name}.{log_fmt}')
    shutil.copy(os.path.join(DATA, f'vmaf_v1_hd.{log_fmt}'), path)
    return path


def _pooled(path, log_fmt, name='vmaf_v1_hd'):
    """The pooled metrics of a json or xml log as floats"""
    if log_fmt == 'json':
        with open(path) as log_file:
            return {k: float(v) for k, v in json.load(log_file)['pooled_metrics'][name].items()}
    metric = [m for m in ET.parse(path).getroot().findall('pooled_metrics/metric') if m.get('name') == name][0]
    return {k: float(v) for k, v in metric.attrib.items() if k != 'name'}


@pytest.mark.parametrize("log_fmt", FORMATS)
def test_read_frames(tmp_path, log_fmt):
    frames = read_frames(_copy(tmp_path, log_fmt), log_fmt)
    assert [n for n, _ in frames] == [0, 1, 2, 3, 4]
    assert [m['vmaf_v1_hd'] for _, m in frames] == pytest.approx(V1_HD)
    assert all('Frame' not in m and 'frameNum' not in m for _, m in frames)


def test_pool():
    assert _pool([1.0, 3.0]) == {'min': 1.0, 'max': 3.0, 'mean': 2.0, 'harmonic_mean': pytest.approx(2 / (1 / 2 + 1 / 4) - 1)}


@pytest.mark.parametrize("log_fmt", FORMATS)
def test_trim_log(tmp_path, log_fmt):
    path = _copy(tmp_path, log_fmt)
    trim_log(path, log_fmt, 1, 3, 100)
    frames = read_frames(path, log_fmt)
    assert [n for n, _ in frames] == [101, 102, 103]
    assert [m['vmaf_v1_hd'] for _, m in frames] == pytest.approx(V1_HD[1:4])
    if log_fmt != 'csv':
        pooled = _pooled(path, log_fmt)
        assert pooled['min'] == pytest.approx(min(V1_HD[1:4]), abs=1e-6)
        assert pooled['max'] == pytest.approx(max(V1_HD[1:4]), abs=1e-6)
        assert pooled['mean'] == pytest.approx(sum(V1_HD[1:4]) / 3, abs=1e-6)
        assert pooled['harmonic_mean'] == pytest.approx(3 / sum(1 / (v + 1) for v in V1_HD[1:4]) - 1, abs=1e-6)
    log = read_vmaf_log(path)
    assert (log.frame_numbers, log.means['vmaf_v1_hd']) == ([101, 102, 103], pytest.approx(sum(V1_HD[1:4]) / 3, abs=1e-6))


@pytest.mark.parametrize("log_fmt", FORMATS)
def test_join_logs(tmp_path, log_fmt):
    parts = []
    for index, (start, count) in enumerate([(0, 2), (2, 2), (4, 1)]):
        path = _copy(tmp_path, log_fmt, f'part{index}')
        trim_log(path, log_fmt, start, count, 0)
        parts.append(path)
    out = str(tmp_path / f'joined.{log_fmt}')
    join_logs(parts, log_fmt, out)
    frames = read_frames(out, log_fmt)
    assert [n for n, _ in frames] == [0, 1, 2, 3, 4]
    assert [m['vmaf_v1_hd'] for _, m in frames] == pytest.approx(V1_HD)
    full = read_vmaf_log(os.path.join(DATA, f'vmaf_v1_hd.{log_fmt}'))
    joined = read_vmaf_log(out)
    assert joined.means == {name: pytest.approx(value, abs=1e-6) for name, value in full.means.items()}
    assert joined.harmonic_means == {name: pytest.approx(value, abs=1e-6) for name, value in full.harmonic_means.items()}
    if log_fmt == 'json':
        with open(out) as log_file:
            assert json.load(log_file)['fps'] == pytest.approx(37.19, abs=0.01)
    if log_fmt == 'xml':
        assert ET.parse(out).getroot().find('fyi').get('fps') is not None
    if log_fmt == 'csv':
        with open(out, newline='') as csv_file:
            rows = list(csv.reader(csv_file))
        assert rows[0][0] == 'Frame' and len(rows) == 6


def test_joined_fps():
    assert _joined_fps([2, 2], [10.0, 10.0]) == pytest.approx(10.0)
    assert _joined_fps([2, 2], [10.0, 20.0]) == pytest.approx(4 / (0.2 + 0.1))
    assert _joined_fps([2, 2], [10.0, None]) is None


def _heatmap_dir(tmp_path, name, pictures, picture=4):
    folder = tmp_path / name
    folder.mkdir()
    for scale in (0, 1):
        (folder / f'cambi_heatmap_scale_{scale}.gray').write_bytes(bytes((scale * 100 + i) % 256 for i in range(pictures * picture)))
    (folder / 'other.txt').write_text('x')
    return str(folder)


def test_heatmap_files(tmp_path):
    folder = _heatmap_dir(tmp_path, 'heat', 3)
    assert [os.path.basename(p) for p in heatmap_files(folder)] == ['cambi_heatmap_scale_0.gray', 'cambi_heatmap_scale_1.gray']


def test_trim_heatmaps(tmp_path):
    folder = _heatmap_dir(tmp_path, 'heat', 10)
    trim_heatmaps(folder, 2, 3, 10, block=5)
    for scale in (0, 1):
        data = open(os.path.join(folder, f'cambi_heatmap_scale_{scale}.gray'), 'rb').read()
        assert data == bytes((scale * 100 + i) % 256 for i in range(8, 20))


def test_trim_heatmaps_size_mismatch(tmp_path):
    folder = _heatmap_dir(tmp_path, 'heat', 10)
    with pytest.raises(ValueError, match=r"does not hold 7 pictures, one per measured frame"):
        trim_heatmaps(folder, 0, 1, 7)


def test_join_heatmaps(tmp_path):
    a = _heatmap_dir(tmp_path, 'a', 2)
    b = _heatmap_dir(tmp_path, 'b', 3)
    out = tmp_path / 'joined'
    out.mkdir()
    join_heatmaps([a, b], str(out))
    for scale in (0, 1):
        data = open(out / f'cambi_heatmap_scale_{scale}.gray', 'rb').read()
        assert data == bytes((scale * 100 + i) % 256 for i in range(8)) + bytes((scale * 100 + i) % 256 for i in range(12))
    assert sorted(os.listdir(out)) == ['cambi_heatmap_scale_0.gray', 'cambi_heatmap_scale_1.gray']


def test_join_heatmaps_mismatch(tmp_path):
    a = _heatmap_dir(tmp_path, 'a', 2)
    b = tmp_path / 'b'
    b.mkdir()
    (b / 'cambi_heatmap_scale_0.gray').write_bytes(b'xxxx')
    with pytest.raises(ValueError, match=r"CAMBI heatmaps of .* differ from those of"):
        join_heatmaps([a, str(b)], str(tmp_path / 'out'))
