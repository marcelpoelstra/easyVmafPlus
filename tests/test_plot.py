"""Tests for plot.py: the temporal, combined and percentile plots, the selection and export of the
low quality frames and the easyVmafPlusPlot command, with the frames pass replaced by a fake. One
test runs the frames pass with ffmpeg. tests/data/vmaf_v1_hd.* and tests/data/vmaf_v0_hd.json are
logs of libvmaf 3.2.1 over five frames."""

import logging
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from easyVmafPlus import plot
from easyVmafPlus.config import ffmpeg as FFMPEG
from easyVmafPlus.config import ffprobe as FFPROBE
from easyVmafPlus.FFmpeg import VmafLog, read_vmaf_log
from easyVmafPlus.Vmaf import vmaf

DATA = os.path.join(os.path.dirname(__file__), 'data')
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def make_log(path, scores, frame_numbers=None):
    """A VmafLog with the given scores per name; means and harmonic means as libvmaf pools them"""
    count = len(next(iter(scores.values())))
    return VmafLog(path, frame_numbers or list(range(count)), scores,
                   {name: sum(values) / len(values) for name, values in scores.items()},
                   {name: len(values) / sum(1 / (x + 1) for x in values) - 1 for name, values in scores.items()})


def data_log(name):
    return read_vmaf_log(os.path.join(DATA, name))


class TestLowQualityFrames:
    """low_quality_frames: main model scores below the 1st percentile, rounded to 3 decimals."""

    def test_below_first_percentile(self):
        # scores 0 to 99: the 1st percentile is 0.99, so frame 0 lies below it
        log = make_log('a_vmaf.json', {'vmaf_v1_hd': [float(n) for n in range(100)],
                                       'vmaf_v1_hd_phone': [50.0] * 100})
        assert plot.low_quality_frames(log) == [0]

    def test_main_model_and_frame_numbers(self):
        log = make_log('a_vmaf.json', {'vmaf_v1_hd': [90.0, 10.0] + [90.0] * 98,
                                       'vmaf_v1_hd_phone': [90.0] * 99 + [10.0]},
                       frame_numbers=list(range(0, 300, 3)))
        assert plot.low_quality_frames(log) == [3]

    def test_equal_scores(self):
        assert plot.low_quality_frames(make_log('a_vmaf.json', {'vmaf_v1_hd': [80.0] * 10})) == []


class TestPlots:
    """plot_vmaf, plot_multi_vmaf and plot_percentile_vmaf: curves, legends, sizes and the y-axis."""

    def test_temporal_plot(self):
        fig = plot.plot_vmaf(data_log('vmaf_v1_hd.json'))
        ax = fig.axes[0]
        assert ax.get_ylim() == (0.0, 100.0)
        assert [text.get_text().splitlines()[0] for text in ax.get_legend().get_texts()] == [
            'File: vmaf_v1_hd.json, VMAF v1 HD', 'File: vmaf_v1_hd.json, VMAF v1 Phone']
        # 3 + round(4 * log10(5 frames)) inches wide
        assert fig.get_size_inches().tolist() == [6.0, 5.0]

    def test_combined_temporal_plot(self):
        ax = plot.plot_multi_vmaf([data_log('vmaf_v1_hd.csv'), data_log('vmaf_v1_hd.xml')]).axes[0]
        # a curve and a mean line per log and VMAF score
        assert len(ax.get_lines()) == 8
        # each mean line in the colour of its curve
        curves, mean_lines = ax.get_lines()[0::2], ax.get_lines()[1::2]
        assert [line.get_color() for line in mean_lines] == [line.get_color() for line in curves]
        assert ax.get_ylim() == (0.0, 100.0)

    def test_percentile_plot(self):
        fig = plot.plot_percentile_vmaf([data_log('vmaf_v1_hd.json'), data_log('vmaf_v0_hd.json')])
        ax = fig.axes[0]
        assert list(ax.get_xticks()) == [1, 5, 25, 50, 75, 99]
        assert len(ax.get_lines()) == 5
        # 5 + 0.5 * 5 curves inches high
        assert fig.get_size_inches().tolist() == [6.0, 7.5]

    def test_cap_110(self):
        log = make_log('a_vmaf.json', {'vmaf_v1_4k': [95.0, 96.0], 'vmaf_v1_4k_3h': [104.0, 108.0]})
        for fig in (plot.plot_vmaf(log), plot.plot_multi_vmaf([log, log]), plot.plot_percentile_vmaf([log])):
            assert fig.axes[0].get_ylim() == (0.0, 110.0)


class TestSaveFigure:
    """save_figure: the format of the extension, .png without one, and unique names."""

    def test_png_and_unique_names(self, tmp_path):
        fig = plot.plot_vmaf(data_log('vmaf_v1_hd.json'))
        first = plot.save_figure(fig, str(tmp_path / 'a_vmaf_plot.png'))
        second = plot.save_figure(fig, str(tmp_path / 'a_vmaf_plot.png'))
        assert (first, second) == (str(tmp_path / 'a_vmaf_plot.png'), str(tmp_path / 'a_vmaf_plot_2.png'))
        with open(first, 'rb') as image_file:
            assert image_file.read(8) == PNG_SIGNATURE

    def test_format_from_extension(self, tmp_path):
        fig = plot.plot_vmaf(data_log('vmaf_v1_hd.json'))
        with open(plot.save_figure(fig, str(tmp_path / 'plot.svg')), 'rb') as image_file:
            assert image_file.read(5) == b'<?xml'
        assert plot.save_figure(fig, str(tmp_path / 'plot')) == str(tmp_path / 'plot.png')

    def test_unknown_format_leaves_nothing(self, tmp_path):
        fig = plot.plot_vmaf(data_log('vmaf_v1_hd.json'))
        with pytest.raises(ValueError):
            plot.save_figure(fig, str(tmp_path / 'plot.txt'))
        assert os.listdir(tmp_path) == []


def test_timestamp():
    assert [plot._timestamp(s) for s in (0, 7.966667, 3725.5)] == ['00:00:00.000', '00:00:07.967', '01:02:05.500']


@pytest.fixture
def frames_vmaf(tmp_path):
    """A run whose getFrames writes 640x480 red TIFF files for the first `written` requested frames
    (all by default) and returns them with 1.0 + frame number / 30 as their times"""
    def make(written=None):
        calls = []

        def getFrames(frame_numbers, folder):
            calls.append(list(frame_numbers))
            numbers = sorted(frame_numbers)[:written]
            result = []
            for number in numbers:
                path = os.path.join(folder, f'{number}.tif')
                Image.new('RGB', (640, 480), (255, 0, 0)).save(path, format='TIFF')
                result.append((number, path, 1.0 + number / 30))
            return result
        return SimpleNamespace(getFrames=getFrames, main=SimpleNamespace(videoSrc=str(tmp_path / 'dist.mp4')),
                               calls=calls)
    return make


class TestExportTiffFrames:
    """export_tiff_frames: the TIFF files, their names and overlay, the folder and the frames beyond the end."""

    def test_export(self, tmp_path, frames_vmaf):
        shutil.copy(os.path.join(DATA, 'vmaf_v1_hd.json'), tmp_path / 'dist_vmaf.json')
        run = frames_vmaf()
        folder = plot.export_tiff_frames(run, read_vmaf_log(str(tmp_path / 'dist_vmaf.json')))
        assert folder == str(tmp_path / 'dist_vmaf_lowframes')
        # vmaf_v1_hd: 53.299798, 54.369612, 52.443107, 52.641412, 53.13457; below the 1st percentile: frame 2
        assert run.calls == [[2]]
        assert os.listdir(folder) == ['dist_vmaf_VMAF52_frame002.tif']
        with Image.open(os.path.join(folder, 'dist_vmaf_VMAF52_frame002.tif')) as image:
            assert (image.size, image.info.get('compression')) == ((640, 480), 'tiff_lzw')
            assert (0, 0, 0) in [colour for _, colour in image.crop((10, 10, 120, 60)).getcolors(maxcolors=100000)]
            assert image.getpixel((630, 470)) == (255, 0, 0)

    def test_no_low_frames(self, tmp_path, frames_vmaf, caplog):
        run = frames_vmaf()
        with caplog.at_level(logging.INFO, logger='easyVmafPlus.plot'):
            assert plot.export_tiff_frames(run, make_log(str(tmp_path / 'dist_vmaf.json'),
                                                         {'vmaf_v1_hd': [80.0] * 10})) is None
        assert (run.calls, os.listdir(tmp_path)) == ([], [])
        assert 'No frames below the 1st percentile of VMAF v1 HD in' in caplog.text

    def test_frames_beyond_the_end(self, tmp_path, frames_vmaf, caplog):
        # frames 0 and 1 lie below the 1st percentile; the distorted input yields only the first
        log = make_log(str(tmp_path / 'dist_vmaf.json'), {'vmaf_v1_hd': [10.0, 10.0] + [90.0] * 198})
        with caplog.at_level(logging.WARNING, logger='easyVmafPlus.plot'):
            folder = plot.export_tiff_frames(frames_vmaf(written=1), log)
        assert os.listdir(folder) == ['dist_vmaf_VMAF10_frame000.tif']
        assert 'Frames beyond the end of the distorted input, not exported: 1' in caplog.text

    def test_unique_folder(self, tmp_path, frames_vmaf):
        (tmp_path / 'dist_vmaf_lowframes').mkdir()
        log = make_log(str(tmp_path / 'dist_vmaf.json'), {'vmaf_v1_hd': [10.0] + [90.0] * 99})
        assert plot.export_tiff_frames(frames_vmaf(), log) == str(tmp_path / 'dist_vmaf_lowframes_2')
        assert os.listdir(tmp_path / 'dist_vmaf_lowframes') == []

    def test_failure_removes_empty_folder(self, tmp_path):
        def getFrames(frame_numbers, pattern):
            raise subprocess.CalledProcessError(1, ['ffmpeg'])
        run = SimpleNamespace(getFrames=getFrames, main=SimpleNamespace(videoSrc='dist.mp4'))
        with pytest.raises(subprocess.CalledProcessError):
            plot.export_tiff_frames(run, make_log(str(tmp_path / 'dist_vmaf.json'), {'vmaf_v1_hd': [10.0] + [90.0] * 99}))
        assert os.listdir(tmp_path) == []

    def test_nothing_written_removes_folder(self, tmp_path, frames_vmaf):
        log = make_log(str(tmp_path / 'dist_vmaf.json'), {'vmaf_v1_hd': [10.0] + [90.0] * 99})
        assert plot.export_tiff_frames(frames_vmaf(written=0), log) is None
        assert os.listdir(tmp_path) == []

    def test_overlay_text(self, tmp_path, frames_vmaf, monkeypatch):
        drawn = []
        monkeypatch.setattr(plot, '_draw_overlay', lambda image, lines: drawn.append(lines))
        shutil.copy(os.path.join(DATA, 'vmaf_v1_hd.json'), tmp_path / 'dist_vmaf.json')
        plot.export_tiff_frames(frames_vmaf(), read_vmaf_log(str(tmp_path / 'dist_vmaf.json')))
        # frame 2 scores 52.443107; the fake writes it at 1.0 + 2 / 30 s
        assert drawn == [['dist.mp4', 'VMAF v1 HD: 52.443', 'Frame: 2', 'Time: 00:00:01.067']]

    def test_failed_save_removes_file(self, tmp_path, monkeypatch):
        source = tmp_path / 'source.tif'
        Image.new('RGB', (64, 48), (255, 0, 0)).save(source, format='TIFF')

        def getFrames(frame_numbers, folder):
            shutil.copy(source, os.path.join(folder, '0.tif'))
            return [(0, os.path.join(folder, '0.tif'), 1.0)]

        def save(image, fp, **kwargs):
            # a partly written file, then a full disk
            fp.write(b'II*\x00')
            raise OSError(28, 'No space left on device')
        monkeypatch.setattr(Image.Image, 'save', save)
        run = SimpleNamespace(getFrames=getFrames, main=SimpleNamespace(videoSrc='dist.mp4'))
        with pytest.raises(OSError):
            plot.export_tiff_frames(run, make_log(str(tmp_path / 'dist_vmaf.json'), {'vmaf_v1_hd': [10.0] + [90.0] * 99}))
        assert os.listdir(tmp_path) == ['source.tif']


@pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="ffmpeg not found on PATH")
def test_frames_pass_with_ffmpeg(tmp_path):
    """Frames of a distorted input trimmed at 1.0 s: frame n of the frames chain is raw frame n + 10"""
    dist, ref = str(tmp_path / 'dist.mkv'), str(tmp_path / 'ref.mkv')
    for path in (dist, ref):
        subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=160x120:r=10:d=3',
                        '-c:v', 'ffv1', path], check=True)
    run = vmaf(dist, ref, output_fmt='json')
    run.offset = -1.0
    # a % in the folder name, as in URL-encoded file names
    folder = tmp_path / 'My%20Video'
    folder.mkdir()
    # frame 25 lies beyond the 20 frames of the trimmed chain
    written = run.getFrames([5, 7, 25], str(folder))
    assert [(number, seconds) for number, _, seconds in written] == [(5, pytest.approx(1.5)), (7, pytest.approx(1.7))]
    for (number, path, _), raw in zip(written, (15, 17)):
        expected = str(tmp_path / f'raw{raw}.tif')
        subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-i', dist, '-vf', f'select=eq(n\\,{raw})',
                        '-fps_mode', 'passthrough', '-frames:v', '1', '-pix_fmt', 'rgb24', expected], check=True)
        with Image.open(path) as written_frame, Image.open(expected) as raw_frame:
            assert written_frame.tobytes() == raw_frame.tobytes()


@pytest.fixture
def plot_main(monkeypatch, capsys):
    """plot.main with the given arguments; returns the exit code and the output"""
    def run(*argv):
        monkeypatch.setattr(sys, 'argv', ['easyVmafPlusPlot', *argv])
        monkeypatch.setattr(plot.logging, 'basicConfig', lambda **kwargs: None)
        try:
            plot.main()
            code = None
        except SystemExit as exit_info:
            code = exit_info.code
        out, err = capsys.readouterr()
        return SimpleNamespace(code=code, out=out, err=err)
    return run


class TestMain:
    """easyVmafPlusPlot: per-log and combined outputs, unique names and errors."""

    def test_one_log(self, tmp_path, plot_main):
        shutil.copy(os.path.join(DATA, 'vmaf_v1_hd.json'), tmp_path / 'dist_vmaf.json')
        result = plot_main(str(tmp_path / 'dist_vmaf.json'))
        assert result.code is None
        assert sorted(os.listdir(tmp_path)) == ['dist_vmaf.json', 'dist_vmaf_histo.png', 'dist_vmaf_plot.png']
        assert result.out.splitlines() == [f"Plot output path:  {tmp_path / 'dist_vmaf_plot.png'}",
                                           f"Percentile plot output path:  {tmp_path / 'dist_vmaf_histo.png'}"]

    def test_combined(self, tmp_path, plot_main):
        shutil.copy(os.path.join(DATA, 'vmaf_v1_hd.json'), tmp_path / 'a_vmaf.json')
        shutil.copy(os.path.join(DATA, 'vmaf_v1_hd.csv'), tmp_path / 'b_vmaf.csv')
        result = plot_main(str(tmp_path / 'a_vmaf.json'), str(tmp_path / 'b_vmaf.csv'), '-o', str(tmp_path / 'ladder.svg'))
        assert result.code is None
        assert sorted(os.listdir(tmp_path)) == ['a_vmaf.json', 'a_vmaf_histo.png', 'a_vmaf_plot.png', 'b_vmaf.csv',
                                                'b_vmaf_histo.png', 'b_vmaf_plot.png', 'ladder.svg', 'ladder_histo.svg']

    def test_existing_outputs_are_kept(self, tmp_path, plot_main):
        shutil.copy(os.path.join(DATA, 'vmaf_v1_hd.json'), tmp_path / 'dist_vmaf.json')
        (tmp_path / 'dist_vmaf_plot.png').write_bytes(b'keep')
        plot_main(str(tmp_path / 'dist_vmaf.json'))
        assert (tmp_path / 'dist_vmaf_plot.png').read_bytes() == b'keep'
        assert (tmp_path / 'dist_vmaf_plot_2.png').exists()

    @pytest.mark.parametrize("name, content, message", [
        ('a_vmaf.txt', '', "unknown format 'txt'"),
        ('a_vmaf.json', '{"frames": []}', 'holds no frames'),
    ], ids=["unknown-format", "no-frames"])
    def test_errors(self, tmp_path, plot_main, name, content, message):
        (tmp_path / name).write_text(content)
        result = plot_main(str(tmp_path / name))
        assert result.code == 1
        assert result.err.startswith('[easyVmafPlus] ERROR: ')
        assert message in result.err

    def test_help(self, plot_main):
        result = plot_main('-h')
        assert result.code == 0
        assert result.out.startswith('usage: easyVmafPlusPlot')
