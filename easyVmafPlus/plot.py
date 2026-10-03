"""
VMAF plots and the export of the low quality frames, with the features of Plot_Vmaf_improved:
the temporal plot, the combined temporal plot, the percentile plot, and the frames below the
1st percentile as TIFF files with a text overlay. main() is the easyVmafPlusPlot command.
"""

import argparse
import logging
import math
import os
import sys
import tempfile

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

from .cli import _exit_with_error
from .FFmpeg import create_unique_dir, create_unique_file, known_scores, read_vmaf_log

logger = logging.getLogger(__name__)

PERCENTILES = [1, 5, 25, 50, 75, 99]
FONT = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans-Bold.ttf')


def _percentile(values, percentile, decimals):
    """numpy.percentile with its default method, rounded"""
    return round(float(np.percentile(values, percentile)), decimals)


def _set_y_axis(ax, names):
    """y-axis from 0 to the highest score cap of the given VMAF scores, a tick every 5; returns the cap"""
    scores = known_scores()
    cap = max(scores[name][1] for name in names)
    ax.set_ylim(0, cap)
    ax.set_yticks(range(0, int(cap) + 1, 5))
    return cap


def plot_vmaf(log):
    """
    Temporal plot of one VMAF log: a curve per VMAF score, with the 1%, 25%, 75% and 99% lines and
    the harmonic mean line of the main model.
    """
    labels = known_scores()
    frames = len(log.frame_numbers)
    first, last = log.frame_numbers[0], log.frame_numbers[-1]
    fig = Figure(figsize=(3 + round(4 * math.log10(frames)), 5))
    ax = fig.subplots()
    cap = _set_y_axis(ax, log.scores)
    for level in range(0, int(cap) + 1, 5):
        ax.axhline(level, color='grey', linewidth=0.4)
    for level in range(0, int(cap) + 1, 10):
        ax.axhline(level, color='black', linewidth=0.6)
    for name, values in log.scores.items():
        p1, p25, p75, p99 = (_percentile(values, percentile, 3) for percentile in (1, 25, 75, 99))
        ax.plot(log.frame_numbers, values, linewidth=0.7,
                label=f'File: {os.path.basename(log.path)}, {labels[name][0]}\n'
                      f'Frames: {frames} Harmonic mean:{round(log.harmonic_means[name], 2)}\n'
                      f'1%: {p1}  25%: {p25}  75%: {p75} 99%: {p99}')
    main = next(iter(log.scores))
    for percentile, style, colour in ((1, '-', 'red'), (25, ':', 'orange'), (75, ':', 'blue'), (99, ':', 'green')):
        level = _percentile(log.scores[main], percentile, 3)
        ax.plot([first, last], [level, level], style, color=colour)
        ax.annotate(f'{percentile}%: {level}', xy=(first, level), color=colour)
    hmean = round(log.harmonic_means[main], 2)
    ax.plot([first, last], [hmean, hmean], ':', color='black')
    ax.annotate(f'Harm. mean: {hmean}', xy=(first, hmean), color='black')
    ax.set_ylabel('VMAF')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True)
    fig.tight_layout()
    ax.margins(0)
    return fig


def plot_multi_vmaf(logs):
    """Combined temporal plot: a curve per VMAF log and VMAF score, each with a dotted mean line"""
    labels = known_scores()
    fig = Figure()
    ax = fig.subplots()
    for log in logs:
        first, last = log.frame_numbers[0], log.frame_numbers[-1]
        for name, values in log.scores.items():
            p1, p25, p50, p75, p99 = (_percentile(values, percentile, 3) for percentile in (1, 25, 50, 75, 99))
            amean, hmean = round(log.means[name], 2), round(log.harmonic_means[name], 2)
            curve, = ax.plot(log.frame_numbers, values, linewidth=0.7,
                             label=f'File: {os.path.basename(log.path)}, {labels[name][0]}\n'
                                   f'Frames: {len(values)} Mean:{amean} - Harmonic Mean:{hmean}\n'
                                   f'1%: {p1}  25%: {p25}  50%: {p50}, 75%: {p75}, 99%: {p99}')
            ax.plot([first, last], [amean, amean], ':', color=curve.get_color())
            ax.annotate(f'Mean: {amean}', xy=(first, amean))
    _set_y_axis(ax, [name for log in logs for name in log.scores])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, fontsize='x-small')
    fig.tight_layout()
    ax.margins(0)
    return fig


def plot_percentile_vmaf(logs):
    """
    Percentile plot: the VMAF scores at the percentiles 1, 5, 25, 50, 75 and 99, a curve per VMAF
    log and VMAF score.
    """
    labels = known_scores()
    curves = [(log, name) for log in logs for name in log.scores]
    fig = Figure(figsize=(6, 5 + len(curves) * 0.5))
    ax = fig.subplots()
    for log, name in curves:
        values = [_percentile(log.scores[name], percentile, 2) for percentile in PERCENTILES]
        p1, p5, p25, p50, p75, p99 = values
        amean, hmean = round(log.means[name], 2), round(log.harmonic_means[name], 2)
        ax.plot(PERCENTILES, values, '-*', linewidth=0.7,
                label=f'File: {os.path.basename(log.path)}, {labels[name][0]}\n'
                      f'Mean: {amean} - HMean:{hmean}\n'
                      f'1%: {p1} 5%: {p5} 25%: {p25}  50%: {p50} 75%: {p75} 99%: {p99}')
    ax.set_xticks(PERCENTILES)
    _set_y_axis(ax, [name for _, name in curves])
    ax.set_ylabel('VMAF')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, fontsize='x-small')
    ax.grid(True)
    fig.tight_layout()
    ax.margins(0)
    return fig


def save_figure(fig, path):
    """
    Write the figure under a unique name in the format of the path extension, .png when it has
    none, and return the path written.
    """
    if not os.path.splitext(path)[1]:
        path += '.png'
    image_file, path = create_unique_file(path)
    try:
        with image_file:
            fig.savefig(image_file, format=os.path.splitext(path)[1][1:].lower(), dpi=500, bbox_inches='tight')
    except BaseException:
        os.remove(path)
        raise
    return path


def low_quality_frames(log):
    """
    Frame numbers whose main model score lies below the 1st percentile of the main model scores,
    rounded to 3 decimals, as Plot_Vmaf_improved selected them
    """
    values = next(iter(log.scores.values()))
    threshold = _percentile(values, 1, 3)
    return [number for number, score in zip(log.frame_numbers, values) if score < threshold]


def _timestamp(seconds):
    """HH:MM:SS.mmm"""
    milliseconds = round(seconds * 1000)
    hours, rest = divmod(milliseconds, 3600000)
    minutes, rest = divmod(rest, 60000)
    return f'{hours:02}:{minutes:02}:{rest // 1000:02}.{rest % 1000:03}'


def _draw_overlay(image, lines):
    """The lines in white on a black box at the top left, as Plot_Vmaf_improved drew them"""
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, max(1, int(image.height * 0.03)))
    text = '\n'.join(lines)
    box = draw.multiline_textbbox((10, 10), text, font=font)
    draw.rectangle([(10, 10), (10 + box[2], 10 + box[3])], fill='black')
    draw.multiline_text((10, 10), text, font=font, fill=(255, 255, 255))


def export_tiff_frames(myVmaf, log):
    """
    Export the frames below the 1st percentile of the main model as LZW TIFF files with a text
    overlay into a new folder <log>_lowframes, through the frames pass of myVmaf. Returns the
    folder, or None when no frame is exported.
    """
    numbers = low_quality_frames(log)
    main = next(iter(log.scores))
    label = known_scores()[main][0]
    if not numbers:
        logger.info("No frames below the 1st percentile of %s in %s", label, log.path)
        return None
    log_base = os.path.splitext(log.path)[0]
    folder = create_unique_dir(log_base + '_lowframes')
    try:
        with tempfile.TemporaryDirectory(dir=folder) as work:
            written = myVmaf.getFrames(numbers, work)
            missing = sorted(set(numbers) - {number for number, _, _ in written})
            if missing:
                logger.warning("Frames beyond the end of the distorted input, not exported: %s",
                               ', '.join(str(number) for number in missing))
            score_of = dict(zip(log.frame_numbers, log.scores[main]))
            name = os.path.basename(myVmaf.main.videoSrc)
            for number, path, seconds in written:
                score = score_of[number]
                with Image.open(path) as image:
                    _draw_overlay(image, [name, f'{label}: {score:.3f}', f'Frame: {number}',
                                          f'Time: {_timestamp(seconds)}'])
                    tiff_file, tiff_path = create_unique_file(os.path.join(
                        folder, f'{os.path.basename(log_base)}_VMAF{round(score)}_frame{number:03}.tif'))
                    try:
                        with tiff_file:
                            image.save(tiff_file, format='TIFF', compression='tiff_lzw')
                    except BaseException:
                        os.remove(tiff_path)
                        raise
                # the frames pass file goes once its TIFF file is written, so the disk holds one copy
                os.remove(path)
    except BaseException:
        if not os.listdir(folder):
            os.rmdir(folder)
        raise
    if not os.listdir(folder):
        os.rmdir(folder)
        return None
    return folder


def get_args():
    """The arguments of easyVmafPlusPlot"""
    parser = argparse.ArgumentParser(
        prog='easyVmafPlusPlot',
        description='Plot VMAF logs of easyVmafPlus: per log the temporal plot and the percentile plot, '
                    'and with two or more logs the combined plots.')
    parser.add_argument('vmaf_file', type=str, nargs='+', help='VMAF log: json, xml or csv')
    parser.add_argument('-o', '--output', dest='output', type=str, default='plot.png',
                        help='Combined plot of two or more logs; the combined percentile plot gets _histo '
                             'before the extension. (Default: plot.png).')
    return parser.parse_args()


def main():
    args = get_args()

    # Logs go to stderr; stdout carries only the paths written
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s',
                        datefmt='%H:%M:%S', stream=sys.stderr)

    try:
        logs = [read_vmaf_log(path) for path in args.vmaf_file]
        for log in logs:
            log_base = os.path.splitext(log.path)[0]
            print("Plot output path: ", save_figure(plot_vmaf(log), log_base + '_plot.png'), flush=True)
            print("Percentile plot output path: ",
                  save_figure(plot_percentile_vmaf([log]), log_base + '_histo.png'), flush=True)
        if len(logs) > 1:
            stem, extension = os.path.splitext(args.output)
            print("Combined plot output path: ", save_figure(plot_multi_vmaf(logs), args.output), flush=True)
            print("Combined percentile plot output path: ",
                  save_figure(plot_percentile_vmaf(logs), f'{stem}_histo{extension or ".png"}'), flush=True)
    except (ValueError, OSError) as e:
        _exit_with_error(e)
