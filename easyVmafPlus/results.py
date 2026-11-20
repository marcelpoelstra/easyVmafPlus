"""
MIT License

Copyright (c) 2020 Gabriel Davila - gdavila.revelo@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
"""
Trim and join libvmaf logs and CAMBI heatmaps of frame ranges, as upstream
easyVmaf 5.2.0 easyvmaf/results.py does.
"""
import csv
import glob
import json
import os
import shutil
import xml.etree.ElementTree as ET
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple


def read_frames(log_path: str, output_fmt: str) -> List[Tuple[int, Dict[str, float]]]:
    """(frame number, {metric: value}) of every frame of a libvmaf log, in order."""
    if output_fmt == 'csv':
        with open(log_path, mode='r', newline='') as csvFile:
            return [(int(row['Frame']),
                     {k: float(v) for k, v in row.items() if k and k != 'Frame' and v})
                    for row in csv.DictReader(csvFile)]
    if output_fmt == 'xml':
        root = ET.parse(log_path).getroot()
        return [(int(frame.get('frameNum')),
                 {k: float(v) for k, v in frame.attrib.items() if k != 'frameNum'})
                for frame in root.findall('frames/frame')]
    with open(log_path) as jsonFile:
        return [(frame['frameNum'], frame['metrics'])
                for frame in json.load(jsonFile)['frames']]


def _pool(values: Sequence[float]) -> Dict[str, float]:
    """libvmaf pooled metrics; its harmonic mean is n / sum(1 / (x + 1)) - 1."""
    return {
        'min': min(values),
        'max': max(values),
        'mean': mean(values),
        'harmonic_mean': len(values) / sum(1.0 / (v + 1.0) for v in values) - 1.0,
    }


def trim_log(log_path: str, output_fmt: str, start: int, count: int, first_frame: int):
    """
    Keep count frames of a libvmaf log from its frame start, in place. The kept
    frames are numbered first_frame plus their position in the original log,
    and the pooled metrics are recomputed from them.
    """
    if output_fmt == 'csv':
        with open(log_path, mode='r', newline='') as csvFile:
            rows = list(csv.reader(csvFile))
        header, frames = rows[0], rows[1 + start:1 + start + count]
        column = header.index('Frame')
        for row in frames:
            row[column] = str(int(row[column]) + first_frame)
        with open(log_path, mode='w', newline='') as csvFile:
            csv.writer(csvFile, lineterminator='\n').writerows([header] + frames)
    elif output_fmt == 'xml':
        tree = ET.parse(log_path)
        root = tree.getroot()
        parent = root.find('frames')
        frames = parent.findall('frame')
        for frame in frames[:start] + frames[start + count:]:
            parent.remove(frame)
        kept = frames[start:start + count]
        for frame in kept:
            frame.set('frameNum', str(int(frame.get('frameNum')) + first_frame))
        _repool_xml(root, kept)
        tree.write(log_path)
    else:
        with open(log_path) as jsonFile:
            log = json.load(jsonFile)
        kept = log['frames'][start:start + count]
        for frame in kept:
            frame['frameNum'] += first_frame
        log['frames'] = kept
        _repool_json(log)
        with open(log_path, 'w') as jsonFile:
            json.dump(log, jsonFile, indent=2)


def _repool_xml(root, frames):
    """Recompute the pooled metrics of an xml log from its frames."""
    for metric in root.findall('pooled_metrics/metric'):
        name = metric.get('name')
        pooled = _pool([float(frame.get(name)) for frame in frames])
        for key, value in pooled.items():
            metric.set(key, '%.6f' % value)


def _repool_json(log):
    """Recompute the pooled metrics of a json log from its frames."""
    log['pooled_metrics'] = {
        name: {key: round(value, 6) for key, value in
               _pool([frame['metrics'][name] for frame in log['frames']]).items()}
        for name in log.get('pooled_metrics', {})}


def _joined_fps(counts: Sequence[int], speeds: Sequence[Optional[float]]) -> Optional[float]:
    """libvmaf's fps of joined logs: all the frames over the CPU time of every log."""
    if not all(speeds):
        return None
    return sum(counts) / sum(n / fps for n, fps in zip(counts, speeds, strict=True))


def join_logs(log_paths: Sequence[str], output_fmt: str, out_path: str):
    """
    Write to out_path the frames of the libvmaf logs of consecutive ranges, in
    order, with the pooled metrics and the processing speed (fps) recomputed
    from all of them, in the same format. The rest comes from the first log.
    """
    if output_fmt == 'csv':
        header, rows = None, []
        for path in log_paths:
            with open(path, mode='r', newline='') as csvFile:
                part = list(csv.reader(csvFile))
            header = header or part[0]
            rows += part[1:]
        with open(out_path, mode='w', newline='') as csvFile:
            csv.writer(csvFile, lineterminator='\n').writerows([header] + rows)
    elif output_fmt == 'xml':
        trees = [ET.parse(path) for path in log_paths]
        root = trees[0].getroot()
        parent = root.find('frames')
        counts, speeds = [len(parent.findall('frame'))], []
        for tree in trees[1:]:
            frames = tree.getroot().findall('frames/frame')
            counts.append(len(frames))
            parent.extend(frames)
        for tree in trees:
            fyi = tree.getroot().find('fyi')
            speeds.append(float(fyi.get('fps')) if fyi is not None and fyi.get('fps') else None)
        _repool_xml(root, parent.findall('frame'))
        fps = _joined_fps(counts, speeds)
        if fps is not None:
            root.find('fyi').set('fps', '%.2f' % fps)
        trees[0].write(out_path)
    else:
        logs = []
        for path in log_paths:
            with open(path) as jsonFile:
                logs.append(json.load(jsonFile))
        log = logs[0]
        fps = _joined_fps([len(part['frames']) for part in logs],
                          [part.get('fps') for part in logs])
        log['frames'] = [frame for part in logs for frame in part['frames']]
        _repool_json(log)
        if fps is not None:
            log['fps'] = round(fps, 2)
        with open(out_path, 'w') as jsonFile:
            json.dump(log, jsonFile, indent=2)


def heatmap_files(path: str) -> List[str]:
    """CAMBI heatmaps of a libvmaf heatmaps_path directory, one per scale."""
    return sorted(glob.glob(os.path.join(glob.escape(path), 'cambi_heatmap_scale_*.gray')))


def trim_heatmaps(path: str, start: int, count: int, frames: int, block: int = 1 << 24):
    """
    Keep count pictures of each CAMBI heatmap from its picture start, in place.
    libvmaf writes picture n of a run at n times the picture size, which comes
    from the file size and the frames measured.

    Raises:
        ValueError: a heatmap does not hold one picture per measured frame
    """
    for heatmap in heatmap_files(path):
        size = os.path.getsize(heatmap)
        if size == 0 or size % frames:
            raise ValueError(f"CAMBI heatmap {heatmap} ({size} bytes) does not hold "
                             f"{frames} pictures, one per measured frame")
        picture = size // frames
        begin, end = start * picture, (start + count) * picture
        with open(heatmap, 'r+b') as f:
            for offset in range(begin, end, block):
                f.seek(offset)
                data = f.read(min(block, end - offset))
                f.seek(offset - begin)
                f.write(data)
            f.truncate(end - begin)


def join_heatmaps(paths: Sequence[str], out_path: str):
    """
    Concatenate, file by file, the CAMBI heatmaps of consecutive ranges, in
    order, into the heatmap directory out_path, which exists and is empty.

    Raises:
        ValueError: the ranges do not have the same heatmap files
    """
    names = [os.path.basename(p) for p in heatmap_files(paths[0])]
    for path in paths[1:]:
        if [os.path.basename(p) for p in heatmap_files(path)] != names:
            raise ValueError(f"CAMBI heatmaps of {path} differ from those of {paths[0]}")
    for name in names:
        with open(os.path.join(out_path, name), 'wb') as joined:
            for path in paths:
                with open(os.path.join(path, name), 'rb') as part:
                    shutil.copyfileobj(part, joined, 1 << 24)
