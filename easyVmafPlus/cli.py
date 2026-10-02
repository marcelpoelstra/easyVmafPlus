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

import argparse
import csv
import glob
import json
import logging
import os.path
import subprocess
import sys
import xml.etree.ElementTree as ET
from signal import signal, SIGINT
from statistics import mean

from .FFmpeg import check_ffmpeg, HD_MODEL_NAME, HD_NEG_MODEL_NAME, HD_PHONE_MODEL_NAME, _4K_MODEL_NAME
from .Vmaf import vmaf, UnsupportedFramerateError

logger = logging.getLogger(__name__)


def handler(signal_received, frame):
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    sys.exit(0)


def _exit_with_error(message):
    print(f"[easyVmafPlus] ERROR: {message}", file=sys.stderr, flush=True)
    sys.exit(1)


def _build_result(distorted, reference, offset, psnr, model,
                  vmaf_scores=None, vmaf_output_file=None, cambi_heatmap_path=None):
    """
    The -json result for one distorted file: paths, sync offset and PSNR,
    and after a VMAF run the mean score per model and the output paths.
    """
    result = {
        'distorted': distorted,
        'reference': reference,
        'sync': {
            'offset': round(offset, 6) if offset is not None else 0.0,
            'psnr': round(psnr, 6) if psnr is not None else None,
        },
    }
    if vmaf_scores is not None:
        vmaf_block = {'model': model}
        vmaf_block.update({name: round(score, 6) for name, score in vmaf_scores.items()})
        if vmaf_output_file:
            vmaf_block['output_file'] = vmaf_output_file
        if cambi_heatmap_path:
            vmaf_block['cambi_heatmap_path'] = cambi_heatmap_path
        result['vmaf'] = vmaf_block
    return result


def _read_scores(vmafpath, output_fmt, model):
    """Per-frame scores from the VMAF log, per model name"""
    if model == 'HD':
        names = [HD_MODEL_NAME, HD_NEG_MODEL_NAME, HD_PHONE_MODEL_NAME]
    else:
        names = [_4K_MODEL_NAME]
    scores = {name: [] for name in names}

    if output_fmt == 'csv':
        with open(vmafpath, newline='') as csvFile:
            for row in csv.DictReader(csvFile):
                for name in names:
                    scores[name].append(float(row[name]))
    elif output_fmt == 'xml':
        for frame in ET.parse(vmafpath).getroot().findall('frames/frame'):
            for name in names:
                scores[name].append(float(frame.attrib[name]))
    else:
        with open(vmafpath) as jsonFile:
            for frame in json.load(jsonFile)['frames']:
                for name in names:
                    scores[name].append(frame["metrics"][name])
    return scores


def get_args():
    '''This function parses and return arguments passed in'''
    parser = MyParser(prog='easyVmafPlus',
                      description="Script to easy compute VMAF using FFmpeg. It allows to deinterlace, scale and sync Ref and Distorted video samples automatically: \
                        \n\n \t Autodeinterlace: If the Reference or Distorted samples are interlaced, deinterlacing is applied\
                        \n\n \t Autoscale: Reference and Distorted samples are scaled automatically to 1920x1080 or 3840x2160 depending on the VMAF model to use\
                        \n\n \t Autosync: The first frames of the distorted video are used as reference to a sync look up with the Reference video. \
                        \n \t \t The sync is doing by a frame-by-frame look up of the best PSNR\
                        \n \t \t See [-reverse] for more options of syncing\
                        \n\n As output, a json file with VMAF score is created",
                      formatter_class=argparse.RawTextHelpFormatter)
    requiredgroup = parser.add_argument_group('required arguments')
    requiredgroup.add_argument(
        '-d', dest='d', type=str, help='Distorted video', required=True)
    requiredgroup.add_argument(
        '-r', dest='r', type=str, help='Reference video ', required=True)
    parser.add_argument('-sw', dest='sw', type=float, default=0,
                        help='Sync Window: window size in seconds of a subsample of the Reference video. The sync lookup will be done between the first frames of the Distorted input and this Subsample of the Reference. (default=0. No sync).')
    parser.add_argument('-ss', dest='ss', type=float, default=0,
                        help="Sync Start Time. Time in seconds from the beginning of the Reference video to which the Sync Window will be applied from. (default=0).")
    parser.add_argument('-fps', dest='fps', type=float, default=0,
                        help='Video Frame Rate: force frame rate conversion to <fps> value. Autodeinterlace is disabled when setting this')
    parser.add_argument('-subsample', dest='n', type=int, default=1,
                        help="Specifies the subsampling of frames to speed up calculation. (default=1, None).")
    parser.add_argument('-reverse', help="If enable, it Changes the default Autosync behaviour: The first frames of the Reference video are used as reference to sync with the Distorted one. (Default = Disable).", action='store_true')
    parser.add_argument('-model', dest='model', type=str, default="HD",
                        help="Vmaf Model. Options: HD, 4K. (Default: HD).")
    parser.add_argument('-threads', dest='threads', type=int,
                        default=0, help='number of threads, also the number of parallel sync passes (default=0, one per CPU core)')
    parser.add_argument(
        '-verbose', help='Activate verbose loglevel. (Default: info).', action='store_true')
    parser.add_argument(
        '-progress', help='Activate progress indicator for vmaf computation. (Default: false).', action='store_true')
    parser.add_argument(
        '-endsync', help='Activate end sync. This ends the computation when the shortest video ends. (Default: false).', action='store_true')

    parser.add_argument('-output_fmt', dest='output_fmt', type=str, default='json',
                        help='Output vmaf file format. Options: json, xml or csv (Default: json)')

    parser.add_argument(
        '-cambi_heatmap', help='Compute CAMBI and write the CAMBI heatmaps. (Default: false).', action='store_true')
    parser.add_argument(
        '-sync_only', action='store_true', default=False, help='For sync measurement only. No Vmaf processing. Requires -sw.')
    parser.add_argument(
        '-json', action='store_true', default=False,
        help='Print the results as JSON on stdout, one object per distorted file. Logs go to stderr. (Default: false).')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    if args.sync_only and args.sw == 0:
        parser.error('-sync_only requires -sw greater than 0')
    return args


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def main():
    signal(SIGINT, handler)

    '''reading values from cmdParser'''
    cmdParser = get_args()
    main_pattern = cmdParser.d
    reference = cmdParser.r

    ''' to avoid error negative numbers are not allowed'''
    syncWin = abs(cmdParser.sw)
    ss = abs(cmdParser.ss)
    fps = abs(cmdParser.fps)
    n_subsample = abs(cmdParser.n)
    reverse = cmdParser.reverse
    model = cmdParser.model
    verbose = cmdParser.verbose
    output_fmt = cmdParser.output_fmt
    threads = cmdParser.threads
    print_progress = cmdParser.progress
    end_sync = cmdParser.endsync
    cambi_heatmap = cmdParser.cambi_heatmap
    sync_only = cmdParser.sync_only
    use_json = cmdParser.json

    # Setting verbosity
    if verbose:
        loglevel = "verbose"
    else:
        loglevel = "info"

    # Logs go to stderr; stdout carries only the results
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stderr,
    )

    try:
        ffmpeg_info = check_ffmpeg()
    except RuntimeError as e:
        _exit_with_error(e)
    version = ffmpeg_info['version_str']
    if not ffmpeg_info['meets_minimum']:
        _exit_with_error(f"FFmpeg {version} detected. easyVmafPlus requires FFmpeg >= 5.0 built with --enable-libvmaf.")
    if not ffmpeg_info['libvmaf']:
        _exit_with_error(f"FFmpeg {version} has no libvmaf filter. Build FFmpeg with --enable-libvmaf.")
    if not ffmpeg_info['builtin_models']:
        _exit_with_error(f"FFmpeg {version} is installed but libvmaf built-in models are not available. "
                         f"Rebuild libvmaf with '-Dbuilt_in_models=true' and recompile FFmpeg.")
    logger.info("FFmpeg %s detected. Built-in models: available.", version)

    # check output format
    if output_fmt not in ["json", "xml", "csv"]:
        logger.warning("output_fmt '%s' not supported, using json", output_fmt)
        output_fmt = "json"

    '''
    Distorted video path could be loaded as patterns i.e., "myFolder/video-sample-*.mp4"
    In this way, many computations could be done with just one command line.
    '''
    main_pattern = os.path.expanduser(main_pattern)
    mainFiles = glob.glob(main_pattern)

    if not os.path.isfile(reference):
        _exit_with_error(f"Reference Video file not found: {reference}")

    if len(mainFiles) == 0:
        _exit_with_error(f"Distorted Video files not found with the given pattern/name: {main_pattern}")

    for distorted in mainFiles:
        try:
            myVmaf = vmaf(distorted, reference, loglevel=loglevel, subsample=n_subsample, model=model,
                          output_fmt=output_fmt, threads=threads, print_progress=print_progress,
                          end_sync=end_sync, manual_fps=fps, cambi_heatmap=cambi_heatmap)
            '''check if syncWin was set. If true offset is computed automatically, otherwise manual values are used  '''
            if syncWin > 0:
                offset, psnr = myVmaf.syncOffset(syncWin, ss, reverse)
                if sync_only:
                    if use_json:
                        print(json.dumps(_build_result(distorted, reference, offset, psnr, model)), flush=True)
                    else:
                        print(f"Results: {distorted} | offset: {offset} | psnr: {psnr}", flush=True)
                    continue
            else:
                offset = ss
                psnr = None
                if reverse:
                    myVmaf.offset = -offset
                else:
                    myVmaf.offset = offset

            myVmaf.getVmaf()
            vmafpath = myVmaf.ffmpegQos.vmafpath
            scores = {name: mean(values) for name, values in _read_scores(vmafpath, output_fmt, model).items()}
        except (UnsupportedFramerateError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
            _exit_with_error(e)

        cambi_path = myVmaf.ffmpegQos.vmaf_cambi_heatmap_path if cambi_heatmap else None
        if use_json:
            print(json.dumps(_build_result(distorted, reference, offset, psnr, model,
                                           vmaf_scores=scores, vmaf_output_file=vmafpath,
                                           cambi_heatmap_path=cambi_path)), flush=True)
        else:
            print("\n \n \n \n \n ")
            print("=======================================", flush=True)
            print("Results:", distorted, flush=True)
            print("=======================================", flush=True)
            print("VMAF computed", flush=True)
            print("=======================================", flush=True)
            print("offset: ", offset, " | psnr: ", psnr)
            if model == 'HD':
                print("VMAF HD: ", scores[HD_MODEL_NAME])
                print("VMAF Neg: ", scores[HD_NEG_MODEL_NAME])
                print("VMAF Phone: ", scores[HD_PHONE_MODEL_NAME])
            if model == '4K':
                print("VMAF 4K: ", scores[_4K_MODEL_NAME])
            print("VMAF output file path: ", vmafpath)
            if cambi_path:
                print("CAMBI Heatmap output path: ", cambi_path)
            print("\n \n \n \n \n ")


if __name__ == '__main__':
    main()
