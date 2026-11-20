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
import glob
import json
import logging
import math
import os.path
import re
import subprocess
import sys
from signal import signal, SIGINT
from statistics import mean

from .FFmpeg import VMAF_VERSIONS, check_ffmpeg, create_unique_file, read_vmaf_log
from .Vmaf import (vmaf, UnsupportedFramerateError, validate_jobs_config, validate_model_config,
                   validate_range_config)

logger = logging.getLogger(__name__)

# How the validation errors name each vmaf() argument
_FLAG_LABELS = {
    'model': '-model',
    'vmaf_versions': '-vmaf_version',
    'views': '-view',
    'hfr': '-force_hfr',
    'bitdepth': '-bitdepth',
    'enc_size': '-enc_size',
    'enc_bitdepth': '-enc_bitdepth',
    'model_options': '-model_option',
    'start_frame': '-start_frame',
    'frame_count': '-frame_count',
    'subsample': '-subsample',
    'jobs': '-jobs',
    'dynamic_range': '-dynamic_range',
    'reference_color': '-reference_color',
    'distorted_color': '-distorted_color',
}


def handler(signal_received, frame):
    print('SIGINT or CTRL-C detected. Calculation interrupted.', file=sys.stderr)
    sys.exit(130)


def _exit_with_error(message):
    print(f"[easyVmafPlus] ERROR: {message}", file=sys.stderr, flush=True)
    sys.exit(1)


def _build_result(distorted, reference, offset, psnr, model,
                  vmaf_scores=None, vmaf_output_file=None, cambi_heatmap_path=None,
                  plot_path=None, histo_path=None, low_frames_path=None,
                  pix_fmt=None, hfr=None, dynamic_range=None, color=None, frame_range=None):
    """
    The result record of one distorted file, written to the summary file and printed with
    -json: paths, sync offset and PSNR, and after a VMAF run the measurement, the mean score
    per model and the output paths. A non-finite PSNR becomes null with a psnr_status.
    """
    result = {
        'distorted': distorted,
        'reference': reference,
        'sync': {
            'offset': round(offset, 6) if offset is not None else 0.0,
            'psnr': round(psnr, 6) if psnr is not None else None,
        },
    }
    if psnr is not None and not math.isfinite(psnr):
        result['sync']['psnr'] = None
        result['sync']['psnr_status'] = (
            'nan' if math.isnan(psnr) else 'positive_infinity' if psnr > 0 else 'negative_infinity')
    if vmaf_scores is not None:
        vmaf_block = {'model': model}
        if pix_fmt:
            vmaf_block['pix_fmt'] = pix_fmt
        if hfr is not None:
            vmaf_block['hfr'] = hfr
        if dynamic_range:
            vmaf_block['dynamic_range'] = dynamic_range
        if color is not None:
            vmaf_block['color'] = {role: dict(color[role]) for role in ('reference', 'distorted')}
        vmaf_block.update({name: round(score, 6) for name, score in vmaf_scores.items()})
        if vmaf_output_file:
            vmaf_block['output_file'] = vmaf_output_file
        if cambi_heatmap_path:
            vmaf_block['cambi_heatmap_path'] = cambi_heatmap_path
        if plot_path:
            vmaf_block['plot_path'] = plot_path
            vmaf_block['histo_path'] = histo_path
        if low_frames_path:
            vmaf_block['low_frames_path'] = low_frames_path
        if frame_range is not None:
            vmaf_block['range'] = frame_range
        result['vmaf'] = vmaf_block
    return result


def _summary_path(distorted, vmafpath=None):
    """
    The summary file next to the VMAF log, <log>_summary.json; a -sync_only run has no log
    and writes <distorted>_sync_summary.json.
    """
    if vmafpath is None:
        return os.path.splitext(distorted)[0] + '_sync_summary.json'
    return os.path.splitext(vmafpath)[0] + '_summary.json'


def _write_summary(result, path):
    """
    Serialise the whole record as strict JSON, then write it under a unique name, so no
    existing summary is replaced. Returns the path written.
    """
    serialized = json.dumps(result, allow_nan=False, indent=2)
    summary_file, summary_path = create_unique_file(path)
    with summary_file:
        summary_file.write((serialized + '\n').encode('utf-8'))
    return summary_path


def _enc_size(value):
    """argparse type of -enc_size: 'WxH' with positive integers"""
    match = re.fullmatch(r'(\d+)[xX](\d+)', value)
    if not match or not all(int(v) > 0 for v in match.groups()):
        raise argparse.ArgumentTypeError(f"expected WxH, e.g. 1280x720, not '{value}'")
    return int(match.group(1)), int(match.group(2))


def get_args():
    '''This function parses and return arguments passed in'''
    parser = MyParser(prog='easyVmafPlus',
                      description="Script to easy compute VMAF using FFmpeg. It allows to deinterlace, scale and sync Ref and Distorted video samples automatically: \
                        \n\n \t Autodeinterlace: If the Reference or Distorted samples are interlaced, deinterlacing is applied\
                        \n\n \t Autoscale: Reference and Distorted samples are scaled automatically to 1920x1080 or 3840x2160 depending on the VMAF model to use\
                        \n\n \t Autosync: The first frames of the distorted video are used as reference to a sync look up with the Reference video. \
                        \n \t \t The sync is doing by a frame-by-frame look up of the best PSNR\
                        \n \t \t See [-reverse] for more options of syncing\
                        \n\n As output, the per-frame VMAF log (<distorted>_vmaf.json) and a summary of the results (<distorted>_vmaf_summary.json) are created",
                      formatter_class=argparse.RawTextHelpFormatter)

    inputs = parser.add_argument_group('input')
    inputs.add_argument('-d', dest='d', type=str, help='Distorted video, or a glob pattern for batch processing.', required=True)
    inputs.add_argument('-r', dest='r', type=str, help='Reference video.', required=True)
    inputs.add_argument('-fps', dest='fps', type=float, default=0,
                        help='Video Frame Rate: force frame rate conversion to <fps> value. Autodeinterlace is disabled when setting this')

    sync = parser.add_argument_group('synchronization')
    sync.add_argument('-sw', dest='sw', type=float, default=0,
                      help='Sync Window: window size in seconds of a subsample of the Reference video. The sync lookup will be done between the first frames of the Distorted input and this Subsample of the Reference. (default=0. No sync).')
    sync.add_argument('-ss', dest='ss', type=float, default=0,
                      help="Sync Start Time. Time in seconds from the beginning of the Reference video to which the Sync Window will be applied from. Without -sw, the manual offset, with its sign from -reverse. (default=0).")
    sync.add_argument('-reverse', help="If enable, it Changes the default Autosync behaviour: The first frames of the Reference video are used as reference to sync with the Distorted one. (Default = Disable).", action='store_true')
    sync.add_argument('-sync_only', action='store_true', default=False,
                      help='For sync measurement only. No Vmaf processing. Requires -sw.')
    sync.add_argument('-sync_offset', dest='sync_offset', type=float, default=None,
                      help="Manual sync offset in seconds: positive trims the Reference, negative trims the Distorted video, the sign of the reported offset. Not with -sw or -ss. (default=0).")
    sync.add_argument('-endsync', help='Activate end sync. This ends the computation when the shortest video ends. (Default: false).', action='store_true')

    frame_range = parser.add_argument_group(
        'frame range',
        'Measure only a range of frames, numbered as in the log of the full calculation (after sync, deinterlacing and frame rate conversion). Joining the logs of consecutive ranges gives the log of the full calculation. MP4, MOV, Matroska and WebM inputs only.')
    frame_range.add_argument('-start_frame', dest='start_frame', type=int, default=None,
                             help='First frame to measure. (Default: 0).')
    frame_range.add_argument('-frame_count', dest='frame_count', type=int, default=None,
                             help='Number of frames to measure; the last range of a video can have fewer. (Default: to the end).')

    models = parser.add_argument_group('models')
    models.add_argument('-model', dest='model', type=str, default="HD",
                        help="Vmaf Model. Options: HD, 4K. (Default: HD).")
    models.add_argument('-vmaf_version', dest='vmaf_version', nargs='+', choices=VMAF_VERSIONS, default=None,
                        help="VMAF generations to compute: 1 and/or 0.6. '-vmaf_version 1 0.6' computes both in one pass. (Default: 1).")
    models.add_argument('-vmaf_v0', help='Use the VMAF v0.6.1 models instead of the VMAF v1 models, the same as -vmaf_version 0.6. (Default: false).', action='store_true')
    models.add_argument('-view', dest='view', nargs='+', type=str.lower, choices=('3h', '5h', 'phone', '1.5h'), default=None,
                        help="VMAF v1 viewing distance in screen heights (phone = 5h). (Default: 3h 5h for HD; 1.5h 3h for 4K).")
    models.add_argument('-disable_hfr', help='Do not use the VMAF v1 HFR models, which are picked at an effective frame rate of 50 fps or higher. (Default: false).', action='store_true')
    models.add_argument('-force_hfr', help='Use the VMAF v1 HFR models at any frame rate. (Default: false).', action='store_true')
    models.add_argument('-bitdepth', dest='bitdepth', type=str.lower, choices=('auto', '8', '10'), default='auto',
                        help="Measurement bit depth. auto: 10 with VMAF v1 models, otherwise the reference bit depth. (Default: auto).")
    models.add_argument('-dynamic_range', dest='dynamic_range', type=str.lower, choices=('auto', 'sdr', 'hdr'), default='auto',
                        help="Model family. auto: the HDR-VMAF models when either input is PQ (smpte2084), otherwise SDR. hdr forces HDR-VMAF; sdr scores every input with the SDR models. (Default: auto).")
    models.add_argument('-reference_color', dest='reference_color', metavar='COLOR',
                        help="HDR only: reference colorimetry, overriding what the file reports: a preset (hdr10, bt709) or key=value[,key=value] with the keys range, primaries, trc and matrix, e.g. trc=smpte2084.")
    models.add_argument('-distorted_color', dest='distorted_color', metavar='COLOR',
                        help="HDR only: distorted colorimetry, as -reference_color.")

    v1 = parser.add_argument_group('VMAF v1 parameters')
    v1.add_argument('-enc_size', dest='enc_size', type=_enc_size, metavar='WxH',
                    help="VMAF v1 encoding resolution for CAMBI. (Default: distorted video size).")
    v1.add_argument('-enc_bitdepth', dest='enc_bitdepth', type=int, choices=(8, 10, 12),
                    help="VMAF v1 encoding bit depth for CAMBI. (Default: from the distorted pixel format).")
    v1.add_argument('-model_option', dest='model_option', action='append', default=[], type=str.lower,
                    metavar='FEATURE.OPTION=VALUE',
                    help="Advanced VMAF v1 model option override, e.g. cambi.topk=0.5. Repeatable.")

    output = parser.add_argument_group('output')
    output.add_argument('-output_fmt', dest='output_fmt', type=str, default='json',
                        help='Output vmaf file format. Options: json, xml or csv (Default: json)')
    output.add_argument('-cambi_heatmap', help='Compute CAMBI and write the CAMBI heatmaps. (Default: false).', action='store_true')
    output.add_argument('-plot', help='Write the temporal plot and the percentile plot of each distorted file, and combined plots over several distorted files. (Default: false).', action='store_true')
    output.add_argument('-low_frames', help='Export the frames below the 1st percentile of the main VMAF model as TIFF files. (Default: false).', action='store_true')
    output.add_argument('-json', action='store_true', default=False,
                        help='Print the results as JSON on stdout, one object per distorted file. Logs go to stderr. (Default: false).')
    output.add_argument('-verbose', help='Activate verbose loglevel. (Default: info).', action='store_true')
    output.add_argument('-progress', help='Activate progress indicator for vmaf computation. (Default: false).', action='store_true')

    execution = parser.add_argument_group('execution')
    execution.add_argument('-threads', dest='threads', type=int, default=0,
                           help='number of libvmaf threads, also the number of parallel sync passes; per job with -jobs (default=0, one per CPU core, with -jobs CPU count / jobs)')
    execution.add_argument('-jobs', dest='jobs', type=int, default=1,
                           help='Number of FFmpeg processes that compute VMAF at the same time, each on a range of contiguous frames, joined into one log with the same frames and scores as a single process. MP4, MOV, Matroska and WebM inputs only; not with -subsample. (Default: 1).')
    execution.add_argument('-subsample', dest='n', type=int, default=1,
                           help="Specifies the subsampling of frames to speed up calculation. (default=1, None).")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    if args.sync_only and args.sw == 0:
        parser.error('-sync_only requires -sw greater than 0')
    if args.sync_offset is not None:
        if not math.isfinite(args.sync_offset):
            parser.error('-sync_offset must be finite')
        if args.sw != 0:
            parser.error('-sync_offset and -sw are mutually exclusive: -sync_offset is a manual offset, -sw searches for one')
        if args.ss != 0:
            parser.error('-sync_offset and -ss are mutually exclusive: -ss without -sw is the manual offset')
    if args.vmaf_v0 and args.vmaf_version is not None:
        parser.error('-vmaf_v0 and -vmaf_version are mutually exclusive: -vmaf_v0 equals -vmaf_version 0.6')
    if args.disable_hfr and args.force_hfr:
        parser.error('-disable_hfr and -force_hfr are mutually exclusive')
    if args.n < 1:
        parser.error('-subsample must be an integer of at least 1')
    if args.threads < 0:
        parser.error('-threads must be an integer greater than or equal to zero')
    if args.sync_only and (args.start_frame is not None or args.frame_count is not None):
        parser.error('-start_frame and -frame_count measure VMAF; -sync_only does not')
    if args.sync_only and args.jobs > 1:
        parser.error('-jobs splits the VMAF calculation; -sync_only does not compute VMAF')
    if args.vmaf_version:
        args.vmaf_versions = tuple(args.vmaf_version)
    else:
        args.vmaf_versions = ('0.6',) if args.vmaf_v0 else ('1',)
    args.hfr = 'on' if args.force_hfr else 'off' if args.disable_hfr else 'auto'
    try:
        validate_model_config(args.model, args.vmaf_versions, args.view, args.hfr, args.bitdepth,
                              args.enc_size, args.enc_bitdepth, args.model_option, labels=_FLAG_LABELS,
                              dynamic_range=args.dynamic_range, reference_color=args.reference_color,
                              distorted_color=args.distorted_color)
        validate_range_config(args.start_frame, args.frame_count, args.n, labels=_FLAG_LABELS)
        validate_jobs_config(args.jobs, args.n, labels=_FLAG_LABELS)
    except ValueError as e:
        parser.error(str(e))
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
    n_subsample = cmdParser.n
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
    write_plots = cmdParser.plot
    low_frames = cmdParser.low_frames
    has_v1 = '1' in cmdParser.vmaf_versions

    if write_plots or low_frames:
        # matplotlib loads only for runs that plot or export frames
        from . import plot

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
        _exit_with_error(f"FFmpeg {version} detected. easyVmafPlus requires FFmpeg >= 9.0 built with --enable-libvmaf.")
    if not ffmpeg_info['libvmaf']:
        _exit_with_error(f"FFmpeg {version} has no libvmaf filter. Build FFmpeg with --enable-libvmaf.")
    if '0.6' in cmdParser.vmaf_versions and not ffmpeg_info['builtin_models']:
        _exit_with_error(f"FFmpeg {version} is installed but libvmaf built-in models are not available. "
                         f"Rebuild libvmaf with '-Dbuilt_in_models=true' and recompile FFmpeg.")
    if has_v1 and not ffmpeg_info['v1_models']:
        _exit_with_error(f"FFmpeg {version} cannot run the VMAF v1 models. easyVmafPlus needs libvmaf 3.2.1 or later. "
                         f"The v0.6.1 models run with -vmaf_v0.")
    logger.info("FFmpeg %s detected. Built-in models: %s. VMAF v1 models: %s. HDR-VMAF: %s.", version,
                "available" if ffmpeg_info['builtin_models'] else "not available",
                "available" if ffmpeg_info['v1_models'] else "not available",
                "available" if ffmpeg_info['hdr_vmaf'] else "not available")

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

    logs = []
    for distorted in mainFiles:
        try:
            myVmaf = vmaf(distorted, reference, loglevel=loglevel, subsample=n_subsample, model=model,
                          output_fmt=output_fmt, threads=threads, print_progress=print_progress,
                          end_sync=end_sync, manual_fps=fps, cambi_heatmap=cambi_heatmap,
                          vmaf_versions=cmdParser.vmaf_versions, views=cmdParser.view, hfr=cmdParser.hfr,
                          bitdepth=cmdParser.bitdepth, enc_size=cmdParser.enc_size,
                          enc_bitdepth=cmdParser.enc_bitdepth, model_options=cmdParser.model_option,
                          dynamic_range=cmdParser.dynamic_range, reference_color=cmdParser.reference_color,
                          distorted_color=cmdParser.distorted_color, start_frame=cmdParser.start_frame,
                          frame_count=cmdParser.frame_count, jobs=cmdParser.jobs)
            '''check if syncWin was set. If true offset is computed automatically, otherwise manual values are used  '''
            if syncWin > 0:
                offset, psnr = myVmaf.syncOffset(syncWin, ss, reverse)
                if sync_only:
                    result = _build_result(distorted, reference, offset, psnr, model)
                    summary_path = _write_summary(result, _summary_path(distorted))
                    if use_json:
                        print(json.dumps(result, allow_nan=False), flush=True)
                    else:
                        print(f"Results: {distorted} | offset: {offset} | psnr: {psnr}", flush=True)
                        print(f"Summary file path: {summary_path}", flush=True)
                    continue
            elif cmdParser.sync_offset is not None:
                # `or 0.0` also turns -sync_offset -0 into 0.0
                offset = cmdParser.sync_offset or 0.0
                psnr = None
                myVmaf.offset = offset
            else:
                offset = ss
                psnr = None
                if reverse:
                    myVmaf.offset = -offset
                else:
                    myVmaf.offset = offset

            myVmaf.getVmaf()
            vmafpath = myVmaf.ffmpegQos.vmafpath
            names = [vmaf_model.name for vmaf_model in myVmaf.models]
            log = read_vmaf_log(vmafpath)
            scores = {name: mean(log.scores[name]) for name in names}
            plot_path = histo_path = low_frames_path = None
            if write_plots:
                log_base = os.path.splitext(vmafpath)[0]
                plot_path = plot.save_figure(plot.plot_vmaf(log), log_base + '_plot.png')
                histo_path = plot.save_figure(plot.plot_percentile_vmaf([log]), log_base + '_histo.png')
                logs.append(log)
            if low_frames:
                low_frames_path = plot.export_tiff_frames(myVmaf, log)
            frame_range = None
            if myVmaf.start_frame is not None or myVmaf.frame_count is not None:
                frame_range = {'start_frame': myVmaf.start_frame or 0, 'frame_count': myVmaf.frame_count,
                               'frames_scored': myVmaf.frames_scored}
            cambi_path = myVmaf.ffmpegQos.vmaf_cambi_heatmap_path if cambi_heatmap else None
            result = _build_result(distorted, reference, offset, psnr, model,
                                   vmaf_scores=scores, vmaf_output_file=vmafpath,
                                   cambi_heatmap_path=cambi_path, plot_path=plot_path,
                                   histo_path=histo_path, low_frames_path=low_frames_path,
                                   pix_fmt=myVmaf.pix_fmt, hfr=myVmaf.hfr_active,
                                   dynamic_range=myVmaf.dynamic_range, color=myVmaf.color,
                                   frame_range=frame_range)
            summary_path = _write_summary(result, _summary_path(distorted, vmafpath))
        except (UnsupportedFramerateError, ValueError, RuntimeError, subprocess.CalledProcessError, OSError) as e:
            _exit_with_error(e)

        if use_json:
            print(json.dumps(result, allow_nan=False), flush=True)
        else:
            print("\n \n \n \n \n ")
            print("=======================================", flush=True)
            print("Results:", distorted, flush=True)
            print("=======================================", flush=True)
            print("VMAF computed", flush=True)
            print("=======================================", flush=True)
            print("offset: ", offset, " | psnr: ", psnr)
            print(f"pix_fmt: {myVmaf.pix_fmt} | HFR: {'on' if myVmaf.hfr_active else 'off'}")
            print(f"dynamic range: {myVmaf.dynamic_range}")
            if myVmaf.color is not None:
                print("color (range/primaries/trc/matrix): " + " | ".join(
                    f"{role} {'/'.join(myVmaf.color[role].values())}" for role in ('reference', 'distorted')))
            if frame_range is not None:
                last = frame_range['start_frame'] + frame_range['frames_scored'] - 1
                print(f"range: frames {frame_range['start_frame']}-{last} ({frame_range['frames_scored']} frames)")
            for vmaf_model in myVmaf.models:
                print(f"{vmaf_model.label}: ", scores[vmaf_model.name])
            print("VMAF output file path: ", vmafpath)
            print("Summary file path: ", summary_path)
            if cambi_path:
                print("CAMBI Heatmap output path: ", cambi_path)
            if plot_path:
                print("Plot output path: ", plot_path)
                print("Percentile plot output path: ", histo_path)
            if low_frames_path:
                print("Low frames output path: ", low_frames_path)
            print("\n \n \n \n \n ")

    if len(logs) > 1:
        reference_base = os.path.splitext(reference)[0]
        try:
            combined_path = plot.save_figure(plot.plot_multi_vmaf(logs), reference_base + '_vmaf_combined_plot.png')
            combined_histo_path = plot.save_figure(plot.plot_percentile_vmaf(logs),
                                                   reference_base + '_vmaf_combined_histo.png')
        except (ValueError, OSError) as e:
            _exit_with_error(e)
        if use_json:
            logger.info("Combined plot output path: %s", combined_path)
            logger.info("Combined percentile plot output path: %s", combined_histo_path)
        else:
            print("Combined plot output path: ", combined_path)
            print("Combined percentile plot output path: ", combined_histo_path, flush=True)


if __name__ == '__main__':
    main()
