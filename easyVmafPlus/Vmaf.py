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
from .FFmpeg import FFprobe
from .FFmpeg import FFmpegQos
from .FFmpeg import select_models
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List
import logging
import math
import os

logger = logging.getLogger(__name__)

# The VMAF v1 HFR models are calibrated for the ~50/60 fps regime
# (Netflix/vmaf v3.2.1, resource/doc/models_v1.md)
HFR_MIN_FPS = 50


@dataclass
class FeatureConfig:
    """
    One entry of the libvmaf feature= option: the feature name and its
    parameters, joined by escaped colons.
    """
    name: str
    params: Dict[str, str] = field(default_factory=dict)

    def to_string(self):
        parts = [f'name={self.name}'] + [f'{k}={v}' for k, v in self.params.items()]
        return '\\\\:'.join(parts)


class UnsupportedFramerateError(ValueError):
    """
    Raised when no deinterlace filter covers the frame rates of REF and MAIN
    for their interlace combination.
    """
    pass


class video():
    """
    Video class to parse information of video streams obtained
    by FFmpeg.FFprobe
    """

    def __init__(self, videoSrc, loglevel="info"):
        self.videoSrc = videoSrc
        self.loglevel = loglevel
        self.streamInfo = None
        self.framesInfo = None
        self.packetsInfo = None
        self._formatInfo = None
        self._interlaced = None
        self.interlacedFrames = None
        self.totalFrames = None
        self.bytesFramesTotal = None
        self.getStreamInfo()
        self.duration = self.getDuration()

    @property
    def formatInfo(self):
        """ffprobe format info, probed on first use"""
        if self._formatInfo is None:
            self.getFormatInfo()
        return self._formatInfo

    @property
    def interlaced(self):
        """True when most frames of the first 5 seconds are interlaced, probed on first use"""
        if self._interlaced is None:
            self.getFramesInfo()
        return self._interlaced

    def _updateFramesSummary(self):
        interlacedFrames_count = 0
        bytesFramesTotal = 0
        if self.framesInfo is None:
            return
        for frame in self.framesInfo:
            interlacedFrames_count = interlacedFrames_count + \
                int(frame['interlaced_frame'])
            bytesFramesTotal = bytesFramesTotal + int(frame['pkt_size'])
        self.interlacedFrames = interlacedFrames_count
        self.totalFrames = len(self.framesInfo)
        self.bytesFramesTotal = bytesFramesTotal
        self._interlaced = bool(round(self.interlacedFrames/self.totalFrames))

    def getDuration(self):
        """Duration minus start time in seconds, floored to milliseconds"""
        try:
            duration = float(self.streamInfo['duration']) - float(self.streamInfo['start_time'])
            if duration < 0:
                duration = float(self.streamInfo['duration'])
        except KeyError:
            duration = float(self.formatInfo['duration']) - float(self.formatInfo['start_time'])
            if duration < 0:
                duration = float(self.formatInfo['duration'])
        return math.floor(duration * 1000) / 1000

    def getStreamInfo(self):
        logger.info("\n\n=======================================")
        logger.info("[easyVmafPlus] Getting stream info... %s", self.videoSrc)
        logger.info("=======================================")
        self.streamInfo = FFprobe(self.videoSrc, self.loglevel).getStreamInfo()
        return self.streamInfo

    def getFramesInfo(self):
        logger.info("\n\n=======================================")
        logger.info("[easyVmafPlus] Getting frames info... %s", self.videoSrc)
        logger.info("=======================================")
        self.framesInfo = FFprobe(self.videoSrc, self.loglevel).getFramesInfo()
        self._updateFramesSummary()
        return self.framesInfo

    def getPacketsInfo(self):
        logger.info("\n\n=======================================")
        logger.info("[easyVmafPlus] Getting packets info... %s", self.videoSrc)
        logger.info("=======================================")
        self.packetsInfo = FFprobe(self.videoSrc, self.loglevel).getPacketsInfo()
        return self.packetsInfo

    def getFormatInfo(self):
        logger.info("\n\n=======================================")
        logger.info("[easyVmafPlus] Getting format info... %s", self.videoSrc)
        logger.info("=======================================")
        self._formatInfo = FFprobe(self.videoSrc, self.loglevel).getFormatInfo()
        logger.debug("%s", self._formatInfo)
        return self._formatInfo


class vmaf():
    """
    Video class to manage VMAF computation of video streams. This class allows:
        - Upscale or downscale the MAIN or REF videos automatically according to the Vmaf model (1080, 4K, etc)
        - Deinterlace automatically the MAIN and REF videos if needed
        - To SYNC (in time) the MAIN and REF videos using psnr computation
        - Frame rate conversion (if needed)
    """

    def __init__(self, mainSrc, refSrc, output_fmt, model="HD", phone=False, loglevel="info", subsample=1, threads=0, print_progress=False, end_sync=False,  manual_fps=0, cambi_heatmap=False, vmaf_v0=False, disable_hfr=False):
        self.loglevel = loglevel
        self.main = video(mainSrc, self.loglevel)
        self.ref = video(refSrc, self.loglevel)
        self.model = model
        self.phone = phone
        self.subsample = subsample
        self.ffmpegQos = FFmpegQos(
            self.main.videoSrc, self.ref.videoSrc, self.loglevel)
        self.target_resolution = None
        self.offset = 0
        self.manual_fps = manual_fps
        self._initResolutions()
        self.output_fmt = output_fmt
        self.threads = threads
        self.print_progress = print_progress
        self.end_sync = end_sync
        self.cambi_heatmap = cambi_heatmap
        self.vmaf_v0 = vmaf_v0
        self.disable_hfr = disable_hfr
        self.models = []
        self._filters_applied = False

    def _initResolutions(self):
        """
        initialization of resolutions for each vmaf model
        """
        if self.model == 'HD':
            self.target_resolution = [1920, 1080]
        elif self.model == '4K':
            self.target_resolution = [3840, 2160]
        else:
            raise ValueError(f"Invalid VMAF model: {self.model!r}. Supported: HD, 4K")

    def _applyScaleFilters(self, qos):
        """
        scaling MAIN and REF in the given FFmpegQos if they dont match with the resolution requiered by the vmaf model (target resolution)
        """
        refResolution = [self.ref.streamInfo['width'],
                         self.ref.streamInfo['height']]
        mainResolution = [self.main.streamInfo['width'],
                          self.main.streamInfo['height']]
        if refResolution != self.target_resolution:
            if not qos.invertedSrc:
                qos.ref.setScaleFilter(
                    self.target_resolution[0], self.target_resolution[1])
            if qos.invertedSrc:
                qos.main.setScaleFilter(
                    self.target_resolution[0], self.target_resolution[1])

        if mainResolution != self.target_resolution:
            if not qos.invertedSrc:
                qos.main.setScaleFilter(
                    self.target_resolution[0], self.target_resolution[1])
            if qos.invertedSrc:
                qos.ref.setScaleFilter(
                    self.target_resolution[0], self.target_resolution[1])

    def _autoScale(self):
        """
        scaling MAIN and REF of self.ffmpegQos (see _applyScaleFilters)
        """
        if self._filters_applied:
            logger.warning("_autoScale() called again without clearFilters(); "
                           "the filter chains can hold duplicate filters")
        self._applyScaleFilters(self.ffmpegQos)
        self._filters_applied = True

    def _deinterlaceFrame(self, factor, stream, fps):
        """yadif frame mode on stream, then fps when REF is not exactly factor times MAIN"""
        ref_fps = getFrameRate(self.ref.streamInfo['r_frame_rate'])
        main_fps = getFrameRate(self.main.streamInfo['r_frame_rate'])

        stream.setDeintFrameFilter()
        if round(ref_fps, 2) != round(factor*main_fps, 2):
            stream.setFpsFilter(round(fps, 5))

    def _deinterlaceField(self, factor, stream, fps):
        """yadif field mode on stream, then fps when REF is not exactly factor times MAIN"""
        ref_fps = getFrameRate(self.ref.streamInfo['r_frame_rate'])
        main_fps = getFrameRate(self.main.streamInfo['r_frame_rate'])

        stream.setDeintFieldFilter()
        if round(ref_fps, 2) != round(factor*main_fps, 2):
            stream.setFpsFilter(round(fps, 5))

    def _applyDeinterlaceFilters(self, qos):
        """
        This functions normalizes the framerate between MAIN and REF video streams in the given FFmpegQos (if needed)
        """
        ref_fps = getFrameRate(self.ref.streamInfo['r_frame_rate'])
        main_fps = getFrameRate(self.main.streamInfo['r_frame_rate'])
        unsupported = (f"No deinterlace filter available for the given framerate combination. "
                       f"ref={round(ref_fps, 5)}fps (interlaced={self.ref.interlaced}), "
                       f"main={round(main_fps, 5)}fps (interlaced={self.main.interlaced}). "
                       f"Consider using the -fps flag to force a frame rate manually.")

        if self.ref.interlaced == self.main.interlaced:
            """ Not Deinterlace would be required. So this functions normalizes the fps between REF and MAIN
            """
            if round(ref_fps) < round(main_fps):
                """
                frame rate conversion over MAIN video. The lowest framerate is choosed (REF fps).
                """
                logger.warning("Frame rate conversion can produce bad vmaf scores")
                qos.main.setFpsFilter(round(ref_fps, 5))
            elif round(ref_fps) > round(main_fps):
                """
                frame rate conversion over REF video. The lowest framerate is choosed (MAIN fps).
                """
                logger.warning("Frame rate conversion can produce bad vmaf scores")
                qos.ref.setFpsFilter(round(main_fps, 5))
            else:
                """
                This just pass the original framerate to the ffmpeg filter (lavfi) when no frame rate
                conversion is requiered. For some reason it is mandatory by lavfi in order to work properly.
                """
                qos.main.setFpsFilter(round(main_fps, 5))
                qos.ref.setFpsFilter(round(ref_fps, 5))

        elif self.ref.interlaced and not self.main.interlaced:
            """
            REF interlaced  | MAIN progressive
            """
            if round(ref_fps) == round(main_fps*2):
                # Examples: REF=60i, MAIN=30p
                # REF=59.97i, MAIN=30p, etc
                if not qos.invertedSrc:
                    self._deinterlaceFrame(2, qos.ref, main_fps)
                else:
                    self._deinterlaceFrame(2, qos.main, main_fps)

            elif round(ref_fps) == round(main_fps):
                # Examples:
                # REF=30i, MAIN=30p
                # REF=29.97i, MAIN=30p, etc
                if not qos.invertedSrc:
                    self._deinterlaceFrame(1, qos.ref, main_fps)
                else:
                    self._deinterlaceFrame(1, qos.main, main_fps)

            elif round(ref_fps) == round(main_fps/2):
                # Examples:
                # REF=30i, MAIN=60p
                # REF=29.97i, MAIN=60p, etc
                if not qos.invertedSrc:
                    self._deinterlaceField(0.5, qos.ref, main_fps)
                else:
                    self._deinterlaceField(0.5, qos.main, main_fps)

            else:
                raise UnsupportedFramerateError(unsupported)

        elif not self.ref.interlaced and self.main.interlaced:
            """
            Input Progressive (REF) | Output Interlaced (MAIN)
            """
            if round(ref_fps) == round(main_fps*2):
                # Examples: REF=60p, MAIN=30i
                # REF=60p, MAIN=29.97i, etc
                if not qos.invertedSrc:
                    self._deinterlaceField(2, qos.main, ref_fps)
                else:
                    self._deinterlaceField(2, qos.ref, ref_fps)

            elif round(ref_fps) == round(main_fps):
                # Examples:
                # REF=30p, MAIN=30i
                # REF=30p, MAIN=29.97i, etc
                if not qos.invertedSrc:
                    self._deinterlaceFrame(1, qos.main, ref_fps)
                else:
                    self._deinterlaceFrame(1, qos.ref, ref_fps)

            elif round(ref_fps) == round(main_fps/2):
                # Examples:
                # REF=30p, MAIN=60i
                logger.warning("Frame rate conversion can produce bad vmaf scores")
                if not qos.invertedSrc:
                    self._deinterlaceField(0.5, qos.main, ref_fps)
                else:
                    self._deinterlaceField(0.5, qos.ref, ref_fps)

            else:
                raise UnsupportedFramerateError(unsupported)

    def _autoDeinterlace(self):
        """
        normalizes the framerate of self.ffmpegQos (see _applyDeinterlaceFilters)
        """
        self._applyDeinterlaceFilters(self.ffmpegQos)

    def _forceFps(self):
        logger.warning("Forcing frame rate conversion manually")
        self.ffmpegQos.main.setFpsFilter(self.manual_fps)
        self.ffmpegQos.ref.setFpsFilter(self.manual_fps)

    def _comparedFrameRate(self):
        """
        Frame rate of the frames libvmaf compares, after _applyDeinterlaceFilters or _forceFps:
        the -fps value; the lower frame rate of two progressive inputs; the frame rate of the
        progressive input when the other is interlaced. None for two interlaced inputs: libvmaf
        then compares interlaced frames, and the r_frame_rate of an interlaced stream can be
        its field rate (the REF=60i, MAIN=30p case of _applyDeinterlaceFilters).
        """
        if self.manual_fps != 0:
            return self.manual_fps
        ref_fps = getFrameRate(self.ref.streamInfo['r_frame_rate'])
        main_fps = getFrameRate(self.main.streamInfo['r_frame_rate'])
        if self.ref.interlaced and self.main.interlaced:
            return None
        if self.ref.interlaced:
            return main_fps
        if self.main.interlaced:
            return ref_fps
        return min(ref_fps, main_fps)

    def _useHfr(self):
        """True for a v1 run without -disable_hfr whose compared frame rate is HFR_MIN_FPS or more"""
        if self.vmaf_v0 or self.disable_hfr:
            return False
        rate = self._comparedFrameRate()
        return rate is not None and rate >= HFR_MIN_FPS

    def _cambiEncodeParams(self):
        """
        Encode-side width, height and bit depth of MAIN for the CAMBI feature of the
        v1 models. Without bits_per_raw_sample CAMBI uses the input bit depth.
        """
        params = {
            'cambi.enc_width': str(self.main.streamInfo['width']),
            'cambi.enc_height': str(self.main.streamInfo['height']),
        }
        bitdepth = str(self.main.streamInfo.get('bits_per_raw_sample', ''))
        if bitdepth.isdigit():
            params['cambi.enc_bitdepth'] = bitdepth
        return params

    def _computePsnrAtOffset(self, offset, reverse):
        """
        PSNR between REF trimmed at offset and MAIN trimmed at 0, both 0.5 s long.
        With reverse, MAIN and REF swap roles. Each call builds its own FFmpegQos,
        so calls can run in parallel. Returns (offset, psnr).
        """
        if not reverse:
            qos = FFmpegQos(self.main.videoSrc, self.ref.videoSrc, self.loglevel)
        else:
            qos = FFmpegQos(self.ref.videoSrc, self.main.videoSrc, self.loglevel)
            qos.invertedSrc = True

        qos.ref.setTrimFilter(offset, 0.5)
        qos.main.setTrimFilter(0, 0.5)
        self._applyScaleFilters(qos)
        if self.manual_fps == 0:
            self._applyDeinterlaceFilters(qos)
        else:
            qos.main.setFpsFilter(self.manual_fps)
            qos.ref.setFpsFilter(self.manual_fps)

        return (offset, qos.getPsnr())

    def syncOffset(self, syncWindow=3, start=0, reverse=False):
        """
        Method to get the offset needed to sync REF and MAIN (if any).
            syncWindow -->  Window Size in seconds to try to sync REF and MAIN videos. i.e., if the video to sync
                            last 600 seconds, the sync look up will be done just within a subsample of syncWindow size.
                            By default, the syncWindow is applied to REF.
            start -->  start time in seconds from the begining of the video where the syncWindow begin.
                        By default, the start time applies to REF.
            reverse --> If this option is set to TRUE. It is considered that MAIN is delayed in comparition to REF: 'syncWindow' and 'start' variables will be
                        applied to MAIN.
                        By default, it is supposed that the REF video is delayed in comparition with the MAIN video.

        It returns the offset value to get REF and MAIN synced and the PSNR computed.
        """

        logger.info("=" * 39)
        logger.info("Syncing... Computing PSNR values...")
        logger.info("=" * 39)
        logger.info("Distorted: %s @ %s fps | %s %s",
                    self.main.videoSrc,
                    round(getFrameRate(self.main.streamInfo['r_frame_rate']), 5),
                    self.main.streamInfo['width'],
                    self.main.streamInfo['height'])
        logger.info("Reference: %s @ %s fps | %s %s",
                    self.ref.videoSrc,
                    round(getFrameRate(self.ref.streamInfo['r_frame_rate']), 5),
                    self.ref.streamInfo['width'],
                    self.ref.streamInfo['height'])
        logger.info("=" * 39)
        logger.info("%-20s %s", "offset(s)", "psnr[dB]")

        fps = getFrameRate(self.ref.streamInfo['r_frame_rate'])
        frameDuration = 1/fps
        startFrame = int(round(start/frameDuration))
        framesInSyncWindow = int(round(syncWindow/frameDuration))
        offsets = [(startFrame + i) * frameDuration for i in range(framesInSyncWindow)]

        if self.manual_fps == 0:
            # Every pass reads the interlace flags: probe them once, before the passes start
            for stream in (self.main, self.ref):
                if stream.framesInfo is None:
                    stream.getFramesInfo()

        max_workers = self.threads if self.threads > 0 else os.cpu_count()
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._computePsnrAtOffset, offset, reverse)
                       for offset in offsets]
            try:
                for future in as_completed(futures):
                    offset, psnr = future.result()
                    results.append((offset, psnr))
                    logger.info("%-20s %s", offset, psnr)
            except BaseException:
                # A failed pass or CTRL-C: drop the passes that have not started
                executor.shutdown(wait=True, cancel_futures=True)
                raise

        # Sorted by offset, so a PSNR tie picks the lowest offset
        results.sort(key=lambda result: result[0])
        offset, maxPsnr = max(results, key=lambda result: result[1])
        """
         The reverse variable/flag indicates if the offset (for time syncing ) was applied over the MAIN or over the REF.
         It is FALSE if it was applied over REF (default) and TRUE if it was applied over the MAIN.
         With reverse, every PSNR pass swapped MAIN and REF in its own FFmpegQos. self.ffmpegQos keeps
         the original MAIN and REF, so only the sign of the offset changes.
        """
        if reverse:
            offset = -1 * offset
        self.offset = offset

        return [self.offset, maxPsnr]

    def setOffset(self, value=None):
        """
        Apply Offset to trim Filter. Runs after the scale and frame rate filters.
            If offset > 0: Ref delayed compared to  Main. Trimfilter cuts Ref
            if offset < 0: Main delayed compared to Ref. Trimfilter cuts Main
        """

        if value is not None:
            """ overrides the value in self.offset"""
            self.offset = value

        if self.offset > 0:
            offset = self.offset
            duration = min(self.main.duration, self.ref.duration-offset)
            self.ffmpegQos.ref.setTrimFilter(offset, duration)
            self.ffmpegQos.main.setTrimFilter(0, duration)

        elif self.offset < 0:
            offset = abs(self.offset)
            duration = min(self.main.duration - offset, self.ref.duration)
            self.ffmpegQos.main.setTrimFilter(offset, duration)
            self.ffmpegQos.ref.setTrimFilter(0, duration)

    def _build_feature_string(self):
        """
        The libvmaf feature= value: psnr always, cambi with -cambi_heatmap,
        the features joined by |.
        """
        features: List[FeatureConfig] = [FeatureConfig('psnr')]
        if self.cambi_heatmap:
            features.append(FeatureConfig('cambi', {
                'full_ref': 'true',
                'enc_width': str(self.main.streamInfo['width']),
                'enc_height': str(self.main.streamInfo['height']),
                'src_width': str(self.ref.streamInfo['width']),
                'src_height': str(self.ref.streamInfo['height']),
            }))
        return '|'.join(feature.to_string() for feature in features)

    def getVmaf(self, autoSync=False):
        """
        Filter order: clearFilters, _autoScale, _autoDeinterlace or _forceFps,
        syncOffset when autoSync, setOffset, and for the v1 models the 10-bit format.
        """
        self.ffmpegQos.clearFilters()
        self.ffmpegQos.main.clearFilters()
        self.ffmpegQos.ref.clearFilters()
        self._filters_applied = False

        """AutoScale according to vmaf model and deinterlace the source if needed """
        self._autoScale()

        if self.manual_fps == 0:
            self._autoDeinterlace()
        else:
            self._forceFps()

        """Lookup for sync between Main and reference. Default: dissable
           It is suggested to run syncOffset manually before getVmaf()
        """
        if autoSync:
            self.syncOffset()
        """Apply Offset filters, if offset =0 nothing happens """
        self.setOffset()

        # VMAF v1: HFR variants, CAMBI encode parameters and 10-bit input
        # (Netflix/vmaf v3.2.1, resource/doc/models_v1.md)
        self.models = select_models(self.model, vmaf_v0=self.vmaf_v0, hfr=self._useHfr())
        if not self.vmaf_v0:
            cambi_params = self._cambiEncodeParams()
            for model in self.models:
                model.params.update(cambi_params)
            self.ffmpegQos.main.setFormatFilter('yuv420p10le')
            self.ffmpegQos.ref.setFormatFilter('yuv420p10le')

        self.features = self._build_feature_string()

        logger.info("=" * 39)
        logger.info("Computing VMAF...")
        logger.info("=" * 39)
        logger.info("Distorted: %s @ %s fps | %s %s",
                    self.main.videoSrc,
                    round(getFrameRate(self.main.streamInfo['r_frame_rate']), 5),
                    self.main.streamInfo['width'],
                    self.main.streamInfo['height'])
        logger.info("Reference: %s @ %s fps | %s %s",
                    self.ref.videoSrc,
                    round(getFrameRate(self.ref.streamInfo['r_frame_rate']), 5),
                    self.ref.streamInfo['width'],
                    self.ref.streamInfo['height'])
        logger.info("Offset:     %s", self.offset)
        logger.info("Model:      %s", self.model)
        logger.info("Phone:      %s", self.phone)
        logger.debug("loglevel:   %s", self.loglevel)
        logger.info("subsample:  %s", self.subsample)
        logger.info("output_fmt: %s", self.output_fmt)
        logger.info("=" * 39)

        vmafProcess = self.ffmpegQos.getVmaf(models=self.models, subsample=self.subsample,
                                             output_fmt=self.output_fmt, threads=self.threads, print_progress=self.print_progress, end_sync=self.end_sync, features=self.features, cambi_heatmap=self.cambi_heatmap)
        return vmafProcess


def getFrameRate(r_frame_rate):
    num, den = r_frame_rate.split('/')
    return int(num)/int(den)
