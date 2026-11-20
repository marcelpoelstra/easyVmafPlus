"""
easyVmafPlus: FFmpeg-based VMAF computation with automatic preprocessing.

Public API:
    from easyVmafPlus import vmaf, UnsupportedFramerateError, UnsupportedModelConfigError, UnsupportedRangeError
    from easyVmafPlus.FFmpeg import FFprobe, FFmpegQos, inputFFmpeg
"""
from .Vmaf import vmaf, UnsupportedFramerateError, UnsupportedModelConfigError, UnsupportedRangeError
from .FFmpeg import FFprobe, FFmpegQos, inputFFmpeg

__all__ = [
    "vmaf",
    "UnsupportedFramerateError",
    "UnsupportedModelConfigError",
    "UnsupportedRangeError",
    "FFprobe",
    "FFmpegQos",
    "inputFFmpeg",
]
