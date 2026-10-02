"""
easyVmafPlus: FFmpeg-based VMAF computation with automatic preprocessing.

Public API:
    from easyVmafPlus import vmaf, UnsupportedFramerateError
    from easyVmafPlus.FFmpeg import FFprobe, FFmpegQos, inputFFmpeg
"""
from .Vmaf import vmaf, UnsupportedFramerateError
from .FFmpeg import FFprobe, FFmpegQos, inputFFmpeg

__all__ = [
    "vmaf",
    "UnsupportedFramerateError",
    "FFprobe",
    "FFmpegQos",
    "inputFFmpeg",
]
