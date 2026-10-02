"""Tests for config.py, the FFMPEG and FFPROBE environment variables, the public API of the package
and python -m easyVmafPlus."""

import importlib
import os
import subprocess
import sys

import easyVmafPlus
from easyVmafPlus import FFmpeg, Vmaf, config

EXECUTABLES = 'from easyVmafPlus.FFmpeg import FFmpegQos, FFprobe; print(FFmpegQos._executable); print(FFprobe._executable)'


def _python(*args, env):
    return subprocess.run([sys.executable, *args], env=env, capture_output=True, text=True)


def test_config_uses_which(monkeypatch):
    found = {'ffmpeg': '/opt/ffmpeg/bin/ffmpeg'}
    monkeypatch.setattr('shutil.which', found.get)
    try:
        importlib.reload(config)
        assert (config.ffmpeg, config.ffprobe) == ('/opt/ffmpeg/bin/ffmpeg', None)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_environment_variables():
    env = dict(os.environ, FFMPEG='/x/ffmpeg', FFPROBE='/x/ffprobe')
    assert _python('-c', EXECUTABLES, env=env).stdout.splitlines() == ['/x/ffmpeg', '/x/ffprobe']


def test_without_environment_variables():
    env = {name: value for name, value in os.environ.items() if name not in ('FFMPEG', 'FFPROBE')}
    expected = 'from easyVmafPlus import config; print(config.ffmpeg); print(config.ffprobe)'
    assert _python('-c', EXECUTABLES, env=env).stdout == _python('-c', expected, env=env).stdout


def test_public_api():
    assert easyVmafPlus.__all__ == ['vmaf', 'UnsupportedFramerateError', 'FFprobe', 'FFmpegQos', 'inputFFmpeg']
    assert (easyVmafPlus.vmaf, easyVmafPlus.UnsupportedFramerateError) == (Vmaf.vmaf, Vmaf.UnsupportedFramerateError)
    assert (easyVmafPlus.FFprobe, easyVmafPlus.FFmpegQos, easyVmafPlus.inputFFmpeg) == (
        FFmpeg.FFprobe, FFmpeg.FFmpegQos, FFmpeg.inputFFmpeg)


def test_module_help():
    result = _python('-m', 'easyVmafPlus', '-h', env=os.environ)
    assert result.returncode == 0
    assert result.stdout.startswith('usage: easyVmafPlus [-h] -d D -r R ')


def test_module_without_arguments():
    result = _python('-m', 'easyVmafPlus', env=os.environ)
    assert (result.returncode, result.stdout) == (1, '')
    assert result.stderr.startswith('usage: easyVmafPlus [-h] -d D -r R ')
