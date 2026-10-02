# easyVmafPlus

easyVmafPlus is an enhanced fork of [easyVmaf](https://github.com/gdavila/easyVmaf) by Gabriel Davila. It is a Python tool built on FFmpeg and ffprobe that prepares a reference and a distorted video for [VMAF](https://github.com/Netflix/vmaf). It handles deinterlacing, upscaling and downscaling, frame-to-frame syncing and frame rate adaptation.

Details about how the original tool works can be found in [this OTTVerse article](https://ottverse.com/vmaf-easyvmaf/).

## What easyVmafPlus adds

| Change | Details |
|---|---|
| Hardware accelerated decoding | Every ffmpeg run passes `-hwaccel auto` to both the distorted and the reference input. When no hardware decoder is available, FFmpeg decodes in software. |
| `easyVmafPlus` command | easyVmafPlus is a Python package. `pipx install .` installs the `easyVmafPlus` command, which runs from any directory. |
| Own Docker image | The Dockerfile builds FFmpeg and libvmaf from source and copies the code from this repository. The image is can be built for `linux/amd64` and `linux/arm64`. This repository offers ready built images under "Packages" |


## Features from easyVmaf

easyVmafPlus includes the features of easyVmaf up to commit [cdcdd80](https://github.com/gdavila/easyVmaf/commit/cdcdd8015ac5e8b84573712963ca4d7135abfeb5), without the CUDA support:

* [Cambi](https://github.com/Netflix/vmaf/blob/master/resource/doc/cambi.md#options), the Netflix banding detector, is computed with `-cambi_heatmap`, which also writes the [Cambi heatmaps](https://github.com/Netflix/vmaf/issues/936). They can be viewed with [ffplay](https://github.com/Netflix/vmaf/issues/1016#issuecomment-1099591977).
* `-json` prints the results as one JSON object per distorted file on stdout.
* `-output_fmt` writes the VMAF log as `json`, `xml` or `csv`.
* The sync passes run in parallel, as many at a time as `-threads`, or one per CPU core.
* At start, easyVmafPlus checks that FFmpeg is 5.0 or later and has libvmaf with its built-in models.
* Log messages go to stderr, with a timestamp.
* File paths with special characters are escaped in the libvmaf filter options.
* The command line follows the [libvmaf filter documentation](https://ffmpeg.org/ffmpeg-filters.html#libvmaf).
* The VMAF models are the built-in models of FFmpeg 5.0 and later.
* With the HD model, the HD, HD Neg and HD Phone scores are computed in one run.

## Requirements

| Requirement | Notes |
|---|---|
| Linux or macOS | |
| Python 3.9 or later | Verified with Python 3.12 (Docker image) and Python 3.14 (macOS) |
| [pipx](https://pipx.pypa.io/) | Installs the `easyVmafPlus` command, together with the Python module [ffmpeg-progress-yield](https://github.com/slhck/ffmpeg-progress-yield) |
| FFmpeg 5.0 or later, built with libvmaf | Since easyVmaf 2.0 only FFmpeg 5.0 and later is supported. For older FFmpeg versions, use easyVmaf 1.3. |

## Installation

```bash
git clone https://github.com/marcelpoelstra/easyVmafPlus.git
cd easyVmafPlus
pipx install .
```

pipx puts the `easyVmafPlus` command in `~/.local/bin`. If that directory is not on your `PATH`, run `pipx ensurepath` once.

To remove the command again:

```bash
pipx uninstall easyVmafPlus
```

### Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

## Usage

```console
$ easyVmafPlus -h
usage: easyVmafPlus [-h] -d D -r R [-sw SW] [-ss SS] [-fps FPS] [-subsample N]
                    [-reverse] [-model MODEL] [-threads THREADS] [-verbose]
                    [-progress] [-endsync] [-output_fmt OUTPUT_FMT]
                    [-cambi_heatmap] [-sync_only] [-json]

Script to easy compute VMAF using FFmpeg. It allows to deinterlace, scale and sync Ref and Distorted video samples automatically:                         

 	 Autodeinterlace: If the Reference or Distorted samples are interlaced, deinterlacing is applied                        

 	 Autoscale: Reference and Distorted samples are scaled automatically to 1920x1080 or 3840x2160 depending on the VMAF model to use                        

 	 Autosync: The first frames of the distorted video are used as reference to a sync look up with the Reference video.                         
 	 	 The sync is doing by a frame-by-frame look up of the best PSNR                        
 	 	 See [-reverse] for more options of syncing                        

 As output, a json file with VMAF score is created

options:
  -h, --help            show this help message and exit
  -sw SW                Sync Window: window size in seconds of a subsample of the Reference video. The sync lookup will be done between the first frames of the Distorted input and this Subsample of the Reference. (default=0. No sync).
  -ss SS                Sync Start Time. Time in seconds from the beginning of the Reference video to which the Sync Window will be applied from. (default=0).
  -fps FPS              Video Frame Rate: force frame rate conversion to <fps> value. Autodeinterlace is disabled when setting this
  -subsample N          Specifies the subsampling of frames to speed up calculation. (default=1, None).
  -reverse              If enable, it Changes the default Autosync behaviour: The first frames of the Reference video are used as reference to sync with the Distorted one. (Default = Disable).
  -model MODEL          Vmaf Model. Options: HD, 4K. (Default: HD).
  -threads THREADS      number of threads, also the number of parallel sync passes (default=0, one per CPU core)
  -verbose              Activate verbose loglevel. (Default: info).
  -progress             Activate progress indicator for vmaf computation. (Default: false).
  -endsync              Activate end sync. This ends the computation when the shortest video ends. (Default: false).
  -output_fmt OUTPUT_FMT
                        Output vmaf file format. Options: json, xml or csv (Default: json)
  -cambi_heatmap        Compute CAMBI and write the CAMBI heatmaps. (Default: false).
  -sync_only            For sync measurement only. No Vmaf processing. Requires -sw.
  -json                 Print the results as JSON on stdout, one object per distorted file. Logs go to stderr. (Default: false).

required arguments:
  -d D                  Distorted video
  -r R                  Reference video 
```

`-d` accepts a glob pattern, for example `"myFolder/video-sample-*.mp4"`. easyVmafPlus then computes VMAF for every matching file against the same reference.

### Output files

| File | Location |
|---|---|
| VMAF log, `<distorted name>_vmaf.json` (`_vmaf.xml` with `-output_fmt xml`, `_vmaf.csv` with `-output_fmt csv`) | Next to the distorted video |
| CAMBI heatmap, `<distorted name>_cambi_heatmap` (with `-cambi_heatmap`) | Next to the distorted video |

## Sync examples

The examples use the samples in `video_samples/`. Run them from that directory.

### Reference delayed against the distorted video

![Sync window applied to the reference video](readme/easyVmaf1.svg)

The first frame of `BBB_sampleA_distorted.mp4` matches the frame at 1.5 seconds in `BBB_reference_10s.mp4`. With `-sw 1 -ss 1`, easyVmafPlus searches a sync window of 1 second, starting 1 second into the reference. It picks the offset with the highest PSNR.

```console
$ easyVmafPlus -r BBB_reference_10s.mp4 -d BBB_sampleA_distorted.mp4 -sw 1 -ss 1
...
=======================================
Results: BBB_sampleA_distorted.mp4
=======================================
VMAF computed
=======================================
offset:  1.5  | psnr:  40.032121
VMAF HD:  90.86849254117647
VMAF Neg:  88.9895249254902
VMAF Phone:  99.9116301372549
VMAF output file path:  BBB_sampleA_distorted_vmaf.json
```

### Distorted video delayed against the reference

![Sync window applied to the distorted video](readme/easyVmaf2.svg)

Here the first frame of the reference `BBB_sampleA_distorted.mp4` matches the frame at 1.0 second in the distorted `BBB_sampleB_distorted.mp4`. The `-reverse` flag applies the sync window to the distorted video instead of the reference. The offset is then reported as a negative value.

```console
$ easyVmafPlus -r BBB_sampleA_distorted.mp4 -d BBB_sampleB_distorted.mp4 -sw 2 -ss 0 -reverse
...
=======================================
Results: BBB_sampleB_distorted.mp4
=======================================
VMAF computed
=======================================
offset:  -1.0  | psnr:  37.254979
VMAF HD:  55.61167999607843
VMAF Neg:  53.97209485098039
VMAF Phone:  74.38952342745098
VMAF output file path:  BBB_sampleB_distorted_vmaf.json
```

### JSON output

With `-json`, stdout carries one JSON object per distorted file. The log messages stay on stderr.

```console
$ easyVmafPlus -r BBB_reference_10s.mp4 -d BBB_sampleA_distorted.mp4 -sw 1 -ss 1 -json 2>/dev/null
{"distorted": "BBB_sampleA_distorted.mp4", "reference": "BBB_reference_10s.mp4", "sync": {"offset": 1.5, "psnr": 40.032121}, "vmaf": {"model": "HD", "vmaf_hd": 90.868493, "vmaf_hd_neg": 88.989525, "vmaf_hd_phone": 99.91163, "output_file": "BBB_sampleA_distorted_vmaf.json"}}
```

## Docker image

The image `ghcr.io/marcelpoelstra/easyvmafplus` is published to the GitHub Container Registry for `linux/amd64` and `linux/arm64`. It is based on `python:3.12-slim`, with FFmpeg 8.1 and libvmaf 3.0.0 built from source. See the [Dockerfile](Dockerfile) for details.

| Tag | Published on |
|---|---|
| `latest` | Every push to `master` |
| `sha-<commit>` | Every push to `master`, with the short commit hash |
| `<version>` | Every version tag `v*`, for example `v1.2.3` publishes `1.2.3` |

To analyse your own files, mount the folder that holds them. The VMAF log is written next to the distorted video, so it ends up in the same folder.

```bash
docker run --rm -v <local-path-to-your-video-files>:/<custom-name-folder> ghcr.io/marcelpoelstra/easyvmafplus -r /<custom-name-folder>/video-1.mp4 -d /<custom-name-folder>/video-2.mp4
```

The image contains the samples from `video_samples/`:

```text
NAME                        TIME

                           t=0
                            |
BBB_reference_10s.mp4       */-----------------------------*/
BBB_sampleA_distorted.mp4           */---------------------*/
BBB_sampleB_distorted.mp4       */-------------------------*/
```

VMAF between `BBB_reference_10s.mp4` and `BBB_sampleA_distorted.mp4`:

```bash
docker run --rm ghcr.io/marcelpoelstra/easyvmafplus -r video_samples/BBB_reference_10s.mp4 -d video_samples/BBB_sampleA_distorted.mp4 -sw 1 -ss 1
```

VMAF between `BBB_sampleA_distorted.mp4` and `BBB_sampleB_distorted.mp4`:

```bash
docker run --rm ghcr.io/marcelpoelstra/easyvmafplus -r video_samples/BBB_sampleA_distorted.mp4 -d video_samples/BBB_sampleB_distorted.mp4 -sw 2 -ss 0 -reverse
```

### Building the image yourself

```bash
docker build -t easyvmafplus .
docker build --platform linux/amd64 -t easyvmafplus:amd64 .
```

The FFmpeg and libvmaf versions are build arguments. These are the defaults:

```bash
docker build --build-arg FFMPEG_version=8.1 --build-arg VMAF_version=3.0.0 -t easyvmafplus .
```

On an Apple silicon Mac, the `linux/amd64` build must run under QEMU emulation 

```bash
docker run --privileged --rm tonistiigi/binfmt --uninstall qemu-x86_64
docker run --privileged --rm tonistiigi/binfmt --install amd64
```

### Docker Compose

`docker-compose.yml` builds the image as `easyvmafplus:latest` and mounts `video_samples/` on `/videos`. Set `VIDEO_DIR` to mount another folder.

```bash
docker compose build
docker compose run --rm easyvmafplus -r /videos/BBB_reference_10s.mp4 -d /videos/BBB_sampleA_distorted.mp4 -sw 1 -ss 1
VIDEO_DIR=<local-path-to-your-video-files> docker compose run --rm easyvmafplus -r /videos/video-1.mp4 -d /videos/video-2.mp4
```



## Licence

MIT, see [LICENSE](LICENSE). The original easyVmaf is written by Gabriel Davila.
