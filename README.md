# easyVmafPlus

easyVmafPlus is an enhanced fork of [easyVmaf](https://github.com/gdavila/easyVmaf) by Gabriel Davila. It is a Python tool built on FFmpeg and ffprobe that prepares a reference and a distorted video for [VMAF](https://github.com/Netflix/vmaf). It handles deinterlacing, upscaling and downscaling, frame-to-frame syncing and frame rate adaptation.

Details about how the original tool works can be found in [this OTTVerse article](https://ottverse.com/vmaf-easyvmaf/).

## What easyVmafPlus adds

| Change | Details |
|---|---|
| Hardware accelerated decoding | Every ffmpeg run passes `-hwaccel auto` to both the distorted and the reference input. When no hardware decoder is available, FFmpeg decodes in software. |
| VMAF v1 models | easyVmafPlus computes VMAF with the VMAF v1 models of libvmaf 3.2.1 by default, in their HFR variants at 50 frames per second or more. `-vmaf_v0` runs the v0.6.1 models, `-disable_hfr` the standard v1 models. See [VMAF models](#vmaf-models). |
| More decoders in the Docker image | The image decodes JPEG XL with libjxl, AVS2 with libdavs2 and AVS3 with libuavs3d. FFmpeg 9.0 adds Animated WebP. |
| `easyVmafPlus` command | easyVmafPlus is a Python package. `pipx install .` installs the `easyVmafPlus` command, which runs from any directory. |
| Own Docker image | The Dockerfile builds FFmpeg and libvmaf from source and copies the code from this repository. The image is can be built for `linux/amd64` and `linux/arm64`. This repository offers ready built images under "Packages" |


## Features from easyVmaf

easyVmafPlus includes the features of easyVmaf up to commit [cdcdd80](https://github.com/gdavila/easyVmaf/commit/cdcdd8015ac5e8b84573712963ca4d7135abfeb5), without the CUDA support:

* [Cambi](https://github.com/Netflix/vmaf/blob/master/resource/doc/cambi.md#options), the Netflix banding detector, is computed with `-cambi_heatmap`, which also writes the [Cambi heatmaps](https://github.com/Netflix/vmaf/issues/936). They can be viewed with [ffplay](https://github.com/Netflix/vmaf/issues/1016#issuecomment-1099591977).
* `-json` prints the results as one JSON object per distorted file on stdout.
* `-output_fmt` writes the VMAF log as `json`, `xml` or `csv`.
* The sync passes run in parallel, as many at a time as `-threads`, or one per CPU core.
* At start, easyVmafPlus checks that FFmpeg is 9.0 or later and that libvmaf runs the VMAF v1 models, or with `-vmaf_v0` its built-in models.
* Log messages go to stderr, with a timestamp.
* File paths with special characters are escaped in the libvmaf filter options.
* The command line follows the [libvmaf filter documentation](https://ffmpeg.org/ffmpeg-filters.html#libvmaf).
* With `-vmaf_v0`, the VMAF models are the built-in v0.6.1 models of libvmaf.
* With `-vmaf_v0` and the HD model, the HD, HD Neg and HD Phone scores are computed in one run.

## Requirements

| Requirement | Notes |
|---|---|
| Linux or macOS | |
| Python 3.11 or later | Verified with Python 3.12 (Docker image) and Python 3.14 (macOS) |
| [pipx](https://pipx.pypa.io/) | Installs the `easyVmafPlus` command, together with the Python module [ffmpeg-progress-yield](https://github.com/slhck/ffmpeg-progress-yield) |
| FFmpeg 9.0 or later, built with libvmaf 3.2.1 or later | The VMAF v1 models need libvmaf 3.2.1 or later. |

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
                    [-reverse] [-model MODEL] [-vmaf_v0] [-disable_hfr]
                    [-threads THREADS] [-verbose] [-progress] [-endsync]
                    [-output_fmt OUTPUT_FMT] [-cambi_heatmap] [-sync_only]
                    [-json]

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
  -vmaf_v0              Use the VMAF v0.6.1 models instead of the VMAF v1 models. (Default: false).
  -disable_hfr          Do not use the VMAF v1 HFR models, which are picked at a compared frame rate of 50 fps or higher. (Default: false).
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

## VMAF models

By default easyVmafPlus computes VMAF with the VMAF v1 models of libvmaf 3.2.1, described in the [libvmaf model documentation](https://github.com/Netflix/vmaf/blob/v3.2.1/resource/doc/models_v1.md). `-vmaf_v0` runs the v0.6.1 models instead.

| `-model` | Default run | With `-vmaf_v0` |
|---|---|---|
| HD | `vmaf_v1_hd` (1080p, 3H), `vmaf_v1_hd_phone` (phone, 5H) | `vmaf_hd`, `vmaf_hd_neg`, `vmaf_hd_phone` |
| 4K | `vmaf_v1_4k` (2160p, 1.5H), `vmaf_v1_4k_3h` (2160p, 3H, scores up to 110) | `vmaf_4k` |

These are the score names in the VMAF log and in the `-json` output.

When the frames are compared at 50 frames per second or more, the v1 models run in their HFR variants, and the score names end in `_hfr`. The compared frame rate is the `-fps` value, or else the lower frame rate of two progressive inputs. With one interlaced input, it is the frame rate of the progressive input. Two interlaced inputs keep the standard v1 models, and so does `-disable_hfr`.

A v1 run converts both inputs to 10 bits (`yuv420p10le`), as the libvmaf documentation recommends. It also passes the width, height and bit depth of the distorted video to the CAMBI feature of the v1 models.

The v1 model files ship with easyVmafPlus in `easyVmafPlus/models/`, under the BSD+Patent licence of libvmaf: see [easyVmafPlus/models/LICENSE](easyVmafPlus/models/LICENSE).

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
VMAF v1 HD:  93.25625810196078
VMAF v1 Phone:  95.39317779607843
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
VMAF v1 HD:  69.45323145490197
VMAF v1 Phone:  75.73841608235294
VMAF output file path:  BBB_sampleB_distorted_vmaf.json
```

### JSON output

With `-json`, stdout carries one JSON object per distorted file. The log messages stay on stderr.

```console
$ easyVmafPlus -r BBB_reference_10s.mp4 -d BBB_sampleA_distorted.mp4 -sw 1 -ss 1 -json 2>/dev/null
{"distorted": "BBB_sampleA_distorted.mp4", "reference": "BBB_reference_10s.mp4", "sync": {"offset": 1.5, "psnr": 40.032121}, "vmaf": {"model": "HD", "vmaf_v1_hd": 93.256258, "vmaf_v1_hd_phone": 95.393178, "output_file": "BBB_sampleA_distorted_vmaf.json"}}
```

## Docker image

The image `ghcr.io/marcelpoelstra/easyvmafplus` is published to the GitHub Container Registry for `linux/amd64` and `linux/arm64`. It is based on `python:3.12-slim`, with FFmpeg 9.0.2, libvmaf 3.2.1, davs2 1.7 and uavs3d 1.2 built from source. See the [Dockerfile](Dockerfile) for details.

FFmpeg in the image includes libdavs2, which is GPL, so the FFmpeg build is licensed under GPL version 3 or later. It is built from these sources:

| Component | Source |
|---|---|
| FFmpeg 9.0.2 | `https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n9.0.2.tar.gz` |
| libvmaf 3.2.1 | `https://github.com/Netflix/vmaf/archive/v3.2.1.tar.gz` |
| davs2 1.7 | `https://github.com/pkuvcl/davs2/archive/refs/tags/1.7.tar.gz` |
| uavs3d 1.2 | `https://github.com/uavs3/uavs3d/archive/refs/tags/1.2.tar.gz` |
| dav1d, libjxl | Debian 13 packages `dav1d` and `libjxl0.11` |

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

The FFmpeg, libvmaf, davs2 and uavs3d versions are build arguments. These are the defaults:

```bash
docker build --build-arg FFMPEG_version=9.0.2 --build-arg VMAF_version=3.2.1 --build-arg DAVS2_version=1.7 --build-arg UAVS3D_version=1.2 -t easyvmafplus .
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

MIT, see [LICENSE](LICENSE). The original easyVmaf is written by Gabriel Davila. The VMAF v1 model files in `easyVmafPlus/models/` are under the BSD+Patent licence of libvmaf, see [easyVmafPlus/models/LICENSE](easyVmafPlus/models/LICENSE). The FFmpeg build in the Docker image is GPL version 3 or later, see [Docker image](#docker-image).
