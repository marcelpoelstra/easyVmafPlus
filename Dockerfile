# Global ARGs, redeclared in the stages that use them.
# FFmpeg and libvmaf are commits of master: HDR-VMAF needs the built-in hdr_vmaf_v1.1.8 models
# (libvmaf 74f930d1) and the frame colorimetry vf_libvmaf passes to libvmaf (FFmpeg 58584087cf),
# in no release of either yet
ARG FFMPEG_commit=88c46fce27e311c73772d9b83db4047dd83b1294
ARG VMAF_commit=a6ed0f18285cab504b779bbccb0bbaddc3bf44c2
ARG ZIMG_version=3.0.6
ARG DAVS2_version=1.7
ARG UAVS3D_version=1.2

FROM python:3.12-slim AS base

FROM base AS build

ARG FFMPEG_commit
ARG VMAF_commit
ARG ZIMG_version
ARG DAVS2_version
ARG UAVS3D_version

# get and install building tools
RUN \
	export TZ='UTC' && \
	ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
	apt-get update -yqq && \
	apt-get install --no-install-recommends \
		ninja-build \
		wget \
		doxygen \
		autoconf \
		automake \
		cmake \
		g++ \
		gcc \
		libdav1d-dev \
		libjxl-dev \
		libtool \
		pkg-config \
		make \
		nasm \
		xxd \
		yasm -y && \
	apt-get autoremove -y && \
	apt-get clean -y && \
	pip3 install --user meson

# install zimg, which libvmaf converts the HDR pictures with; from source, so that every
# platform of the image uses the same build
WORKDIR /tmp/zimg
RUN \
	wget https://github.com/sekrit-twc/zimg/archive/refs/tags/release-${ZIMG_version}.tar.gz && \
	tar -xzf release-${ZIMG_version}.tar.gz && \
	cd zimg-release-${ZIMG_version} && \
	./autogen.sh && \
	./configure --prefix=/usr/local --libdir=/usr/local/lib --disable-static && \
	make -j$(nproc) && \
	make install && \
	ldconfig && \
	install -Dm644 COPYING /usr/local/share/licenses/zimg/COPYING && \
	rm -rf /tmp/zimg

# install libvmaf; -Denable_zimg=true converts the pictures to the HDR-VMAF models' PQ,
# BT.2020 and ICtCp with zimg, from the colorimetry FFmpeg passes
WORKDIR /tmp/vmaf
RUN \
	export PATH="${HOME}/.local/bin:${PATH}" && \
	wget https://github.com/Netflix/vmaf/archive/${VMAF_commit}.tar.gz && \
	tar -xzf ${VMAF_commit}.tar.gz && \
	cd vmaf-${VMAF_commit}/libvmaf/ && \
	meson build --buildtype release -Dbuilt_in_models=true -Denable_zimg=true --libdir=lib && \
	ninja -vC build && \
	ninja -vC build test && \
	ninja -vC build install && \
	ldconfig && \
	mkdir -p /usr/local/share/model && \
	cp -R ../model/* /usr/local/share/model && \
	install -Dm644 ../LICENSE /usr/local/share/licenses/libvmaf/LICENSE && \
	rm -rf /tmp/vmaf

# install davs2, the AVS2 decoder; tag 1.7 lacks the aarch64 assembly its Makefile lists,
# and its configure accepts 8-bit output only
WORKDIR /tmp/davs2
RUN \
	wget https://github.com/pkuvcl/davs2/archive/refs/tags/${DAVS2_version}.tar.gz && \
	tar -xzf ${DAVS2_version}.tar.gz && \
	cd davs2-${DAVS2_version}/build/linux && \
	if [ "$(uname -m)" = "aarch64" ]; then asm_flag="--disable-asm"; else asm_flag=""; fi && \
	./configure --enable-shared --disable-static --disable-cli $asm_flag && \
	make -j$(nproc) && \
	make install && \
	rm -rf /tmp/davs2

# install uavs3d, the AVS3 decoder
WORKDIR /tmp/uavs3d
RUN \
	wget https://github.com/uavs3/uavs3d/archive/refs/tags/${UAVS3D_version}.tar.gz && \
	tar -xzf ${UAVS3D_version}.tar.gz && \
	mkdir -p uavs3d-${UAVS3D_version}/build/linux && \
	cd uavs3d-${UAVS3D_version}/build/linux && \
	cmake -DCOMPILE_10BIT=1 -DBUILD_SHARED_LIBS=1 ../.. && \
	make -j$(nproc) && \
	make install && \
	rm -rf /tmp/uavs3d

# install ffmpeg from a commit archive of master; libdavs2 is GPL, so the build needs --enable-gpl.
# The pax header of the archive names its commit, and the build stops when it is not the pinned
# one. HAVE_LIBVMAF_PICTURE_CONVERT in config.h means the libvmaf filter passes the frame
# colorimetry. A commit archive has no .git, so make REVISION= gives ffmpeg the version
# git-<hash>, which check_ffmpeg reads as a dev build. The licence texts and SOURCE under
# /usr/local/share/licenses/ffmpeg say where the source code of the binaries is.
WORKDIR /tmp/ffmpeg
RUN \
	export LD_LIBRARY_PATH="/usr/local/lib" && \
	export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" && \
	wget -O ffmpeg.tar.gz https://github.com/FFmpeg/FFmpeg/archive/${FFMPEG_commit}.tar.gz && \
	commit=$(python3 -c "import sys, tarfile; t = tarfile.open(sys.argv[1]); t.next(); print(t.pax_headers['comment'])" ffmpeg.tar.gz) && \
	{ [ "${commit}" = "${FFMPEG_commit}" ] || { echo "FFmpeg archive is commit ${commit}, not ${FFMPEG_commit}"; exit 1; }; } && \
	sha256=$(sha256sum ffmpeg.tar.gz | cut -d' ' -f1) && \
	tar -xzf ffmpeg.tar.gz && \
	cd FFmpeg-${FFMPEG_commit} && \
	./configure --enable-libvmaf --enable-version3 --enable-gpl --enable-shared --enable-libdav1d --enable-libjxl --enable-libdavs2 --enable-libuavs3d && \
	grep -q '^#define HAVE_LIBVMAF_PICTURE_CONVERT 1$' config.h && \
	revision="git-$(echo ${FFMPEG_commit} | cut -c1-12)" && \
	make -j$(nproc) REVISION="${revision}" && \
	make install REVISION="${revision}" && \
	install -Dm644 -t /usr/local/share/licenses/ffmpeg COPYING.GPLv3 COPYING.LGPLv3 LICENSE.md && \
	printf '%s\n' \
		"FFmpeg ${revision} (master), GPL-3.0-or-later (--enable-gpl --enable-version3)." \
		"" \
		"Its source code, unmodified, is the archive this image was built from:" \
		"  https://github.com/FFmpeg/FFmpeg/archive/${FFMPEG_commit}.tar.gz" \
		"  sha256 ${sha256}" \
		"It is https://git.ffmpeg.org/ffmpeg.git at commit ${commit}." \
		"" \
		"It was configured and built by the Dockerfile of https://github.com/marcelpoelstra/easyVmafPlus." \
		"ffmpeg -buildconf prints its configure flags." \
		> /usr/local/share/licenses/ffmpeg/SOURCE && \
	rm -rf /tmp/ffmpeg

FROM base AS release

ARG FFMPEG_commit
ARG VMAF_commit
ARG ZIMG_version
ARG DAVS2_version
ARG UAVS3D_version

LABEL org.opencontainers.image.title="easyVmafPlus"
LABEL org.opencontainers.image.description="FFmpeg-based VMAF computation with automatic deinterlacing, scaling and sync, hardware accelerated decoding and HDR-VMAF"
# easyVmafPlus (MIT), the model files (BSD-2-Clause-Patent), FFmpeg with libdavs2 (GPL-3.0-or-later), zimg (WTFPL)
LABEL org.opencontainers.image.licenses="MIT AND BSD-2-Clause-Patent AND GPL-3.0-or-later AND WTFPL"
LABEL org.opencontainers.image.source="https://github.com/marcelpoelstra/easyVmafPlus"
LABEL ffmpeg.commit="${FFMPEG_commit}"
LABEL libvmaf.commit="${VMAF_commit}"
LABEL zimg.version="${ZIMG_version}"
LABEL davs2.version="${DAVS2_version}"
LABEL uavs3d.version="${UAVS3D_version}"

ENV LD_LIBRARY_PATH="/usr/local/lib"

RUN \
	export TZ='UTC' && \
	ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
	apt-get update -yqq && \
	apt-get install -y --no-install-recommends \
		dav1d \
		libjxl0.11 && \
	apt-get autoremove -y && \
	apt-get clean -y

COPY --from=build /usr/local /usr/local/

# app setup
WORKDIR /app/easyVmafPlus
COPY pyproject.toml README.md LICENSE ./
COPY easyVmafPlus ./easyVmafPlus
COPY video_samples ./video_samples
RUN pip3 install --no-cache-dir . && rm -rf build easyVmafPlus.egg-info
# Build the matplotlib font cache into the image, so a new container does not build it again
RUN python3 -c "import matplotlib.font_manager"

ENTRYPOINT [ "python3", "-u", "-m", "easyVmafPlus" ]
