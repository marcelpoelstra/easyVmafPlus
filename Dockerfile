# Global ARGs, redeclared in the stages that use them
ARG FFMPEG_version=9.0.2
ARG VMAF_version=3.2.1
ARG DAVS2_version=1.7
ARG UAVS3D_version=1.2

FROM python:3.12-slim AS base

FROM base AS build

ARG FFMPEG_version
ARG VMAF_version
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
		pkg-config \
		make \
		nasm \
		xxd \
		yasm -y && \
	apt-get autoremove -y && \
	apt-get clean -y && \
	pip3 install --user meson

# install libvmaf
WORKDIR /tmp/vmaf
RUN \
	export PATH="${HOME}/.local/bin:${PATH}" && \
	wget https://github.com/Netflix/vmaf/archive/v${VMAF_version}.tar.gz && \
	tar -xzf v${VMAF_version}.tar.gz && \
	cd vmaf-${VMAF_version}/libvmaf/ && \
	meson build --buildtype release -Dbuilt_in_models=true --libdir=lib && \
	ninja -vC build && \
	ninja -vC build test && \
	ninja -vC build install && \
	mkdir -p /usr/local/share/model && \
	cp -R ../model/* /usr/local/share/model && \
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

# install ffmpeg; libdavs2 is GPL, so the build needs --enable-gpl
WORKDIR /tmp/ffmpeg
RUN \
	export LD_LIBRARY_PATH="/usr/local/lib" && \
	export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" && \
	wget https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n${FFMPEG_version}.tar.gz && \
	tar -xzf n${FFMPEG_version}.tar.gz && \
	cd FFmpeg-n${FFMPEG_version} && \
	./configure --enable-libvmaf --enable-version3 --enable-gpl --enable-shared --enable-libdav1d --enable-libjxl --enable-libdavs2 --enable-libuavs3d && \
	make -j$(nproc) && \
	make install && \
	rm -rf /tmp/ffmpeg

FROM base AS release

ARG FFMPEG_version
ARG VMAF_version
ARG DAVS2_version
ARG UAVS3D_version

LABEL org.opencontainers.image.title="easyVmafPlus"
LABEL org.opencontainers.image.description="FFmpeg-based VMAF computation with automatic deinterlacing, scaling and sync, and hardware accelerated decoding"
LABEL org.opencontainers.image.licenses="MIT AND BSD-2-Clause-Patent AND GPL-3.0-or-later"
LABEL ffmpeg.version="${FFMPEG_version}"
LABEL libvmaf.version="${VMAF_version}"
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
