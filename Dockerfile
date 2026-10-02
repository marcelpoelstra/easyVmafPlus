# Global ARGs, redeclared in the stages that use them
ARG FFMPEG_version=8.1
ARG VMAF_version=3.0.0

FROM python:3.12-slim AS base

FROM base AS build

ARG FFMPEG_version
ARG VMAF_version

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

# install ffmpeg
WORKDIR /tmp/ffmpeg
RUN \
	export LD_LIBRARY_PATH="/usr/local/lib" && \
	export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" && \
	wget https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n${FFMPEG_version}.tar.gz && \
	tar -xzf n${FFMPEG_version}.tar.gz && \
	cd FFmpeg-n${FFMPEG_version} && \
	./configure --enable-libvmaf --enable-version3 --enable-shared --enable-libdav1d && \
	make -j$(nproc) && \
	make install && \
	rm -rf /tmp/ffmpeg

FROM base AS release

ARG FFMPEG_version
ARG VMAF_version

LABEL org.opencontainers.image.title="easyVmafPlus"
LABEL org.opencontainers.image.description="FFmpeg-based VMAF computation with automatic deinterlacing, scaling and sync, and hardware accelerated decoding"
LABEL org.opencontainers.image.licenses="MIT"
LABEL ffmpeg.version="${FFMPEG_version}"
LABEL libvmaf.version="${VMAF_version}"

ENV LD_LIBRARY_PATH="/usr/local/lib"

RUN \
	export TZ='UTC' && \
	ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
	apt-get update -yqq && \
	apt-get install -y --no-install-recommends \
		dav1d && \
	apt-get autoremove -y && \
	apt-get clean -y

COPY --from=build /usr/local /usr/local/

# app setup
WORKDIR /app/easyVmafPlus
COPY pyproject.toml README.md LICENSE ./
COPY easyVmafPlus ./easyVmafPlus
COPY video_samples ./video_samples
RUN pip3 install --no-cache-dir . && rm -rf build easyVmafPlus.egg-info

ENTRYPOINT [ "python3", "-u", "-m", "easyVmafPlus" ]
