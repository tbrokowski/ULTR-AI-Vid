# ARM64-ONLY Dockerfile
# This image is intended to run exclusively on ARM64 (aarch64) GPU clusters
# (e.g., NVIDIA Grace / Grace Hopper, or other ARM64 servers with NVIDIA GPUs).
# It will FAIL EARLY if built or run on a non-ARM architecture.
FROM --platform=linux/arm64 nvcr.io/nvidia/pytorch:25.08-py3

# Install small set of useful system packages (optional)
# Add/remove as needed. ffmpeg is often useful for video.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git curl ca-certificates \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && echo "Base packages installed on $(uname -m)"

# Workdir inside the container (NGC images often use /workspace)
WORKDIR /workspace

# Install Python dependencies first for better Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY . ./
RUN pip install --no-cache-dir ./  # Install local package in editable mode

# Optional extras for LLM/Axolotl-style stacks. Disabled by default.
# Enable at build time with: --build-arg INSTALL_LLM_EXTRAS=1 --build-arg TORCH_CUDA_ARCH_LIST=9.0

ARG INSTALL_LLM_EXTRAS=0
ARG TORCH_CUDA_ARCH_LIST=9.0
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=24

RUN if [ "$INSTALL_LLM_EXTRAS" = "1" ]; then \
    set -eux; \
    arch=$(uname -m); \
    if [ "$arch" != "aarch64" ] && [ "$arch" != "arm64" ]; then \
        echo "INSTALL_LLM_EXTRAS is only supported on ARM64 in this Dockerfile" >&2; exit 1; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends build-essential ninja-build cmake && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    echo "ARM64 detected: skipping flash-attn & xformers (unreliable/slow on arm64)" && \
    pip install --no-cache-dir trl==0.9.6 deepspeed==0.14.4 && \
    pip install --no-cache-dir git+https://github.com/lm-sys/FastChat.git@27a05b04a35510afb1d767ae7e5990cbd278f8fe && \
    pip install --no-cache-dir git+https://github.com/huggingface/transformers.git@026a173a64372e9602a16523b8fae9de4b0ff428; \
  fi

# Enforce ARM64 early (fail fast if mis-built under emulation wrong arch)
RUN set -eux; arch=$(uname -m); if [ "$arch" != "aarch64" ] && [ "$arch" != "arm64" ]; then echo "ERROR: This image is ARM64-only (detected $arch)" >&2; exit 1; fi

# Install VS Code CLI for remote tunnel access (ARM64 specific)
RUN mkdir -p /workspace/code
WORKDIR /workspace/code
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64' --output vscode_cli.tar.gz
RUN tar -xf vscode_cli.tar.gz
RUN mv ./code /usr/bin
RUN rm -rf /workspace/code

# Copy your code into the image (good for cluster runs)
# When developing locally, you can override this with a bind mount (-v $PWD:/workspace)
COPY . .

# Default entrypoint: call Python; you pass your script + args at runtime
# Example: docker run ... image-name train_clip_drl_mil_Final-2.py --cfg configs/Finalruns/tb_drl_mil_Final_fold0.yaml
ENTRYPOINT ["python", "-u"]
