# Minimal Dockerfile to run this repo on GPU with PyTorch base image
# Base image requested by user (includes CUDA, cuDNN, PyTorch, Python)
FROM nvcr.io/nvidia/pytorch:25.08-py3

# Install small set of useful system packages (optional)
# Add/remove as needed. ffmpeg is often useful for video.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

# Workdir inside the container (NGC images often use /workspace)
WORKDIR /workspace

# Install Python dependencies first for better Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Optional extras for LLM/Axolotl-style stacks. Disabled by default.
# Enable at build time with: --build-arg INSTALL_LLM_EXTRAS=1 --build-arg TORCH_CUDA_ARCH_LIST=9.0

ARG INSTALL_LLM_EXTRAS=0
ARG TORCH_CUDA_ARCH_LIST=9.0
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=24

RUN if [ "$INSTALL_LLM_EXTRAS" = "1" ]; then \
    apt-get update && apt-get install -y --no-install-recommends build-essential ninja-build cmake && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
        pip install --no-cache-dir flash-attn==2.6.2 --no-build-isolation && \
        pip install --no-cache-dir trl==0.9.6 deepspeed==0.14.4 && \
        pip install --no-cache-dir git+https://github.com/lm-sys/FastChat.git@27a05b04a35510afb1d767ae7e5990cbd278f8fe && \
        pip install --no-cache-dir -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers && \
        pip install --no-cache-dir git+https://github.com/huggingface/transformers.git@026a173a64372e9602a16523b8fae9de4b0ff428 ; \
    fi

# Copy your code into the image (good for cluster runs)
# When developing locally, you can override this with a bind mount (-v $PWD:/workspace)
COPY . .

# Default entrypoint: call Python; you pass your script + args at runtime
# Example: docker run ... image-name train_clip_drl_mil_Final-2.py --cfg configs/Finalruns/tb_drl_mil_Final_fold0.yaml
ENTRYPOINT ["python", "-u"]
