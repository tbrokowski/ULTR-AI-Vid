FROM pytorch/pytorch@sha256:dab81780fd94483b67b4b5679cc0024939b08e48540d39476d284cb29002ed69
COPY requirements /opt/paired-depth/requirements
RUN python -m pip install --no-cache-dir -r /opt/paired-depth/requirements/paired-depth-support.lock \
 && python -c 'import torch,torchvision; assert torch.__version__ == "2.8.0+cu126"; assert torchvision.__version__ == "0.23.0+cu126"'
WORKDIR /app
COPY ultrai /app/ultrai
COPY NetworkArchitecture /app/NetworkArchitecture
COPY configs /app/configs
COPY scripts/paired_depth /app/scripts/paired_depth
COPY rcp/paired_depth /app/rcp/paired_depth
ENTRYPOINT ["python", "-m", "ultrai.paired_depth"]
