#!/usr/bin/env bash
set -euo pipefail
umask 077
python -m venv --system-site-packages /scratch/users/falke/benin-paired-depth-dann/env/runtime
/scratch/users/falke/benin-paired-depth-dann/env/runtime/bin/pip install --disable-pip-version-check -r /scratch/users/falke/benin-paired-depth-dann/code/rcp/paired_depth/support.lock
/scratch/users/falke/benin-paired-depth-dann/env/runtime/bin/python -c 'import torch, torchvision; assert torch.__version__.split("+")[0] == "2.8.0"; assert torchvision.__version__.split("+")[0] == "0.23.0"; assert torch.version.cuda == "12.6"; print(torch.__version__, torchvision.__version__, torch.version.cuda)'
/scratch/users/falke/benin-paired-depth-dann/env/runtime/bin/pip freeze > /scratch/users/falke/benin-paired-depth-dann/artifacts/environment.freeze.txt
/scratch/users/falke/benin-paired-depth-dann/env/runtime/bin/python -m ultrai.paired_depth prepare --videos /benin/datasets/ULTR-AI/LusBeninVideos --output /scratch/users/falke/benin-paired-depth-dann/artifacts/data --workers 8
