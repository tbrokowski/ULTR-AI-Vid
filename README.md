# ULTR-AI-Vid

Ultrasound video classification with HMV-MIL and a CLIP image encoder.

## Benin paired-depth training

This workflow trains on Benin ultrasound recordings acquired at exactly 5 and
15 cm. It provides supervised baselines, paired consistency, scan- and patient-level
domain adversarial losses, class conditioning and synthetic acquisition styles.
Both depths use available Benin labels.

- [Method and implementation](docs/paired_depth/REPORT.md)
- [Installation and reproduction](docs/paired_depth/REPRODUCE.md)
- [Method handover PDF](docs/paired_depth/REPORT.pdf)
- [Optional EPFL RCP execution](rcp/paired_depth/README.md)

Use `python -m ultrai.paired_depth` for preparation, training and evaluation, and
`python -m scripts.paired_depth.experiments` for portable configuration generation.
Run commands from the repository root. The guide describes the required private
videos, labels, metadata and patient partitions, and execution on a Linux NVIDIA
GPU machine. Existing training entrypoints remain available.

The handover documents the implementation and protocol. Experiment outputs belong
in a separate study directory.
