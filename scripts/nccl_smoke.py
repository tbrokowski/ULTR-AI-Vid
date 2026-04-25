#!/usr/bin/env python3
"""Minimal distributed smoke test for the cluster/container runtime."""

import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"host={socket.gethostname()}")
        print(f"torch={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"cuda_device_count={torch.cuda.device_count()}")
        for key in [
            "CUDA_VISIBLE_DEVICES",
            "SLURM_JOB_GPUS",
            "SLURM_GPUS",
            "NCCL_NET",
            "NCCL_NET_PLUGIN",
            "NCCL_SOCKET_IFNAME",
            "NCCL_IB_DISABLE",
            "NCCL_P2P_DISABLE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ]:
            print(f"{key}={os.environ.get(key, '<unset>')}")

    if world_size > 1:
        init_kwargs = {
            "backend": "nccl",
            "init_method": "env://",
            "rank": rank,
            "world_size": world_size,
            "timeout": timedelta(minutes=5),
        }
        if torch.cuda.is_available():
            init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        try:
            dist.init_process_group(**init_kwargs)
        except TypeError:
            init_kwargs.pop("device_id", None)
            dist.init_process_group(**init_kwargs)

        value = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        print(f"rank={rank} local_rank={local_rank} all_reduce={value.item()}")
        dist.destroy_process_group()
    else:
        print("world_size=1; no distributed initialization attempted")


if __name__ == "__main__":
    main()
