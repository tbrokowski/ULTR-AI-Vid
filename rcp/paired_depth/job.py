"""Render a single Run:AI job with read-only Benin and isolated writable scratch."""
import argparse
import json

IMAGE = "pytorch/pytorch@sha256:dab81780fd94483b67b4b5679cc0024939b08e48540d39476d284cb29002ed69"


def job(name, command, gpu=1, hours=4, pool="a100-40g"):
    if gpu not in (0, 1) or hours <= 0:
        raise ValueError("Single-GPU jobs only, with a positive hard deadline")
    root = "/scratch/users/falke/benin-paired-depth-dann"
    value = lambda x: {"value": x}
    return {"apiVersion": "run.ai/v2alpha1", "kind": "TrainingWorkload",
            "metadata": {"name": name, "namespace": "runai-light-falke", "labels": {"project": "light-falke", "study": "benin-paired-depth"}},
            "spec": {"usage": "Submit", "active": value(True), "image": value(IMAGE),
                     "command": value(__import__("shlex").join(["bash", "-c", "cd " + root + "/code && exec " + __import__("shlex").join(["timeout", "--signal=TERM", "--kill-after=120", str(int(hours * 3600 - 120)), *command])])),
                     "gpu": value(str(gpu)), "cpu": value("8"), "cpuLimit": value("8"),
                     "memory": value("48G"), "memoryLimit": value("48G"), "backoffLimit": value(0),
                     "runAsUser": value(True), "runAsUid": value(315954), "runAsGid": value(30133),
                     "supplementalGroups": value("84257"), "nodePools": value(pool), "largeShm": value(True),
                     "workingDir": value("/tmp"), "terminationGracePeriodSeconds": value(120),
                     "environment": {"items": {"HF_HOME": value(root + "/artifacts/hf-cache"),
                                                "PYTHONUNBUFFERED": value("1"), "OMP_NUM_THREADS": value("4"),
                                                "MPLCONFIGDIR": value(root + "/artifacts/matplotlib")}},
                     "pvcs": {"items": {"a-study": value({"claimName": "light-scratch", "existingPvc": True, "path": "/scratch", "readOnly": False}),
                                         "z-benin": value({"claimName": "light-scratch", "existingPvc": True, "path": "/benin", "readOnly": True})}}}}



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--hours", type=float, default=4)
    p.add_argument("--pool", default="a100-40g")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    print(json.dumps(job(a.name, command, a.gpu, a.hours, a.pool), indent=2))
