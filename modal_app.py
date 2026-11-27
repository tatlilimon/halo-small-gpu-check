"""Run Halo's two shipped single-GPU Qwen3-4B recipes for a few steps on several Modal GPU types.

    modal run modal_app.py                                 # every run in RUNS, into results/1
    modal run modal_app.py --out results/2                 # the same again, as a second pass
    modal run modal_app.py --only qlora_a10,qlora_h100     # some of the runs

For each run, <out>/<name>.json records what ran and how it ended, and
<out>/<name>.log is the complete output of `halo launch`.
"""

import itertools
import json
import subprocess
from pathlib import Path

import modal

VERSION = "1.3.0rc1"
RECIPES = {
    "lora": "examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml",
    "qlora": "examples/sft/qwen3/qwen3-4b-ultrachat-qlora.yaml",
}
STEPS = 6
# The QLoRA recipe's batch shape for the LoRA recipe, which ships with batch 4 and
# accumulation 4: the same 16 rows per optimizer step, one row at a time.
BATCH_1 = "--per_device_train_batch_size=1 --gradient_accumulation_steps=16"
# For the T4, which FlashAttention does not support. With PyTorch's attention it needs about
# a minute for one 4,096-token row, so this pair of runs takes one row per optimizer step.
SDPA_ONE_ROW = "--attn_implementation=sdpa --gradient_accumulation_steps=1"
# name: (Modal GPU type, recipe, arguments added to the recipe)
RUNS = {
    "qlora_a10": ("A10", "qlora", ""),
    "qlora_l4": ("L4", "qlora", ""),
    "qlora_l40s": ("L40S", "qlora", ""),
    "qlora_a100-40gb": ("A100-40GB", "qlora", ""),
    "qlora_a100-80gb": ("A100-80GB", "qlora", ""),
    "qlora_h100": ("H100", "qlora", ""),
    "qlora_b200": ("B200", "qlora", ""),
    "qlora_t4": ("T4", "qlora", ""),
    "lora_a10": ("A10", "lora", ""),
    "lora_l4": ("L4", "lora", ""),
    "lora_l40s": ("L40S", "lora", ""),
    "lora_a100-40gb": ("A100-40GB", "lora", ""),
    "lora_a100-80gb": ("A100-80GB", "lora", ""),
    "lora_h100": ("H100", "lora", ""),
    "lora_b200": ("B200", "lora", ""),
    "lora_batch1_a10": ("A10", "lora", BATCH_1),
    "lora_batch1_l4": ("L4", "lora", BATCH_1),
    "lora_batch1_h100": ("H100", "lora", BATCH_1),
    "qlora_sdpa_t4": ("T4", "qlora", SDPA_ONE_ROW),
    "qlora_sdpa_h100": ("H100", "qlora", SDPA_ONE_ROW),
}

app = modal.App("halo-small-gpu-check")
volume = modal.Volume.from_name("halo-small-gpu-check", create_if_missing=True)


def halo_image(flavor: str) -> modal.Image:
    # Clearing the entrypoint only keeps the base image's banner out of the logs.
    return (
        modal.Image.from_registry(f"public.ecr.aws/whitecircle/halo:{flavor}-{VERSION}")
        .entrypoint([])
        .env({"HF_HOME": "/vol/hf"})
    )


def run_recipe(recipe: str, arguments: str) -> dict:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()  # fmt: skip
    command = [
        "halo", "launch", "sft", RECIPES[recipe], "--output_dir=/tmp/run", f"--max_steps={STEPS}",
        "--save_strategy=no", "--eval_strategy=no", "--report_to=none", "--logging_steps=1",
        "--enable_efficiency_metrics=true",
        # The recipes train on all of UltraChat 200k; a slice is enough for a few steps.
        "--dataset_ratio=0.02", *arguments.split(),
    ]  # fmt: skip
    # A failing recipe is a result, not an error of this script.
    done = subprocess.run(command, cwd="/workspace", capture_output=True, text=True, check=False)
    volume.commit()
    # The recipes name the model and the dataset without a revision; this records the ones that were used.
    snapshots = sorted(str(p.relative_to("/vol/hf/hub")) for p in Path("/vol/hf/hub").glob("*/snapshots/*"))
    return {
        "gpu": gpu,
        "command": " ".join(command),
        "exit_code": done.returncode,
        "snapshots": snapshots,
        "log": done.stdout + done.stderr,
    }


@app.function(image=halo_image("blackwell"), gpu="A10", volumes={"/vol": volume}, timeout=40 * 60)
def on_blackwell_image(recipe: str, arguments: str) -> dict:
    return run_recipe(recipe, arguments)


@app.function(image=halo_image("hopper"), gpu="H100", volumes={"/vol": volume}, timeout=40 * 60)
def on_hopper_image(recipe: str, arguments: str) -> dict:
    return run_recipe(recipe, arguments)


def image_flavor(gpu: str) -> str:
    # Halo's README assigns the hopper image to H100 and the blackwell image to every other card here.
    return "hopper" if gpu == "H100" else "blackwell"


def start(name: str) -> modal.FunctionCall:
    gpu, recipe, arguments = RUNS[name]
    function = {"blackwell": on_blackwell_image, "hopper": on_hopper_image}[image_flavor(gpu)]
    return function.with_options(gpu=gpu).spawn(recipe, arguments)


def finish(name: str, call: modal.FunctionCall, out: Path) -> None:
    gpu, recipe, arguments = RUNS[name]
    try:
        result = call.get()
    except modal.exception.Error as error:
        # One lost container should not discard the other runs; rerun this one with --only.
        print(f"{name}: no result, {type(error).__name__}: {error}")
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.log").write_text(result.pop("log"), encoding="utf-8")
    record = {
        "requested_gpu": gpu,
        "image": f"{image_flavor(gpu)}-{VERSION}",
        "recipe": recipe,
        "arguments": arguments,
        **result,
    }
    (out / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{name}: {result['gpu']}, exit code {result['exit_code']}")


@app.local_entrypoint()
def main(only: str = "", out: str = "results/1", parallel: int = 8):
    first, *rest = only.split(",") if only else list(RUNS)
    # The first run downloads the model and the dataset to the volume; the others read them from there.
    finish(first, start(first), Path(out))
    # Modal's Starter plan allows ten GPUs at a time.
    for names in itertools.batched(rest, parallel):
        calls = {name: start(name) for name in names}
        for name, call in calls.items():
            finish(name, call, Path(out))
