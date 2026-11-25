# halo-small-gpu-check

[Halo](https://github.com/whitecircle/halo) ships two Docker images: `hopper` for H100 and H200, and `blackwell` for B200 and B300. Its README has one sentence about smaller cards: "Single-GPU LoRA/QLoRA also runs on Ampere and Ada cards from `halo:blackwell`, but that path is not validated."

This repo measures that path. It runs Halo's two shipped single-GPU recipes for Qwen3-4B for six optimizer steps on five Ampere and Ada card types and on one older card (T4), with the two documented setups (H100 on the hopper image, B200 on the blackwell image) as references. For every run it records whether training runs, how much GPU memory it takes, how fast it is, and whether the training loss matches the H100 run.

All runs use Halo's public 1.3.0rc1 images on [Modal](https://modal.com). The set of 20 runs was run twice. The complete output of every run is in `results/`.

## Summary

1. The QLoRA recipe runs unchanged on all five Ampere and Ada card types: A10 and A10G, L4, L40S, A100 40 GB and A100 80 GB. Peak allocated memory is 7.76 GiB on every one of them (Halo's README says 7.9 GiB) and PyTorch reserves 10.35 to 10.36 GiB, so the recipe fits a 24 GB card.
2. The training loss matches the documented setup. Over the six steps, no card's loss differs from the first-pass H100 run by more than 0.003, while repeating the H100 run itself gives differences up to 0.006. The comparison spans three FlashAttention versions: 2 on the Ampere and Ada cards, 3 on H100 and 4 on B200.
3. The LoRA recipe as shipped does not fit a 24 GB card. On A10, A10G and L4 it stops in the first step with `torch.OutOfMemoryError`. On the larger Ampere and Ada cards it peaks at 25.43 GiB allocated and 36.42 GiB reserved. With `--per_device_train_batch_size=1 --gradient_accumulation_steps=16`, which is the batch shape of the QLoRA recipe, it peaks at 12.75 to 12.76 GiB, runs on both 24 GB card types and gives the same losses. On those cards it is then 7 to 9% faster than the QLoRA recipe.
4. A T4 does not run the recipe as shipped. Halo selects FlashAttention-2 for it, which raises "FlashAttention only supports Ampere GPUs or newer" in the first step. With `--attn_implementation=sdpa` training runs and the losses match those of H100 with the same backend, but at 50 and 73 tokens/s in the two passes (83 and 56 seconds per 4,096-token row).
5. The QLoRA recipe processes about 1,200 tokens/s on L4, 1,600 on A10 and A10G, 4,150 on L40S and 5,000 to 5,400 on A100, against 8,000 to 8,600 on H100. On H100 and B200 this recipe is bound by something other than the card, so its numbers there do not measure those cards (see [Limitations](#limitations)).
6. Halo's efficiency metrics report MFU as 0 on A10, A10G, L4, L40S and T4, because its table of GPU peak FLOPS does not list these cards.

## Results

`uv run python report.py` prints the tables in this section from the files in `results/`. How to read the columns is described [below](#reading-the-tables).

### Cards

| Modal GPU type | Card reported by nvidia-smi | Memory | Image |
|---|---|---:|---|
| T4 | Tesla T4 | 15,360 MiB | blackwell-1.3.0rc1 |
| A10 | NVIDIA A10, NVIDIA A10G | 23,028 MiB | blackwell-1.3.0rc1 |
| L4 | NVIDIA L4 | 23,034 MiB | blackwell-1.3.0rc1 |
| A100-40GB | NVIDIA A100-SXM4-40GB | 40,960 MiB | blackwell-1.3.0rc1 |
| L40S | NVIDIA L40S | 46,068 MiB | blackwell-1.3.0rc1 |
| H100 | NVIDIA H100 80GB HBM3 | 81,559 MiB | hopper-1.3.0rc1 |
| A100-80GB | NVIDIA A100 80GB PCIe, NVIDIA A100-SXM4-80GB | 81,920 MiB | blackwell-1.3.0rc1 |
| B200 | NVIDIA B200 | 183,359 MiB | blackwell-1.3.0rc1 |

T4 is a Turing card, A10, A10G and A100 are Ampere, L4 and L40S are Ada, H100 is Hopper and B200 is Blackwell. Modal's `A10` type returned an A10G in every run of the first pass and an A10 in every run of the second. Its `A100-80GB` type returned the SXM4 variant in three runs and the PCIe variant in one.

### QLoRA recipe as shipped

[`examples/sft/qwen3/qwen3-4b-ultrachat-qlora.yaml`](https://github.com/whitecircle/halo/blob/v1.3.0rc1/examples/sft/qwen3/qwen3-4b-ultrachat-qlora.yaml): 4-bit base model, batch 1, accumulation 16.

| Card | Pass | Outcome | Attention | Tokens/s | Seconds/step (range) | Peak allocated, GiB | Reserved, GiB | Loss difference to H100, pass 1 |
|---|---|---|---|---:|---:|---:|---:|---:|
| Tesla T4 | 1 | failed: RuntimeError: FlashAttention only supports Ampere GPUs or newer | `flash_attention_2` | n/a | n/a | n/a | n/a | n/a |
| Tesla T4 | 2 | failed: RuntimeError: FlashAttention only supports Ampere GPUs or newer | `flash_attention_2` | n/a | n/a | n/a | n/a | n/a |
| NVIDIA A10G | 1 | completed | `flash_attention_2` | 1,562 | 41.73 (41.27 to 42.27) | 7.76 | 10.35 | 0.001 |
| NVIDIA A10 | 2 | completed | `flash_attention_2` | 1,646 | 39.59 (38.98 to 39.91) | 7.76 | 10.35 | 0.001 |
| NVIDIA L4 | 1 | completed | `flash_attention_2` | 1,177 | 55.37 (55.19 to 55.65) | 7.76 | 10.35 | 0.001 |
| NVIDIA L4 | 2 | completed | `flash_attention_2` | 1,230 | 52.99 (52.31 to 53.50) | 7.76 | 10.35 | 0.001 |
| NVIDIA A100-SXM4-40GB | 1 | completed | `flash_attention_2` | 4,975 | 13.10 (12.73 to 13.45) | 7.76 | 10.35 | 0.002 |
| NVIDIA A100-SXM4-40GB | 2 | completed | `flash_attention_2` | 4,972 | 13.11 (12.84 to 13.39) | 7.76 | 10.35 | 0.001 |
| NVIDIA L40S | 1 | completed | `flash_attention_2` | 4,155 | 15.68 (15.38 to 16.02) | 7.76 | 10.36 | 0.001 |
| NVIDIA L40S | 2 | completed | `flash_attention_2` | 4,148 | 15.71 (15.48 to 15.92) | 7.76 | 10.36 | 0.000 |
| NVIDIA H100 80GB HBM3 | 1 | completed | `flash_attention_3` | 8,573 | 7.60 (7.30 to 7.92) | 7.81 | 10.38 | reference |
| NVIDIA H100 80GB HBM3 | 2 | completed | `flash_attention_3` | 7,997 | 8.15 (7.80 to 8.46) | 7.81 | 10.38 | 0.006 |
| NVIDIA A100-SXM4-80GB | 1 | completed | `flash_attention_2` | 5,275 | 12.36 (12.01 to 12.92) | 7.76 | 10.35 | 0.001 |
| NVIDIA A100-SXM4-80GB | 2 | completed | `flash_attention_2` | 5,438 | 11.98 (11.78 to 12.19) | 7.76 | 10.35 | 0.001 |
| NVIDIA B200 | 1 | completed | `flash_attention_4` | 8,212 | 7.94 (6.96 to 9.72) | 7.76 | 10.36 | 0.003 |
| NVIDIA B200 | 2 | completed | `flash_attention_4` | 8,425 | 7.74 (7.08 to 9.15) | 7.76 | 10.36 | 0.003 |
| NVIDIA B200 | trial | completed | `flash_attention_4` | 16,130 | 4.04 (3.69 to 4.32) | 7.76 | 10.36 | 0.002 |

### LoRA recipe as shipped

[`examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml`](https://github.com/whitecircle/halo/blob/v1.3.0rc1/examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml): bf16 base model, batch 4, accumulation 4.

| Card | Pass | Outcome | Attention | Tokens/s | Seconds/step (range) | Peak allocated, GiB | Reserved, GiB | Loss difference to H100, pass 1 |
|---|---|---|---|---:|---:|---:|---:|---:|
| NVIDIA A10G | 1 | failed: torch.OutOfMemoryError: CUDA out of memory | `flash_attention_2` | n/a | n/a | n/a | n/a | n/a |
| NVIDIA A10 | 2 | failed: torch.OutOfMemoryError: CUDA out of memory | `flash_attention_2` | n/a | n/a | n/a | n/a | n/a |
| NVIDIA L4 | 1 | failed: torch.OutOfMemoryError: CUDA out of memory | `flash_attention_2` | n/a | n/a | n/a | n/a | n/a |
| NVIDIA L4 | 2 | failed: torch.OutOfMemoryError: CUDA out of memory | `flash_attention_2` | n/a | n/a | n/a | n/a | n/a |
| NVIDIA A100-SXM4-40GB | 1 | completed | `flash_attention_2` | 5,704 | 11.43 (11.25 to 11.86) | 25.43 | 36.42 | 0.001 |
| NVIDIA A100-SXM4-40GB | 2 | completed | `flash_attention_2` | 5,709 | 11.42 (11.33 to 11.59) | 25.43 | 36.42 | 0.001 |
| NVIDIA L40S | 1 | completed | `flash_attention_2` | 4,005 | 16.27 (16.03 to 16.52) | 25.43 | 36.42 | 0.001 |
| NVIDIA L40S | 2 | completed | `flash_attention_2` | 4,066 | 16.03 (15.94 to 16.08) | 25.43 | 36.42 | 0.002 |
| NVIDIA H100 80GB HBM3 | 1 | completed | `flash_attention_3` | 14,230 | 4.58 (4.43 to 4.89) | 25.49 | 36.47 | reference |
| NVIDIA H100 80GB HBM3 | 2 | completed | `flash_attention_3` | 13,930 | 4.68 (4.52 to 5.02) | 25.49 | 36.47 | 0.001 |
| NVIDIA A100-SXM4-80GB | 1 | completed | `flash_attention_2` | 6,104 | 10.68 (10.53 to 11.01) | 25.43 | 36.42 | 0.001 |
| NVIDIA A100 80GB PCIe | 2 | completed | `flash_attention_2` | 5,559 | 11.72 (11.63 to 11.81) | 25.43 | 36.42 | 0.001 |
| NVIDIA B200 | 1 | completed | `flash_attention_4` | 26,750 | 2.44 (2.43 to 2.45) | 25.43 | 36.42 | 0.001 |
| NVIDIA B200 | 2 | completed | `flash_attention_4` | 26,540 | 2.46 (2.45 to 2.46) | 25.43 | 36.42 | 0.001 |
| NVIDIA B200 | trial | completed | `flash_attention_4` | 26,630 | 2.45 (2.44 to 2.47) | 25.43 | 36.42 | 0.001 |

The four failed runs end with the same message: PyTorch has 15.36 to 15.37 GiB allocated and cannot allocate another 9.27 GiB on a card whose total capacity it reports as 22.08 GiB.

### LoRA recipe at batch 1

The same recipe with `--per_device_train_batch_size=1 --gradient_accumulation_steps=16`: still 16 rows per optimizer step, one row at a time.

| Card | Pass | Outcome | Attention | Tokens/s | Seconds/step (range) | Peak allocated, GiB | Reserved, GiB | Loss difference to H100, pass 1 |
|---|---|---|---|---:|---:|---:|---:|---:|
| NVIDIA A10G | 1 | completed | `flash_attention_2` | 1,672 | 38.98 (38.88 to 39.17) | 12.76 | 15.25 | 0.001 |
| NVIDIA A10 | 2 | completed | `flash_attention_2` | 1,786 | 36.50 (36.21 to 36.61) | 12.76 | 15.25 | 0.002 |
| NVIDIA L4 | 1 | completed | `flash_attention_2` | 1,258 | 51.82 (51.63 to 52.23) | 12.75 | 15.32 | 0.003 |
| NVIDIA L4 | 2 | completed | `flash_attention_2` | 1,336 | 48.77 (48.48 to 49.00) | 12.75 | 15.32 | 0.001 |
| NVIDIA H100 80GB HBM3 | 1 | completed | `flash_attention_3` | 12,610 | 5.17 (4.90 to 5.38) | 12.81 | 15.32 | reference |
| NVIDIA H100 80GB HBM3 | 2 | completed | `flash_attention_3` | 12,660 | 5.15 (4.94 to 5.34) | 12.81 | 15.32 | 0.001 |

### QLoRA recipe on a T4 with SDPA

The QLoRA recipe with `--attn_implementation=sdpa --gradient_accumulation_steps=1`. One optimizer step is one row here, because a 16-row step would take the T4 15 to 22 minutes. The H100 run is the reference for the losses, not a benchmark: its four measured steps take between 0.67 and 2.14 seconds.

| Card | Pass | Outcome | Attention | Tokens/s | Seconds/step (range) | Peak allocated, GiB | Reserved, GiB | Loss difference to H100, pass 1 |
|---|---|---|---|---:|---:|---:|---:|---:|
| Tesla T4 | 1 | completed | `sdpa` | 50 | 82.54 (81.50 to 83.21) | 12.32 | 13.64 | 0.002 |
| Tesla T4 | 2 | completed | `sdpa` | 73 | 56.03 (55.90 to 56.09) | 12.32 | 13.64 | 0.002 |
| NVIDIA H100 80GB HBM3 | 1 | completed | `sdpa` | 2,969 | 1.38 (0.67 to 2.08) | 7.55 | 8.14 | reference |
| NVIDIA H100 80GB HBM3 | 2 | completed | `sdpa` | 2,908 | 1.41 (0.69 to 2.14) | 7.55 | 8.14 | 0.003 |

On the T4 the recipe peaks at 12.32 GiB allocated, on a card with 15,360 MiB. The same backend on H100 peaks at 7.55 GiB, and FlashAttention-2 on the Ampere and Ada cards at 7.76 GiB.

### Reading the tables

- One row is one run. `Pass` is the directory under `results/` that holds it. `trial` is a launch of the two B200 runs alone, made before the two passes.
- `Card` is the name `nvidia-smi` printed inside the container.
- `Outcome` is `completed` when `halo launch` exited with code 0 after six optimizer steps. Otherwise it is the first sentence of the exception that ended the run.
- `Attention` is the backend Halo logged when training started.
- `Tokens/s` and `Seconds/step` are Halo's `avg_tokens_per_second` and `avg_step_time_seconds`. Halo leaves the first two steps out as warm-up, so both are averages over steps 3 to 6. The range is the fastest and the slowest of those four steps. One step is 16 rows of up to 4,096 tokens (65,234 tokens on average), except in the T4 table.
- `Peak allocated` is Halo's `training_peak_allocated_gb`: the highest `torch.cuda.max_memory_allocated()` of steps 3 to 6, in GiB. `Reserved` is `torch.cuda.memory_reserved()` at the end of the last step. Neither counts memory outside PyTorch's allocator, such as the CUDA context, so `nvidia-smi` shows more.
- The last column is the largest absolute difference, over the six steps, between the run's logged training loss and that of the first-pass H100 run in the same table. Halo logs the loss with four significant digits, so 0.001 is one unit in the last digit. The H100 rows of the second pass show how much the reference setup differs from itself. Runs are not bit-reproducible: in the QLoRA table the two H100 runs log the same loss in the first two steps and differ by 0.006 in the sixth.

## What was run

- **Images.** `public.ecr.aws/whitecircle/halo:hopper-1.3.0rc1` for H100 and `public.ecr.aws/whitecircle/halo:blackwell-1.3.0rc1` for every other card. Halo's README assigns hopper to H100 and H200, and blackwell to B200, B300 and the Ampere and Ada cards; it does not mention older cards such as the T4. On 2026-11-25 the blackwell and hopper tags pointed to `sha256:32fe426f2842439947c4d571542019283992dc05f80c430d79012749677a2b33` and `sha256:2f429b0c33aa1839bf3c6d72ae60d30abb90e882463c6ff4ba128b72cb202927`.
- **Recipes.** The two files linked above, read from the Halo checkout inside the image. Both train rank-64 adapters (132,120,576 parameters) on all attention and MLP projections of `Qwen/Qwen3-4B-Instruct-2507` on UltraChat 200k, with rows packed to 4,096 tokens. They differ in the base model (4-bit NF4 or bf16) and in how the 16 rows of an optimizer step are batched. With batch 4, Halo packs the four rows into one sequence of about 16k tokens.
- **Arguments added to every run.** `--max_steps=6 --save_strategy=no --eval_strategy=no --report_to=none --logging_steps=1 --enable_efficiency_metrics=true --dataset_ratio=0.02 --output_dir=/tmp/run`. Nothing is saved or evaluated, and 2% of the dataset is tokenized (4,156 conversations; the training split packs into 1,204 rows). Every result file records its full command.
- **Model and data.** The recipes name both without a revision. The runs used `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554` and `HuggingFaceH4/ultrachat_200k` at `8049631c405ae6576f93f445c6b8166f76f5505a`; every result file lists the snapshots in its cache.
- **Platform.** Each run is one Modal container with one GPU, started by `modal_app.py`. The image's entrypoint is cleared and `halo launch` runs from `/workspace`. Modal starts the container itself, so the `docker run` flags that Halo's installation guide calls required (`--ipc=host`, `--shm-size=128g` and two ulimits) are not set. CPU and memory are Modal's defaults: 0.125 cores and 128 MiB guaranteed, more when the host has it free. The NVIDIA driver is 620.70.06 in 40 runs and 610.57.04 in two.
- **Dates.** All runs are from 2026-11-25.

## Reproduce

Needs [uv](https://docs.astral.sh/uv/) and a Modal account.

```bash
uv sync --locked
uv run modal setup                               # once
uv run modal run modal_app.py                    # all 20 runs, written to results/1
uv run modal run modal_app.py --out results/2    # the second pass
uv run python report.py                          # the tables above
```

`uv run modal run modal_app.py --only qlora_a10,qlora_h100` runs a subset; the names are the keys of `RUNS` in `modal_app.py`.

The first run of a pass downloads the model and the dataset to a Modal volume. The other runs then start at most eight at a time, because Modal's Starter plan allows ten GPUs at once. The first pass was made before that limit was added and started all of them together. The two passes took 29 and 28 minutes and cost $2.99 and $2.94 according to `modal billing report`.

The logs in `results/` are the output of `halo launch` with em dashes and en dashes replaced by hyphens.

## Limitations

- Six steps on 2% of the data show that training starts, fits in memory and computes the same losses. They do not show that a full run converges, and saving, evaluating, resuming and merging adapters were not exercised.
- Throughput is measured over four steps on shared cloud hosts, with no CPU reserved beyond Modal's default. Between the two passes, tokens/s changed by at most 9% for every run except the T4 one. A third sample exists for B200 only and shows that this understates the spread: the QLoRA recipe on B200 measured 8,212 and 8,425 tokens/s in the two passes and 16,130 in the trial launch. With one row per batch, the step on the fast cards is bound by something other than the card: in both passes B200 and H100 are within 6% of each other on the QLoRA recipe, while B200 is 1.9 times faster on the batch-4 LoRA recipe. The H100 and B200 rows of the batch-1 tables therefore say little about those cards.
- Consumer cards were not tested, because Modal does not offer them. A10, A10G and L4 share their architectures and 24 GB of memory with the RTX 3090 and RTX 4090, but they are different hardware.
- Only one model family, one training method and one sequence length were tried: dense Qwen3-4B, SFT, 4,096 tokens. MoE models, other trainers and multi-GPU runs are out of scope.
- The images are the 1.3.0rc1 release candidate. Later releases may behave differently.

## Notes for Halo

Checked against `v1.3.0rc1` and against `main` at `3a21b4d`, where the two recipes and the code below are the same.

- The LoRA recipe carries no memory note, while the QLoRA recipe says it fits a 24 GB GPU. As shipped the LoRA recipe needs more than 24 GB; with batch 1 and accumulation 16 it peaks at 12.8 GiB.
- `GPU_PEAK_FLOPS` in `halo/hardware.py` has entries for H100, H200, B200, GB200, B300, A100 and A6000. On other cards the efficiency callback logs "Could not detect the GPU model" and reports MFU as 0. `grep -l "Could not detect the GPU model" results/*/*.log` lists the affected runs.
- `preferred_flash_impl` in `halo/models/attention/selection.py` returns `flash_attention_2` for every card that is neither Hopper nor Blackwell, including cards older than Ampere, where FlashAttention-2 fails in the first forward pass. Halo's attention documentation lists FlashAttention-2 as "Ampere+".

## Licenses

The code in this repo is under the Apache License 2.0 (see `LICENSE`). Halo is under the [Halo License](https://github.com/whitecircle/halo/blob/main/LICENSE). `Qwen/Qwen3-4B-Instruct-2507` is under Apache 2.0 and `HuggingFaceH4/ultrachat_200k` under MIT. The repo contains no model weights and no dataset rows.
