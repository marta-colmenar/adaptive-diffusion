#!/usr/bin/env python3
"""Run ordered-prefix MC sampling for the old Bern diffusion checkpoint."""

import argparse
import subprocess
import sys


DEFAULT_CHECKPOINT = "/storage/homefs/mc21s017/denoiser_ablation_diffusion_bern_3vf.pth"
DEFAULT_OUTPUT_NAME = "sequential_prefix_mc_bern_3vf_old_diffusion_ordered"
SAMPLER_SCRIPT = "/storage/homefs/mc21s017/sample_sequential_ordered_prefix_mc.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Launch ordered-prefix test-time sampling on the old Bern diffusion "
            "checkpoint."
        )
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--dataset", type=str, default="bern", choices=["bern", "wa"])
    parser.add_argument("--n_vf", type=int, default=3, choices=[3, 4])
    parser.add_argument("--num_mc", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--mc_chunk", type=int, default=10)
    parser.add_argument("--use_diffusion_model", action="store_true",
                        help="Launch sampler using DiffusionModel (append seen-prefix as extra VF)")
    parser.add_argument("--seen_scale", type=float, default=0.5,
                        help="Scale applied to seen-prefix values forwarded to the sampler")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional args forwarded to the sampler")
    args = parser.parse_args()

    cmd = [
        args.python,
        SAMPLER_SCRIPT,
        "--dataset",
        args.dataset,
        "--n_vf",
        str(args.n_vf),
        "--checkpoint",
        args.checkpoint,
        "--output_name",
        args.output_name,
        "--num_mc",
        str(args.num_mc),
        "--batch_size",
        str(args.batch_size),
        "--mc_chunk",
        str(args.mc_chunk),
    ]

    if args.extra_args:
        forwarded = args.extra_args[1:] if args.extra_args[0] == "--" else args.extra_args
        cmd.extend(forwarded)
    # Forward model choice and seen_scale by default
    if args.use_diffusion_model:
        cmd.append("--use_diffusion_model")
    if args.seen_scale is not None:
        cmd.extend(["--seen_scale", str(args.seen_scale)])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()