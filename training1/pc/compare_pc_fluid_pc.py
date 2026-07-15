#!/usr/bin/env python3
"""
Run and plot MNIST test accuracy for the existing training scripts.

Models compared:
    1. Standard backprop:       training1/backprop/mnist.py
    2. Pure predictive coding:  training1/pc/pc_mnist_one_hidden.py
    3. Fluid predictive coding: training1/pc/fluid_pc_mnist_paper_like.py
    4. Navier-Stokes PC:        training1/pc/pc_mnist_navier_stokes.py

This script does not reimplement the models. It runs the existing files,
parses their epoch outputs, saves a CSV, and creates one test-accuracy plot.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


BACKPROP_EPOCH_RE = re.compile(
    r"Epoch \[(?P<epoch>\d+)/(?P<total>\d+)\].*?"
    r"Test Loss: (?P<loss>[0-9.]+).*?"
    r"Test Acc: (?P<acc>[0-9.]+)%"
)

PC_EPOCH_RE = re.compile(
    r"Epoch \[(?P<epoch>\d+)/(?P<total>\d+)\].*?"
    r"Test E: (?P<loss>[0-9.]+).*?"
    r"Test Acc: (?P<acc>[0-9.]+)%"
)

FLUID_EPOCH_RE = re.compile(
    r"Epoch (?P<epoch>\d+)/(?P<total>\d+).*?"
    r"Test loss (?P<loss>[0-9.]+).*?"
    r"Test acc (?P<acc>[0-9.]+)%"
)

NS_EPOCH_RE = re.compile(
    r"epoch=(?P<epoch>\d+)\s+"
    r"test_acc_no_inner_loop=(?P<fast_acc>[0-9.]+)%\s+"
    r"test_acc_with_pc_inference=(?P<acc>[0-9.]+)%"
)


def run_and_log(
    command: List[str],
    log_path: Path,
    cwd: Path,
    stop_regex: Optional[re.Pattern[str]] = None,
    stop_epoch: Optional[int] = None,
) -> None:
    """Run a command, stream output to the terminal, and save the same output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 88)
    print("Running:")
    print(" ".join(command))
    print(f"Log: {log_path}")
    print("=" * 88)

    stopped_after_final_epoch = False

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout is None:
            raise RuntimeError("Could not capture process output.")

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

            if stop_regex is not None and stop_epoch is not None:
                match = stop_regex.search(line)
                if match and int(match.group("epoch")) >= stop_epoch:
                    stopped_after_final_epoch = True
                    process.terminate()
                    break

        try:
            return_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait()

    if return_code != 0 and not stopped_after_final_epoch:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")


def regex_for(kind: str) -> re.Pattern[str]:
    if kind == "backprop":
        return BACKPROP_EPOCH_RE
    if kind == "pc":
        return PC_EPOCH_RE
    if kind == "fluid":
        return FLUID_EPOCH_RE
    if kind == "navier":
        return NS_EPOCH_RE
    raise ValueError(f"Unknown metric kind: {kind}")


def parse_epoch_metrics(log_path: Path, kind: str) -> List[Dict[str, Optional[float]]]:
    """Extract per-epoch test accuracy from a training log."""
    rows: List[Dict[str, Optional[float]]] = []
    regex = regex_for(kind)

    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = regex.search(line)
        if not match:
            continue

        groups = match.groupdict()
        rows.append(
            {
                "epoch": int(groups["epoch"]),
                "test_loss": float(groups["loss"]) if groups.get("loss") else None,
                "test_acc": float(groups["acc"]),
                "fast_acc": float(groups["fast_acc"]) if groups.get("fast_acc") else None,
            }
        )

    if not rows:
        raise ValueError(
            f"No epoch metrics found in {log_path}. "
            "Check that the script finished and printed epoch summaries."
        )

    return rows


def value_at(rows: List[Dict[str, Optional[float]]], epoch: int, key: str) -> Optional[float]:
    for row in rows:
        if row["epoch"] == epoch:
            return row[key]
    return None


def fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.4f}"


def save_csv(metrics: Dict[str, List[Dict[str, Optional[float]]]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = sorted({int(row["epoch"]) for rows in metrics.values() for row in rows})

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "backprop_test_acc",
                "pure_pc_test_acc",
                "fluid_pc_test_acc",
                "navier_stokes_pc_test_acc",
                "navier_stokes_no_inner_loop_acc",
                "backprop_test_loss",
                "pure_pc_test_loss",
                "fluid_pc_test_loss",
            ],
        )
        writer.writeheader()

        for epoch in epochs:
            writer.writerow(
                {
                    "epoch": epoch,
                    "backprop_test_acc": fmt(value_at(metrics["backprop"], epoch, "test_acc")),
                    "pure_pc_test_acc": fmt(value_at(metrics["pc"], epoch, "test_acc")),
                    "fluid_pc_test_acc": fmt(value_at(metrics["fluid"], epoch, "test_acc")),
                    "navier_stokes_pc_test_acc": fmt(value_at(metrics["navier"], epoch, "test_acc")),
                    "navier_stokes_no_inner_loop_acc": fmt(value_at(metrics["navier"], epoch, "fast_acc")),
                    "backprop_test_loss": fmt(value_at(metrics["backprop"], epoch, "test_loss")),
                    "pure_pc_test_loss": fmt(value_at(metrics["pc"], epoch, "test_loss")),
                    "fluid_pc_test_loss": fmt(value_at(metrics["fluid"], epoch, "test_loss")),
                }
            )


def plot_metrics(metrics: Dict[str, List[Dict[str, Optional[float]]]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    styles = [
        ("backprop", "Standard Backprop", "tab:green", "-", "D", "white", 5),
        ("pc", "Pure PC", "tab:blue", "-", "o", "white", 4),
        ("fluid", "Fluid PC", "tab:orange", "--", "s", "tab:orange", 3),
        ("navier", "Navier-Stokes PC", "tab:red", "-.", "^", "white", 2),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    final_lines = []
    all_epochs = set()

    for key, label, color, linestyle, marker, markerface, zorder in styles:
        rows = metrics[key]
        epochs = [int(row["epoch"]) for row in rows]
        acc = [float(row["test_acc"]) for row in rows]
        all_epochs.update(epochs)

        ax.plot(
            epochs,
            acc,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markerfacecolor=markerface,
            markeredgecolor=color,
            markeredgewidth=2,
            markersize=6,
            linewidth=2.2,
            label=label,
            zorder=zorder,
        )
        final_lines.append(f"{label}: {acc[-1]:.2f}%")

    ax.set_title("MNIST Test Accuracy Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower right")

    if all_epochs and max(all_epochs) <= 25:
        ax.set_xticks(sorted(all_epochs))

    ax.text(
        0.02,
        0.98,
        "\n".join(final_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.92},
    )

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_backprop_command(args: argparse.Namespace, script: Path, checkpoint_dir: Path) -> List[str]:
    return [
        sys.executable,
        "-B",
        str(script),
        "--data_dir",
        args.data_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.backprop_lr),
        "--train_subset",
        str(args.train_subset),
        "--test_subset",
        str(args.test_subset),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--checkpoint_dir",
        str(checkpoint_dir),
    ]


def build_pc_command(args: argparse.Namespace, script: Path, checkpoint_dir: Path) -> List[str]:
    return [
        sys.executable,
        "-B",
        str(script),
        "--data_dir",
        args.data_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--hidden_dim",
        str(args.pc_hidden_dim),
        "--inference_steps",
        str(args.pc_inference_steps),
        "--eval_inference_steps",
        str(args.pc_eval_inference_steps),
        "--hidden_lr",
        str(args.pc_hidden_lr),
        "--weight_lr",
        str(args.pc_weight_lr),
        "--recon_weight",
        str(args.pc_recon_weight),
        "--hidden_weight",
        str(args.pc_hidden_weight),
        "--weight_update",
        args.pc_weight_update,
        "--train_subset",
        str(args.train_subset),
        "--test_subset",
        str(args.test_subset),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--log_interval",
        str(args.pc_log_interval),
        "--checkpoint_dir",
        str(checkpoint_dir),
    ]


def build_fluid_command(args: argparse.Namespace, script: Path) -> List[str]:
    command = [
        sys.executable,
        "-B",
        str(script),
        "--data_dir",
        args.data_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.fluid_lr),
        "--seed",
        str(args.seed),
        "--num_workers",
        str(args.num_workers),
        "--train_subset",
        str(args.train_subset),
        "--test_subset",
        str(args.test_subset),
        "--inner_steps",
        str(args.fluid_inner_steps),
        "--reaction_lr",
        str(args.fluid_reaction_lr),
        "--dt",
        str(args.fluid_dt),
        "--target_cfl",
        str(args.fluid_target_cfl),
        "--diffusion",
        str(args.fluid_diffusion),
        "--pc_pred_weight",
        str(args.fluid_pc_pred_weight),
        "--pc_recon_weight",
        str(args.fluid_pc_recon_weight),
        "--pc_entropy_weight",
        str(args.fluid_pc_entropy_weight),
        "--outer_recon_weight",
        str(args.fluid_outer_recon_weight),
        "--grad_clip",
        str(args.fluid_grad_clip),
        "--print_every",
        str(args.fluid_print_every),
    ]

    if args.fluid_no_projection:
        command.append("--no_projection")
    if args.fluid_no_differentiable_inner:
        command.append("--no_differentiable_inner")

    return command


def build_navier_command(args: argparse.Namespace, script: Path) -> List[str]:
    return [
        sys.executable,
        "-B",
        str(script),
        "--data_dir",
        args.data_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--train_subset",
        str(args.train_subset),
        "--test_subset",
        str(args.test_subset),
        "--lr",
        str(args.navier_lr),
        "--seed",
        str(args.seed),
        "--num_workers",
        str(args.num_workers),
        "--inference_steps",
        str(args.navier_inference_steps),
        "--hidden_lr",
        str(args.navier_hidden_lr),
        "--ns_weight",
        str(args.navier_ns_weight),
        "--prior_weight",
        str(args.navier_prior_weight),
        "--recon_weight",
        str(args.navier_recon_weight),
        "--class_weight",
        str(args.navier_class_weight),
        "--nu",
        str(args.navier_nu),
        "--momentum_weight",
        str(args.navier_momentum_weight),
        "--divergence_weight",
        str(args.navier_divergence_weight),
        "--amort_weight",
        str(args.navier_amort_weight),
        "--ns_init_weight",
        str(args.navier_ns_init_weight),
    ]


def print_summary(metrics: Dict[str, List[Dict[str, Optional[float]]]]) -> None:
    labels = {
        "backprop": "Standard Backprop",
        "pc": "Pure PC",
        "fluid": "Fluid PC",
        "navier": "Navier-Stokes PC",
    }

    print("\n" + "=" * 88)
    print("Comparison summary")
    print("=" * 88)
    for key, label in labels.items():
        final_acc = metrics[key][-1]["test_acc"]
        print(f"{label:22s}: {final_acc:.2f}%")
    print("=" * 88)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--train_subset", type=int, default=10000)
    p.add_argument("--test_subset", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--output_dir", type=str, default="training1/pc/comparison_runs")
    p.add_argument("--plot_name", type=str, default="mnist_test_accuracy_comparison.png")
    p.add_argument("--csv_name", type=str, default="mnist_test_accuracy_comparison.csv")
    p.add_argument("--plot_only", action="store_true", help="Parse existing logs without rerunning training.")

    p.add_argument("--backprop_lr", type=float, default=1e-3)

    p.add_argument("--pc_hidden_dim", type=int, default=128)
    p.add_argument("--pc_inference_steps", type=int, default=20)
    p.add_argument("--pc_eval_inference_steps", type=int, default=25)
    p.add_argument("--pc_hidden_lr", type=float, default=0.8)
    p.add_argument("--pc_weight_lr", type=float, default=0.02)
    p.add_argument("--pc_recon_weight", type=float, default=1.0)
    p.add_argument("--pc_hidden_weight", type=float, default=2.0)
    p.add_argument("--pc_weight_update", type=str, default="local", choices=["local", "adam"])
    p.add_argument("--pc_log_interval", type=int, default=50)

    # Match the pure PC train-time inner-loop budget by default.
    p.add_argument("--fluid_inner_steps", type=int, default=20)
    p.add_argument("--fluid_lr", type=float, default=1e-3)
    p.add_argument("--fluid_reaction_lr", type=float, default=0.25)
    p.add_argument("--fluid_dt", type=float, default=0.5)
    p.add_argument("--fluid_target_cfl", type=float, default=0.35)
    p.add_argument("--fluid_diffusion", type=float, default=0.002)
    p.add_argument("--fluid_pc_pred_weight", type=float, default=0.20)
    p.add_argument("--fluid_pc_recon_weight", type=float, default=0.20)
    p.add_argument("--fluid_pc_entropy_weight", type=float, default=0.00)
    p.add_argument("--fluid_outer_recon_weight", type=float, default=0.10)
    p.add_argument("--fluid_grad_clip", type=float, default=1.0)
    p.add_argument("--fluid_print_every", type=int, default=50)
    p.add_argument("--fluid_no_projection", action="store_true")
    p.add_argument("--fluid_no_differentiable_inner", action="store_true")

    p.add_argument("--navier_lr", type=float, default=1e-3)
    p.add_argument("--navier_inference_steps", type=int, default=20)
    p.add_argument("--navier_hidden_lr", type=float, default=0.25)
    p.add_argument("--navier_ns_weight", type=float, default=0.02)
    p.add_argument("--navier_prior_weight", type=float, default=0.10)
    p.add_argument("--navier_recon_weight", type=float, default=1.0)
    p.add_argument("--navier_class_weight", type=float, default=1.0)
    p.add_argument("--navier_nu", type=float, default=0.01)
    p.add_argument("--navier_momentum_weight", type=float, default=1.0)
    p.add_argument("--navier_divergence_weight", type=float, default=5.0)
    p.add_argument("--navier_amort_weight", type=float, default=0.50)
    p.add_argument("--navier_ns_init_weight", type=float, default=0.005)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path.cwd()
    scripts = {
        "backprop": repo_root / "training1/backprop/mnist.py",
        "pc": repo_root / "training1/pc/pc_mnist_one_hidden.py",
        "fluid": repo_root / "training1/pc/fluid_pc_mnist_paper_like.py",
        "navier": repo_root / "training1/pc/pc_mnist_navier_stokes.py",
    }

    for script in scripts.values():
        if not script.exists():
            raise FileNotFoundError(script)

    output_dir = Path(args.output_dir)
    logs = {
        "backprop": output_dir / "backprop_mnist_run.log",
        "pc": output_dir / "pure_pc_run.log",
        "fluid": output_dir / "fluid_pc_run.log",
        "navier": output_dir / "navier_stokes_pc_run.log",
    }
    csv_path = output_dir / args.csv_name
    plot_path = output_dir / args.plot_name

    print("Comparison uses existing model files, not scratch reimplementations.")
    print(
        f"Common settings: epochs={args.epochs}, train_subset={args.train_subset}, "
        f"test_subset={args.test_subset}, batch_size={args.batch_size}, seed={args.seed}"
    )

    if not args.plot_only:
        run_and_log(
            build_backprop_command(args, scripts["backprop"], output_dir / "backprop_checkpoints"),
            logs["backprop"],
            repo_root,
            BACKPROP_EPOCH_RE,
            args.epochs,
        )
        run_and_log(
            build_pc_command(args, scripts["pc"], output_dir / "pc_checkpoints"),
            logs["pc"],
            repo_root,
            PC_EPOCH_RE,
            args.epochs,
        )
        run_and_log(
            build_fluid_command(args, scripts["fluid"]),
            logs["fluid"],
            repo_root,
            FLUID_EPOCH_RE,
            args.epochs,
        )
        run_and_log(
            build_navier_command(args, scripts["navier"]),
            logs["navier"],
            repo_root,
            NS_EPOCH_RE,
            args.epochs,
        )
    else:
        print("plot_only=True, parsing existing logs:")
        for key, log_path in logs.items():
            print(f"  {key}: {log_path}")

    metrics = {
        "backprop": parse_epoch_metrics(logs["backprop"], "backprop"),
        "pc": parse_epoch_metrics(logs["pc"], "pc"),
        "fluid": parse_epoch_metrics(logs["fluid"], "fluid"),
        "navier": parse_epoch_metrics(logs["navier"], "navier"),
    }

    save_csv(metrics, csv_path)
    plot_metrics(metrics, plot_path)
    print_summary(metrics)

    print(f"Saved metrics: {csv_path}")
    print(f"Saved plot   : {plot_path}")


if __name__ == "__main__":
    main()
