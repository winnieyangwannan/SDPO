#!/usr/bin/env python3
"""
Extract and plot training metrics from VERL/SDPO log files.

Usage:
    python scripts/plot_training_metrics.py /path/to/logfile.log
    python scripts/plot_training_metrics.py /path/to/logfile.log --output plot.png
    python scripts/plot_training_metrics.py /path/to/log1.log /path/to/log2.log --labels "SDPO" "GRPO"

    # Plot and save to file
    python scripts/plot_training_metrics.py /checkpoint/agentic-models/$USER/output/SDPO/5386899.log -o /checkpoint/agentic-models/$USER/output/SDPO/SDPO/5386899_train_val.png

"""

import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_log_line(line: str) -> Optional[Dict[str, float]]:
    """Parse a single log line and extract metrics."""
    # Match lines like: step:1 - metric1:value1 - metric2:value2 ...
    if "step:" not in line:
        return None
    
    metrics = {}
    # Split by " - " to get key:value pairs
    parts = line.split(" - ")
    
    for part in parts:
        # Handle key:value format
        if ":" in part:
            # Find the last colon which separates key from value
            # Handle cases like "timing_s/agent_loop/generate_sequences/min:6.917"
            match = re.match(r'^([^:]+(?::[^:]+)*):([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)$', part.strip())
            if match:
                key = match.group(1).strip()
                try:
                    value = float(match.group(2))
                    metrics[key] = value
                except ValueError:
                    continue
    
    return metrics if metrics else None


def extract_metrics_from_log(log_path: str) -> Dict[str, List[Tuple[int, float]]]:
    """Extract all metrics from a log file, indexed by training step."""
    metrics_by_name = defaultdict(list)
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed and 'training/global_step' in parsed:
                step = int(parsed['training/global_step'])
                for key, value in parsed.items():
                    metrics_by_name[key].append((step, value))
    
    # Sort by step and remove duplicates (keep first occurrence)
    for key in metrics_by_name:
        seen_steps = set()
        unique_data = []
        for step, value in sorted(metrics_by_name[key]):
            if step not in seen_steps:
                seen_steps.add(step)
                unique_data.append((step, value))
        metrics_by_name[key] = unique_data
    
    return dict(metrics_by_name)


def get_experiment_name(log_path: str) -> str:
    """Extract experiment name from log file."""
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("Experiment:"):
                return line.split(":", 1)[1].strip()
    return Path(log_path).stem


def plot_metrics(
    log_files: List[str],
    labels: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    metrics_to_plot: Optional[List[str]] = None,
):
    """Plot training metrics from one or more log files."""
    
    if metrics_to_plot is None:
        metrics_to_plot = [
            'critic/score/mean',
            'actor/pg_loss',
            'val-aux/livecodebench/score/mean@16',
            'val-aux/livecodebench/acc/best@16/mean',  # pass@16
            'val-aux/livecodebench/acc/maj@8/mean',
            'val-aux/livecodebench/acc/worst@16/mean',
        ]
    
    # Extract metrics from all log files
    all_data = []
    for i, log_path in enumerate(log_files):
        data = extract_metrics_from_log(log_path)
        label = labels[i] if labels and i < len(labels) else get_experiment_name(log_path)
        all_data.append((label, data))
    
    # Filter to metrics that exist
    available_metrics = set()
    for _, data in all_data:
        available_metrics.update(data.keys())
    
    metrics_to_plot = [m for m in metrics_to_plot if m in available_metrics]
    
    if not metrics_to_plot:
        print("No matching metrics found. Available metrics:")
        for m in sorted(available_metrics)[:20]:
            print(f"  - {m}")
        return
    
    # Create subplots
    n_metrics = len(metrics_to_plot)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    axes = np.array(axes).flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
    
    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        for (label, data), color in zip(all_data, colors):
            if metric_name in data:
                steps, values = zip(*data[metric_name])
                ax.plot(steps, values, label=label, color=color, alpha=0.8)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel(metric_name.split('/')[-1])
        ax.set_title(metric_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(metrics_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        output_dir = Path(output_path).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {output_dir}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def print_summary(log_path: str):
    """Print a summary of metrics from a log file."""
    data = extract_metrics_from_log(log_path)
    exp_name = get_experiment_name(log_path)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")
    
    key_metrics = ['critic/score/mean', 'actor/pg_loss', 'training/global_step']
    
    for metric in key_metrics:
        if metric in data:
            steps, values = zip(*data[metric])
            print(f"\n{metric}:")
            print(f"  Steps: {min(steps)} -> {max(steps)}")
            print(f"  Values: {values[0]:.4f} -> {values[-1]:.4f}")
            if metric == 'critic/score/mean':
                print(f"  Max: {max(values):.4f} at step {steps[values.index(max(values))]}")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from VERL log files')
    parser.add_argument('log_files', nargs='+', help='Path to log file(s)')
    parser.add_argument('--labels', nargs='+', help='Labels for each log file')
    parser.add_argument('--output', '-o', help='Output path for the plot (default: show)')
    parser.add_argument('--metrics', nargs='+', help='Specific metrics to plot')
    parser.add_argument('--summary', action='store_true', help='Print summary instead of plotting')
    parser.add_argument('--list-metrics', action='store_true', help='List available metrics')
    
    args = parser.parse_args()
    
    if args.list_metrics:
        data = extract_metrics_from_log(args.log_files[0])
        print("Available metrics:")
        for m in sorted(data.keys()):
            print(f"  - {m}")
        return
    
    if args.summary:
        for log_path in args.log_files:
            print_summary(log_path)
        return
    
    plot_metrics(
        args.log_files,
        labels=args.labels,
        output_path=args.output,
        metrics_to_plot=args.metrics,
    )


if __name__ == '__main__':
    main()
