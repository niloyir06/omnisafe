"""Semantic run analysis script.

Reads progress.csv files for baseline and semantic variants, computes:
- Learning curves with smoothing
- Cost vs return tradeoff scatter
- Shaping signal diagnostics (ratio, correlation with returns & costs)
- Margin distribution & clamp fraction trend
- Beta schedule overlay
- Safety efficiency: return per unit cost
Outputs plots to an output directory and prints summary statistics.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SMOOTH_WINDOW = 5

METRIC_RET = 'Metrics/EpRet'
METRIC_COST = 'Metrics/EpCost'
STEP_COL = 'TotalEnvSteps'
SHAPING = 'Semantics/Shaping'
RAW_REWARD = 'Semantics/RawReward'
BETA = 'Semantics/Beta'
RAW_MARGIN = 'Semantics/RawMargin'
NORM_MARGIN = 'Semantics/NormMargin'
CLAMP_FRAC = 'Semantics/ClampFrac'
CAPTURE_COUNT = 'Semantics/CaptureCount'


def _smooth(series: pd.Series, window: int = SMOOTH_WINDOW) -> pd.Series:
    if window <= 1 or len(series) < window:
        return series
    return series.rolling(window, min_periods=1).mean()


def load_progress_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop duplicated columns (some csv may repeat due to logging config) by keeping first occurrence
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def shaping_ratio(df: pd.DataFrame) -> float:
    if SHAPING not in df or RAW_REWARD not in df:
        return float('nan')
    denom = df[RAW_REWARD].abs().replace(0, np.nan).mean()
    if denom is None or math.isnan(denom) or denom == 0:
        return float('nan')
    return df[SHAPING].abs().mean() / denom


def correlation_with_future(df: pd.DataFrame, col: str, horizon: int = 5) -> float:
    if col not in df or METRIC_RET not in df:
        return float('nan')
    # shift returns backward so margin_t aligns with ret_{t+h}
    ret_shift = df[METRIC_RET].shift(-horizon)
    valid = ~ret_shift.isna()
    if valid.sum() < 10:
        return float('nan')
    return float(np.corrcoef(df.loc[valid, col], ret_shift[valid])[0, 1])


def summarize_run(df: pd.DataFrame) -> Dict[str, float]:
    return {
        'final_return': df[METRIC_RET].iloc[-1],
        'final_cost': df[METRIC_COST].iloc[-1],
        'avg_return': df[METRIC_RET].mean(),
        'avg_cost': df[METRIC_COST].mean(),
        'best_return': df[METRIC_RET].max(),
        'min_cost': df[METRIC_COST].min(),
        'shaping_ratio': shaping_ratio(df),
        'beta_mean': df[BETA].mean() if BETA in df else float('nan'),
        'beta_final': df[BETA].iloc[-1] if BETA in df else float('nan'),
        'raw_margin_mean': df[RAW_MARGIN].mean() if RAW_MARGIN in df else float('nan'),
        'raw_margin_std': df[RAW_MARGIN].std() if RAW_MARGIN in df else float('nan'),
        'norm_margin_std': df[NORM_MARGIN].std() if NORM_MARGIN in df else float('nan'),
        'clamp_frac_last': df[CLAMP_FRAC].iloc[-1] if CLAMP_FRAC in df else float('nan'),
        'margin_ret_corr_h5': correlation_with_future(df, RAW_MARGIN, 5),
        'margin_ret_corr_h20': correlation_with_future(df, RAW_MARGIN, 20),
    }


def plot_curves(runs: Dict[str, pd.DataFrame], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # Return & Cost curves
    for metric, name in [(METRIC_RET, 'return'), (METRIC_COST, 'cost')]:
        plt.figure(figsize=(8,4))
        for label, df in runs.items():
            if metric not in df: continue
            plt.plot(df[STEP_COL], _smooth(df[metric]), label=label)
        plt.xlabel('Env Steps')
        plt.ylabel(metric)
        plt.title(f'{metric} (smoothed)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f'{name}_curve.png')
        plt.close()

    # Cost vs Return scatter (final points)
    plt.figure(figsize=(5,5))
    for label, df in runs.items():
        plt.scatter(df[METRIC_COST].iloc[-1], df[METRIC_RET].iloc[-1], label=label)
    plt.xlabel('Final Cost')
    plt.ylabel('Final Return')
    plt.title('Cost vs Return (final)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'cost_return_final.png')
    plt.close()

    # Beta schedule
    if any(BETA in df for df in runs.values()):
        plt.figure(figsize=(8,4))
        for label, df in runs.items():
            if BETA in df:
                plt.plot(df[STEP_COL], df[BETA], label=label)
        plt.xlabel('Env Steps')
        plt.ylabel('Beta')
        plt.title('Shaping Coefficient Schedule')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / 'beta_schedule.png')
        plt.close()

    # Margin distribution over time (semantic runs)
    if any(RAW_MARGIN in df for df in runs.values()):
        plt.figure(figsize=(8,4))
        for label, df in runs.items():
            if RAW_MARGIN in df:
                plt.plot(df[STEP_COL], df[RAW_MARGIN], alpha=0.4, label=f'{label}-raw')
        plt.xlabel('Env Steps')
        plt.ylabel('Raw Margin')
        plt.title('Raw Margin Trajectory')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / 'raw_margin_traj.png')
        plt.close()

    # Clamp fraction trend
    if any(CLAMP_FRAC in df for df in runs.values()):
        plt.figure(figsize=(8,4))
        for label, df in runs.items():
            if CLAMP_FRAC in df:
                plt.plot(df[STEP_COL], df[CLAMP_FRAC], label=label)
        plt.xlabel('Env Steps')
        plt.ylabel('ClampFrac')
        plt.title('Clamp Fraction Trend')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / 'clamp_fraction.png')
        plt.close()

    # Shaping magnitude vs return curve
    if any(SHAPING in df for df in runs.values()):
        plt.figure(figsize=(8,4))
        for label, df in runs.items():
            if SHAPING in df:
                plt.plot(df[STEP_COL], _smooth(df[SHAPING].abs()), label=label)
        plt.xlabel('Env Steps')
        plt.ylabel('|Shaping| (smoothed)')
        plt.title('Shaping Magnitude')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / 'shaping_magnitude.png')
        plt.close()


def aggregate_runs(run_paths: List[Path]) -> pd.DataFrame:
    dfs = [load_progress_csv(p) for p in run_paths if p.exists()]
    if not dfs:
        return pd.DataFrame()
    # Align on step column via merge-asof like interpolation
    base = dfs[0][[STEP_COL]].copy()
    base = base.drop_duplicates()
    for df in dfs:
        base = base.merge(df[[STEP_COL, METRIC_RET, METRIC_COST]], on=STEP_COL, how='outer', suffixes=(False, False))
    base.sort_values(STEP_COL, inplace=True)
    # Compute mean & std across columns with common name patterns
    ret_cols = [c for c in base.columns if c.endswith(METRIC_RET.split('/')[-1]) or c == METRIC_RET]
    cost_cols = [c for c in base.columns if c.endswith(METRIC_COST.split('/')[-1]) or c == METRIC_COST]
    # Fallback simple: just return last df
    return dfs[0]


def find_progress_files(root: Path, match: str) -> List[Path]:
    paths = []
    for p in root.rglob('progress.csv'):
        if match in str(p):
            paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', type=str, nargs='+', required=True, help='Root dirs containing runs')
    parser.add_argument('--labels', type=str, nargs='+', required=True, help='Labels matching roots order')
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--semantic-filter', type=str, default='PPOLagSem', help='Substring to detect semantic runs')
    args = parser.parse_args()

    runs: Dict[str, pd.DataFrame] = {}
    for root, label in zip(args.roots, args.labels):
        root_path = Path(root)
        # For each seed under root, load and concatenate (simple vertical concat) after adding a seed id
        progress_files = list(root_path.rglob('progress.csv'))
        if not progress_files:
            print(f'No progress.csv under {root}')
            continue
        # For now take last (latest) run per label
        latest = max(progress_files, key=lambda p: p.stat().st_mtime)
        df = load_progress_csv(latest)
        runs[label] = df

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Plot curves & diagnostics
    plot_curves(runs, outdir)

    # Summaries
    summary = {label: summarize_run(df) for label, df in runs.items()}
    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('Summary:')
    for label, stats in summary.items():
        print(label, stats)

    # If both baseline and semantic present, compute relative deltas
    if len(runs) >= 2:
        labels = list(runs.keys())
        base = runs[labels[0]]
        for other_label in labels[1:]:
            other = runs[other_label]
            if METRIC_RET in base and METRIC_RET in other:
                ret_delta = other[METRIC_RET].iloc[-1] - base[METRIC_RET].iloc[-1]
                cost_delta = other[METRIC_COST].iloc[-1] - base[METRIC_COST].iloc[-1]
                print(f'Final Δ Ret ({other_label} - {labels[0]}): {ret_delta:.3f}, Δ Cost: {cost_delta:.3f}')


if __name__ == '__main__':
    main()
