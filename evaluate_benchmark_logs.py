#!/usr/bin/env python
"""
DREMA Benchmark Log Parser & Comparative Analyzer.
Analyzes log files generated during benchmark runs and computes comparative statistics:
- End-to-end Loop Latency & Framerate (Hz)
- Step 1 (TSDF), Step 2 (VDC: Prune, Render, Init), Step 3 (SE3 Tracking)
- Table Churn (Evicted & Added Primitives per frame)
- Active Gaussians memory stability
- Object tracking precision (delta displacement)
"""

import sys
import os
import re
import glob
import numpy as np


def parse_log_file(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        text = f.read()

    data = {
        'filepath': filepath,
        'name': os.path.basename(filepath),
        'total_latency': [],
        'fps': [],
        'step1_tsdf': [],
        'step2_vdc': [],
        'step2_prune': [],
        'step2_render': [],
        'step2_init': [],
        'pruned': [],
        'evicted': [],
        'added': [],
        'active_3dgs': [],
        'step3_se3': [],
        'w_mean': [],
        'stray_in_static': [],
        'vram_mb': []
    }

    # Regex patterns
    re_total = re.compile(r"Total Step Latency:\s*([\d\.]+)ms\s*\(([\d\.]+)\s*Hz\)(?:\s*\|\s*GPU VRAM:\s*([\d\.]+)MB)?")
    re_step1 = re.compile(r"Step 1 \(TSDF Ingest\):\s*([\d\.]+)ms")
    re_step2 = re.compile(r"Step 2 \(VDC Mapping\):\s*([\d\.]+)ms\s*\[Prune:\s*([\d\.]+)ms\s*\|\s*3DGS Render:\s*([\d\.]+)ms\s*\|\s*Init:\s*([\d\.]+)ms.*?\]\s*\|\s*Pruned:\s*(\d+)\s*\|\s*Evicted:\s*(\d+)\s*\|\s*Added:\s*(\d+)\s*\|\s*Active 3DGS:\s*([\d\,]+)")
    re_step3 = re.compile(r"Step 3 \(RecurGS Tracking\):\s*([\d\.]+)ms")
    re_tsdf = re.compile(r"W_mean:\s*([\d\.]+)")
    re_stray = re.compile(r"(\d+)\s*stray in static background")

    for line in text.splitlines():
        m_tot = re_total.search(line)
        if m_tot:
            data['total_latency'].append(float(m_tot.group(1)))
            data['fps'].append(float(m_tot.group(2)))
            if m_tot.group(3):
                data['vram_mb'].append(float(m_tot.group(3)))

        m_s1 = re_step1.search(line)
        if m_s1:
            data['step1_tsdf'].append(float(m_s1.group(1)))

        m_s2 = re_step2.search(line)
        if m_s2:
            data['step2_vdc'].append(float(m_s2.group(1)))
            data['step2_prune'].append(float(m_s2.group(2)))
            data['step2_render'].append(float(m_s2.group(3)))
            data['step2_init'].append(float(m_s2.group(4)))
            data['pruned'].append(int(m_s2.group(5)))
            data['evicted'].append(int(m_s2.group(6)))
            data['added'].append(int(m_s2.group(7)))
            data['active_3dgs'].append(int(m_s2.group(8).replace(',', '')))

        m_s3 = re_step3.search(line)
        if m_s3:
            data['step3_se3'].append(float(m_s3.group(1)))

        m_w = re_tsdf.search(line)
        if m_w:
            data['w_mean'].append(float(m_w.group(1)))

        m_stray = re_stray.search(line)
        if m_stray:
            data['stray_in_static'].append(int(m_stray.group(1)))

    return data


def format_stats(arr):
    if len(arr) == 0:
        return "N/A"
    return f"{np.mean(arr):.1f} ± {np.std(arr):.1f} (med: {np.median(arr):.1f})"


def compare_logs(log_paths: list):
    parsed = [parse_log_file(p) for p in log_paths]
    parsed = [p for p in parsed if p and len(p['total_latency']) > 0]

    if not parsed:
        print("No valid benchmark log data found.")
        return

    print("\n" + "=" * 115)
    print(f"{'DREMA PERCEPTION BENCHMARK COMPARATIVE REPORT':^115}")
    print("=" * 115)

    headers = ["Metric", *[p['name'][:22] for p in parsed]]
    fmt_row = "{:<32} " + " | ".join(["{:<24}"] * len(parsed))
    print(fmt_row.format(*headers))
    print("-" * 115)

    metrics = [
        ("Frames Analyzed", lambda p: f"{len(p['total_latency'])} frames"),
        ("Loop Framerate (Hz)", lambda p: f"{np.mean(p['fps']):.1f} Hz (max: {np.max(p['fps']):.1f})"),
        ("Total Loop Latency (ms)", lambda p: format_stats(p['total_latency'])),
        ("Step 1: TSDF Ingest (ms)", lambda p: format_stats(p['step1_tsdf'])),
        ("Step 2: VDC Mapping (ms)", lambda p: format_stats(p['step2_vdc'])),
        ("  ├─ Raycast Prune (ms)", lambda p: format_stats(p['step2_prune'])),
        ("  ├─ 3DGS Render (ms)", lambda p: format_stats(p['step2_render'])),
        ("  └─ VDC Init (ms)", lambda p: format_stats(p['step2_init'])),
        ("Table Churn (Evicted/frame)", lambda p: f"{np.mean(p['evicted']):.0f} / frame"),
        ("Table Churn (Added/frame)", lambda p: f"{np.mean(p['added']):.0f} / frame"),
        ("Step 3: SE(3) Tracking (ms)", lambda p: format_stats(p['step3_se3'])),
        ("Active Gaussians (count)", lambda p: f"{int(np.mean(p['active_3dgs'])):,} ± {np.std(p['active_3dgs']):.0f}"),
        ("Stray in Static (Orphans)", lambda p: f"{np.sum(p['stray_in_static'])} total (0 is perfect)"),
        ("GPU VRAM (MB)", lambda p: f"{np.mean(p['vram_mb']):.0f} MB" if len(p['vram_mb']) > 0 else "N/A")
    ]

    for label, fn in metrics:
        vals = [fn(p) for p in parsed]
        print(fmt_row.format(label, *vals))

    print("=" * 115 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = sorted(glob.glob("logs/benchmarks/*.log"))

    compare_logs(paths)
