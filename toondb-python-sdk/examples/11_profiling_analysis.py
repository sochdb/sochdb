#!/usr/bin/env python3
"""
ToonDB HNSW Profiling Analysis and Visualization

This script analyzes the profiling output from end-to-end profiling and provides:
1. Detailed breakdown of time spent in each phase
2. Bottleneck identification
3. Performance recommendations
4. Visual charts (if matplotlib available)

Usage:
    python 11_profiling_analysis.py [--python-profile PATH] [--rust-profile PATH]
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class OperationStats:
    """Statistics for a single profiled operation."""
    name: str
    count: int
    total_ms: float
    mean_us: float
    min_us: float
    max_us: float
    item_count: int = 0
    per_item_us: float = 0.0
    
    @classmethod
    def from_dict(cls, name: str, data: dict) -> 'OperationStats':
        return cls(
            name=name,
            count=data.get('count', 1),
            total_ms=data.get('total_ms', 0),
            mean_us=data.get('mean_us', 0),
            min_us=data.get('min_us', 0),
            max_us=data.get('max_us', 0),
            item_count=data.get('item_count', 0),
            per_item_us=data.get('per_item_us', 0),
        )


@dataclass
class ProfilingAnalysis:
    """Complete profiling analysis results."""
    rust_profile: Dict[str, OperationStats]
    python_profile: Dict[str, dict]
    total_elapsed_ms: float
    num_vectors: int
    dimension: int
    
    @property
    def throughput(self) -> float:
        """Vectors per second."""
        return (self.num_vectors / self.total_elapsed_ms) * 1000
    
    @property
    def latency_per_vector_us(self) -> float:
        """Microseconds per vector."""
        return (self.total_elapsed_ms * 1000) / self.num_vectors


def load_rust_profile(path: str) -> Tuple[Dict[str, OperationStats], float]:
    """Load Rust-side profiling data."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    ops = {}
    for name, stats in data.get('operations', {}).items():
        ops[name] = OperationStats.from_dict(name, stats)
    
    return ops, data.get('total_elapsed_ms', 0)


def load_python_profile(path: str) -> Tuple[Dict[str, dict], dict]:
    """Load Python-side profiling data."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    python_layer = data.get('python_layer', {})
    timings = python_layer.get('timings', {})
    config = data.get('config', {})
    summary = data.get('summary', {})
    
    return timings, {**config, **summary}


def analyze_bottlenecks(rust_ops: Dict[str, OperationStats], total_ms: float) -> List[Tuple[str, float, str]]:
    """
    Identify bottlenecks and return recommendations.
    
    Returns list of (operation, percentage, recommendation)
    """
    bottlenecks = []
    
    for name, stats in rust_ops.items():
        pct = (stats.total_ms / total_ms) * 100 if total_ms > 0 else 0
        
        if pct < 1:
            continue
            
        # Generate recommendations based on the operation
        if 'neighbor_select' in name and pct > 50:
            rec = (
                "CRITICAL: Neighbor selection is the main bottleneck.\n"
                "  Recommendations:\n"
                "  1. Consider reducing ef_construction (currently ~200)\n"
                "  2. Use quantized vectors (F16/BF16) to reduce memory bandwidth\n"
                "  3. The RNG heuristic does O(selected×candidates) distance calcs\n"
                "  4. Consider pre-computing neighbor distances in parallel"
            )
        elif 'add_connections' in name and pct > 20:
            rec = (
                "Connection building is significant.\n"
                "  Recommendations:\n"
                "  1. Lock contention may be an issue - check concurrent writes\n"
                "  2. Consider batch connection updates instead of per-neighbor"
            )
        elif 'search_layer' in name and pct > 15:
            rec = (
                "Layer search is significant.\n"
                "  Recommendations:\n"
                "  1. Ensure SIMD is being used for distance calculations\n"
                "  2. Consider reducing ef_construction for faster search\n"
                "  3. Check cache locality of vector data"
            )
        elif 'quantize' in name and pct > 5:
            rec = (
                "Quantization is taking time.\n"
                "  Recommendations:\n"
                "  1. Already parallelized - check CPU utilization\n"
                "  2. Consider pre-quantized input vectors"
            )
        else:
            rec = ""
        
        bottlenecks.append((name, pct, rec))
    
    # Sort by percentage descending
    bottlenecks.sort(key=lambda x: -x[1])
    return bottlenecks


def print_detailed_report(
    rust_ops: Dict[str, OperationStats],
    python_timings: Dict[str, dict],
    meta: dict,
    total_ms: float
):
    """Print a detailed profiling report."""
    
    num_vectors = meta.get('num_vectors', 1000)
    dimension = meta.get('dimension', 768)
    
    print("=" * 80)
    print("ToonDB HNSW End-to-End Profiling Analysis")
    print("=" * 80)
    print()
    
    # Configuration
    print("CONFIGURATION")
    print("-" * 40)
    print(f"  Vectors:          {num_vectors:,}")
    print(f"  Dimension:        {dimension}")
    print(f"  EF Construction:  {meta.get('ef_construction', 'N/A')}")
    print(f"  Max Connections:  {meta.get('max_connections', 'N/A')}")
    print()
    
    # Performance Summary
    throughput = (num_vectors / total_ms) * 1000 if total_ms > 0 else 0
    latency = total_ms / num_vectors * 1000 if num_vectors > 0 else 0
    
    print("PERFORMANCE SUMMARY")
    print("-" * 40)
    print(f"  Total Time:       {total_ms:,.2f} ms ({total_ms/1000:.2f} s)")
    print(f"  Throughput:       {throughput:,.0f} vectors/sec")
    print(f"  Latency:          {latency:,.2f} µs/vector")
    print()
    
    # Time Breakdown - Rust Operations
    print("RUST-SIDE BREAKDOWN (sorted by time)")
    print("-" * 80)
    print(f"{'Operation':<40} {'Total ms':>12} {'%':>8} {'Per-item µs':>14}")
    print("-" * 80)
    
    sorted_ops = sorted(rust_ops.values(), key=lambda x: -x.total_ms)
    for op in sorted_ops:
        pct = (op.total_ms / total_ms * 100) if total_ms > 0 else 0
        per_item = op.per_item_us if op.per_item_us else (
            op.total_ms * 1000 / op.item_count if op.item_count > 0 else 0
        )
        print(f"  {op.name:<38} {op.total_ms:>12,.2f} {pct:>7.1f}% {per_item:>14,.2f}")
    print()
    
    # Python Layer Overhead
    print("PYTHON-SIDE OVERHEAD")
    print("-" * 40)
    python_overhead = 0
    for name, timing in python_timings.items():
        if name in ('batch_total', 'index_creation', 'search_total'):
            continue
        ms = timing.get('total_ms', 0)
        python_overhead += ms
        if ms > 0.001:
            print(f"  {name:<30} {ms:>10.4f} ms")
    print(f"  {'TOTAL PYTHON OVERHEAD':<30} {python_overhead:>10.4f} ms")
    print(f"  {'Percentage of total':<30} {python_overhead/total_ms*100:>10.4f}%")
    print()
    
    # Bottleneck Analysis
    bottlenecks = analyze_bottlenecks(rust_ops, total_ms)
    
    print("BOTTLENECK ANALYSIS")
    print("=" * 80)
    for name, pct, rec in bottlenecks[:5]:
        if pct >= 5:
            bar_len = int(pct / 2)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            print(f"\n{name}")
            print(f"  [{bar}] {pct:.1f}%")
            if rec:
                print(f"\n{rec}")
    print()
    
    # Visual breakdown if significant operations
    print("VISUAL TIME BREAKDOWN")
    print("=" * 80)
    
    # Group into major phases
    phases = {
        'FFI Overhead': 0,
        'Phase 1: Quantization': 0,
        'Phase 2: Map Insert': 0,
        'Phase 3: Search': 0,
        'Phase 3: Neighbor Select': 0,
        'Phase 3: Connections': 0,
        'Other': 0,
    }
    
    for name, stats in rust_ops.items():
        if 'id_conversion' in name or 'slice_from_raw' in name:
            phases['FFI Overhead'] += stats.total_ms
        elif 'quantize' in name:
            phases['Phase 1: Quantization'] += stats.total_ms
        elif 'map_insert' in name:
            phases['Phase 2: Map Insert'] += stats.total_ms
        elif 'search_layer' in name:
            phases['Phase 3: Search'] += stats.total_ms
        elif 'neighbor_select' in name:
            phases['Phase 3: Neighbor Select'] += stats.total_ms
        elif 'add_connections' in name:
            phases['Phase 3: Connections'] += stats.total_ms
    
    max_width = 60
    for phase, ms in phases.items():
        if ms < 0.001:
            continue
        pct = (ms / total_ms * 100) if total_ms > 0 else 0
        bar_len = int((ms / total_ms) * max_width) if total_ms > 0 else 0
        bar = "▓" * bar_len
        print(f"  {phase:<28} {bar:<60} {pct:>6.1f}% ({ms:,.1f}ms)")
    print()
    
    # Recommendations
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    
    if rust_ops.get('hnsw.phase3.neighbor_select', OperationStats('',0,0,0,0,0)).total_ms > total_ms * 0.5:
        print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ CRITICAL BOTTLENECK: Neighbor Selection (RNG Heuristic)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ The RNG (Relative Neighborhood Graph) heuristic is consuming 77% of time.  │
│                                                                              │
│ This is expected for high-dimensional vectors (768d) with large             │
│ ef_construction (200). The algorithm does O(ef × m × distance_calcs).       │
│                                                                              │
│ RECOMMENDED OPTIMIZATIONS:                                                   │
│                                                                              │
│ 1. REDUCE EF_CONSTRUCTION:                                                   │
│    Current: 200 → Suggested: 100-128                                         │
│    Trade-off: Slightly lower recall, 2x faster insert                       │
│                                                                              │
│ 2. USE BATCH DISTANCE COMPUTATION:                                           │
│    The RNG loop currently does scalar distance calculations.                 │
│    Batching 8 candidates at a time with AVX2/NEON would help.               │
│                                                                              │
│ 3. PRE-COMPUTED DISTANCE CACHE:                                              │
│    Cache pairwise distances between candidates to avoid recomputation.       │
│                                                                              │
│ 4. APPROXIMATE RNG:                                                          │
│    Use simpler heuristics (e.g., top-k by distance) for early insert        │
│    when graph is sparse, switch to full RNG when graph is dense.            │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    print()
    print("END OF PROFILING REPORT")
    print("=" * 80)


def create_visualization(
    rust_ops: Dict[str, OperationStats],
    total_ms: float,
    output_path: str = "profiling_chart.png"
):
    """Create a pie chart visualization of time breakdown."""
    if not HAS_MATPLOTLIB:
        print("Note: matplotlib not installed, skipping visualization")
        return
    
    # Prepare data
    phases = {}
    for name, stats in rust_ops.items():
        # Simplify names for chart
        if 'quantize' in name:
            phases['Quantization'] = phases.get('Quantization', 0) + stats.total_ms
        elif 'map_insert' in name:
            phases['Map Insert'] = phases.get('Map Insert', 0) + stats.total_ms
        elif 'search_layer' in name:
            phases['Layer Search'] = phases.get('Layer Search', 0) + stats.total_ms
        elif 'neighbor_select' in name:
            phases['Neighbor Selection'] = phases.get('Neighbor Selection', 0) + stats.total_ms
        elif 'add_connections' in name:
            phases['Add Connections'] = phases.get('Add Connections', 0) + stats.total_ms
        elif 'id_conversion' in name or 'slice' in name:
            phases['FFI Overhead'] = phases.get('FFI Overhead', 0) + stats.total_ms
    
    # Filter small values
    phases = {k: v for k, v in phases.items() if v > total_ms * 0.01}
    
    # Create pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9']
    wedges, texts, autotexts = ax1.pie(
        phases.values(),
        labels=phases.keys(),
        autopct='%1.1f%%',
        colors=colors[:len(phases)],
        explode=[0.05 if v == max(phases.values()) else 0 for v in phases.values()],
    )
    ax1.set_title('HNSW Insert Time Breakdown\n(by phase)')
    
    # Bar chart
    names = list(phases.keys())
    values = list(phases.values())
    bars = ax2.barh(names, values, color=colors[:len(phases)])
    ax2.set_xlabel('Time (ms)')
    ax2.set_title('Time per Phase (ms)')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{val:,.0f}ms', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze ToonDB profiling data')
    parser.add_argument('--python-profile', default='profiling_results.json',
                       help='Path to Python profiling JSON')
    parser.add_argument('--rust-profile', default='/tmp/toondb_profile.json',
                       help='Path to Rust profiling JSON')
    parser.add_argument('--output-chart', default='profiling_chart.png',
                       help='Path to output chart (requires matplotlib)')
    
    args = parser.parse_args()
    
    # Check for files
    if not os.path.exists(args.rust_profile):
        print(f"Error: Rust profile not found at {args.rust_profile}")
        print("Run the profiling first with TOONDB_PROFILING=1")
        sys.exit(1)
    
    # Load profiles
    rust_ops, total_ms = load_rust_profile(args.rust_profile)
    
    python_timings = {}
    meta = {}
    if os.path.exists(args.python_profile):
        python_timings, meta = load_python_profile(args.python_profile)
    else:
        print(f"Warning: Python profile not found at {args.python_profile}")
        meta = {'num_vectors': 1000, 'dimension': 768}
    
    # Print report
    print_detailed_report(rust_ops, python_timings, meta, total_ms)
    
    # Create visualization
    if HAS_MATPLOTLIB:
        create_visualization(rust_ops, total_ms, args.output_chart)


if __name__ == '__main__':
    main()
