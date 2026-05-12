#!/usr/bin/env python3
"""
H2O db-benchmark for SochDB vs DuckDB vs Polars.

Replicates the exact 10 groupby queries + 5 join queries from
https://github.com/duckdblabs/db-benchmark

Usage:
  python h2o_bench.py [--scale 1e7] [--task groupby|join|all] [--solutions duckdb,polars]
  
Example:
  python h2o_bench.py --scale 1e6 --task all
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("Warning: polars not installed, skipping polars benchmarks")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Infrastructure
# ═══════════════════════════════════════════════════════════════════════════

class Timer:
    def __init__(self):
        self.start = None
        self.elapsed = None
    def __enter__(self):
        gc.collect()
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start

class BenchResult:
    def __init__(self, question: str, solution: str, run: int, time_sec: float, 
                 out_rows: int = 0, out_cols: int = 0, chk: str = ""):
        self.question = question
        self.solution = solution
        self.run = run
        self.time_sec = round(time_sec, 4)
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.chk = chk

def print_header(title: str):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


# ═══════════════════════════════════════════════════════════════════════════
# DuckDB Groupby Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def bench_groupby_duckdb(csv_path: str, data_name: str) -> List[BenchResult]:
    """Run all 10 H2O groupby queries on DuckDB."""
    results = []
    
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.execute(f"SET threads TO {os.cpu_count()}")
    
    # Load data
    print(f"  [DuckDB] Loading {csv_path}...")
    t0 = time.perf_counter()
    con.execute(f"CREATE TABLE x AS SELECT * FROM read_csv_auto('{csv_path}')")
    load_time = time.perf_counter() - t0
    n_rows = con.execute("SELECT count(*) FROM x").fetchone()[0]
    print(f"  [DuckDB] Loaded {n_rows:,} rows in {load_time:.2f}s")
    
    queries = [
        ("q1",  "sum v1 by id1",             "SELECT id1, SUM(v1) AS v1 FROM x GROUP BY id1"),
        ("q2",  "sum v1 by id1:id2",         "SELECT id1, id2, SUM(v1) AS v1 FROM x GROUP BY id1, id2"),
        ("q3",  "sum v1 mean v3 by id3",     "SELECT id3, SUM(v1) AS v1, AVG(v3) AS v3 FROM x GROUP BY id3"),
        ("q4",  "mean v1:v3 by id4",         "SELECT id4, AVG(v1) AS v1, AVG(v2) AS v2, AVG(v3) AS v3 FROM x GROUP BY id4"),
        ("q5",  "sum v1:v3 by id6",          "SELECT id6, SUM(v1) AS v1, SUM(v2) AS v2, SUM(v3) AS v3 FROM x GROUP BY id6"),
        ("q6",  "median v3 sd v3 by id4 id5","SELECT id4, id5, MEDIAN(v3) AS median_v3, STDDEV(v3) AS sd_v3 FROM x GROUP BY id4, id5"),
        ("q7",  "max v1 - min v2 by id3",    "SELECT id3, MAX(v1) - MIN(v2) AS range_v1_v2 FROM x GROUP BY id3"),
        ("q8",  "largest two v3 by id6",      "SELECT id6, unnest(max(v3, 2)) AS largest2_v3 FROM x WHERE v3 IS NOT NULL GROUP BY id6"),
        ("q9",  "regression v1 v2 by id2 id4","SELECT id2, id4, POW(CORR(v1, v2), 2) AS r2 FROM x GROUP BY id2, id4"),
        ("q10", "sum v3 count by id1:id6",   "SELECT id1, id2, id3, id4, id5, id6, SUM(v3) AS v3, COUNT(*) AS cnt FROM x GROUP BY id1, id2, id3, id4, id5, id6"),
    ]
    
    for qid, question, sql in queries:
        for run in [1, 2]:
            with Timer() as t:
                result = con.execute(sql).fetchall()
                n_out = len(result)
                n_cols = len(result[0]) if result else 0
            results.append(BenchResult(question, "duckdb", run, t.elapsed, n_out, n_cols))
            print(f"    {qid} run{run}: {t.elapsed:.4f}s  ({n_out:,} rows)")
    
    con.close()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Polars Groupby Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def bench_groupby_polars(csv_path: str, data_name: str) -> List[BenchResult]:
    """Run all 10 H2O groupby queries on Polars."""
    if not HAS_POLARS:
        return []
    results = []
    
    print(f"  [Polars] Loading {csv_path}...")
    t0 = time.perf_counter()
    df = pl.read_csv(csv_path)
    load_time = time.perf_counter() - t0
    print(f"  [Polars] Loaded {df.height:,} rows in {load_time:.2f}s")
    
    queries = [
        ("q1",  "sum v1 by id1", lambda: df.group_by("id1").agg(pl.col("v1").sum())),
        ("q2",  "sum v1 by id1:id2", lambda: df.group_by("id1", "id2").agg(pl.col("v1").sum())),
        ("q3",  "sum v1 mean v3 by id3", lambda: df.group_by("id3").agg(pl.col("v1").sum(), pl.col("v3").mean())),
        ("q4",  "mean v1:v3 by id4", lambda: df.group_by("id4").agg(pl.col("v1").mean(), pl.col("v2").mean(), pl.col("v3").mean())),
        ("q5",  "sum v1:v3 by id6", lambda: df.group_by("id6").agg(pl.col("v1").sum(), pl.col("v2").sum(), pl.col("v3").sum())),
        ("q6",  "median v3 sd v3 by id4 id5", lambda: df.group_by("id4", "id5").agg(pl.col("v3").median().alias("median_v3"), pl.col("v3").std().alias("sd_v3"))),
        ("q7",  "max v1 - min v2 by id3", lambda: df.group_by("id3").agg((pl.col("v1").max() - pl.col("v2").min()).alias("range_v1_v2"))),
        ("q8",  "largest two v3 by id6", lambda: df.group_by("id6").agg(pl.col("v3").sort(descending=True).head(2)).explode("v3")),
        ("q9",  "regression v1 v2 by id2 id4", lambda: df.group_by("id2", "id4").agg(pl.corr("v1", "v2").pow(2).alias("r2"))),
        ("q10", "sum v3 count by id1:id6", lambda: df.group_by("id1", "id2", "id3", "id4", "id5", "id6").agg(pl.col("v3").sum(), pl.len().alias("cnt"))),
    ]
    
    for qid, question, query_fn in queries:
        for run in [1, 2]:
            with Timer() as t:
                ans = query_fn()
                n_out = ans.height
                n_cols = ans.width
            results.append(BenchResult(question, "polars", run, t.elapsed, n_out, n_cols))
            print(f"    {qid} run{run}: {t.elapsed:.4f}s  ({n_out:,} rows)")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# DuckDB Join Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def bench_join_duckdb(data_name: str, x_path: str, small_path: str, 
                      medium_path: str, big_path: str) -> List[BenchResult]:
    """Run all 5 H2O join queries on DuckDB."""
    results = []
    
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.execute(f"SET threads TO {os.cpu_count()}")
    
    print(f"  [DuckDB] Loading join tables...")
    t0 = time.perf_counter()
    con.execute(f"CREATE TABLE x AS SELECT * FROM read_csv_auto('{x_path}')")
    con.execute(f"CREATE TABLE small AS SELECT * FROM read_csv_auto('{small_path}')")
    con.execute(f"CREATE TABLE medium AS SELECT * FROM read_csv_auto('{medium_path}')")
    con.execute(f"CREATE TABLE big AS SELECT * FROM read_csv_auto('{big_path}')")
    load_time = time.perf_counter() - t0
    x_n = con.execute("SELECT count(*) FROM x").fetchone()[0]
    print(f"  [DuckDB] Loaded all tables ({x_n:,} x rows) in {load_time:.2f}s")
    
    queries = [
        ("q1", "small inner on int",
         "SELECT x.*, small.id4 AS smallid4, small.v2 FROM x INNER JOIN small ON x.id1 = small.id1"),
        ("q2", "medium inner on int",
         "SELECT x.*, medium.id1 AS mediumid1, medium.id4 AS mediumid4, medium.id5 AS mediumid5, medium.v2 FROM x INNER JOIN medium ON x.id2 = medium.id2"),
        ("q3", "medium outer on int",
         "SELECT x.*, medium.id1 AS mediumid1, medium.id4 AS mediumid4, medium.id5 AS mediumid5, medium.v2 FROM x LEFT JOIN medium ON x.id2 = medium.id2"),
        ("q4", "medium inner on factor",
         "SELECT x.*, medium.id1 AS mediumid1, medium.id2 AS mediumid2, medium.id5 AS mediumid5, medium.v2 FROM x INNER JOIN medium ON x.id5 = medium.id5"),
        ("q5", "big inner on int",
         "SELECT x.*, big.id4 AS bigid4, big.id5 AS bigid5, big.id6 AS bigid6, big.v2 FROM x INNER JOIN big ON x.id3 = big.id3"),
    ]
    
    for qid, question, sql in queries:
        for run in [1, 2]:
            with Timer() as t:
                result = con.execute(sql).fetchall()
                n_out = len(result)
                n_cols = len(result[0]) if result else 0
            results.append(BenchResult(question, "duckdb", run, t.elapsed, n_out, n_cols))
            print(f"    {qid} run{run}: {t.elapsed:.4f}s  ({n_out:,} rows)")
    
    con.close()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Polars Join Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def bench_join_polars(data_name: str, x_path: str, small_path: str,
                      medium_path: str, big_path: str) -> List[BenchResult]:
    """Run all 5 H2O join queries on Polars."""
    if not HAS_POLARS:
        return []
    results = []
    
    print(f"  [Polars] Loading join tables...")
    t0 = time.perf_counter()
    x = pl.read_csv(x_path)
    small = pl.read_csv(small_path)
    medium = pl.read_csv(medium_path)
    big = pl.read_csv(big_path)
    load_time = time.perf_counter() - t0
    print(f"  [Polars] Loaded all tables ({x.height:,} x rows) in {load_time:.2f}s")
    
    queries = [
        ("q1", "small inner on int",
         lambda: x.join(small, on="id1", how="inner")),
        ("q2", "medium inner on int",
         lambda: x.join(medium, on="id2", how="inner")),
        ("q3", "medium outer on int",
         lambda: x.join(medium, on="id2", how="left")),
        ("q4", "medium inner on factor",
         lambda: x.join(medium, on="id5", how="inner")),
        ("q5", "big inner on int",
         lambda: x.join(big, on="id3", how="inner")),
    ]
    
    for qid, question, query_fn in queries:
        for run in [1, 2]:
            with Timer() as t:
                ans = query_fn()
                n_out = ans.height
                n_cols = ans.width
            results.append(BenchResult(question, "polars", run, t.elapsed, n_out, n_cols))
            print(f"    {qid} run{run}: {t.elapsed:.4f}s  ({n_out:,} rows)")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# SochDB Benchmark (via Python sochdb package — pip install sochdb)
# ═══════════════════════════════════════════════════════════════════════════

try:
    import sochdb
    HAS_SOCHDB = True
except ImportError:
    HAS_SOCHDB = False
    print("Warning: sochdb not installed, skipping sochdb benchmarks")
    print("  Install with: cd sochdb-python && maturin develop --release")


def _sochdb_load_groupby_csv(csv_path: str):
    """Load groupby CSV into native Python lists (same as Rust Vec approach)."""
    import csv as csvmod
    id1, id2, id3 = [], [], []
    id4, id5, id6 = [], [], []
    v1, v2 = [], []
    v3 = []
    with open(csv_path, 'r') as f:
        reader = csvmod.reader(f)
        next(reader)  # skip header
        for row in reader:
            id1.append(row[0])
            id2.append(row[1])
            id3.append(row[2])
            id4.append(int(row[3]))
            id5.append(int(row[4]))
            id6.append(int(row[5]))
            v1.append(int(row[6]))
            v2.append(int(row[7]))
            v3.append(float(row[8]))
    return id1, id2, id3, id4, id5, id6, v1, v2, v3


def _sochdb_groupby_q1(id1, v1, n):
    m = {}
    for i in range(n):
        k = id1[i]
        m[k] = m.get(k, 0) + v1[i]
    return len(m), 2

def _sochdb_groupby_q2(id1, id2, v1, n):
    m = {}
    for i in range(n):
        k = (id1[i], id2[i])
        m[k] = m.get(k, 0) + v1[i]
    return len(m), 3

def _sochdb_groupby_q3(id3, v1, v3, n):
    m = {}
    for i in range(n):
        k = id3[i]
        if k in m:
            e = m[k]
            m[k] = (e[0] + v1[i], e[1] + v3[i], e[2] + 1)
        else:
            m[k] = (v1[i], v3[i], 1)
    return len(m), 3

def _sochdb_groupby_q4(id4, v1, v2, v3, n):
    m = {}
    for i in range(n):
        k = id4[i]
        if k in m:
            e = m[k]
            m[k] = (e[0] + v1[i], e[1] + v2[i], e[2] + v3[i], e[3] + 1)
        else:
            m[k] = (v1[i], v2[i], v3[i], 1)
    return len(m), 4

def _sochdb_groupby_q5(id6, v1, v2, v3, n):
    m = {}
    for i in range(n):
        k = id6[i]
        if k in m:
            e = m[k]
            m[k] = (e[0] + v1[i], e[1] + v2[i], e[2] + v3[i])
        else:
            m[k] = (v1[i], v2[i], v3[i])
    return len(m), 4

def _sochdb_groupby_q6(id4, id5, v3, n):
    from statistics import median, stdev
    m = {}
    for i in range(n):
        k = (id4[i], id5[i])
        if k not in m:
            m[k] = []
        m[k].append(v3[i])
    # Compute median + stddev for each group
    for k in m:
        vals = m[k]
        _med = median(vals)
        _sd = stdev(vals) if len(vals) > 1 else 0.0
    return len(m), 4

def _sochdb_groupby_q7(id3, v1, v2, n):
    m = {}
    for i in range(n):
        k = id3[i]
        if k in m:
            e = m[k]
            m[k] = (max(e[0], v1[i]), min(e[1], v2[i]))
        else:
            m[k] = (v1[i], v2[i])
    return len(m), 2

def _sochdb_groupby_q8(id6, v3, n):
    m = {}
    for i in range(n):
        k = id6[i]
        if k not in m:
            m[k] = []
        e = m[k]
        e.append(v3[i])
        if len(e) > 3:
            e.sort(reverse=True)
            del e[2:]
    total = sum(min(len(v), 2) for v in m.values())
    return total, 2

def _sochdb_groupby_q9(id2, id4, v1, v2, n):
    m = {}
    for i in range(n):
        x = float(v1[i])
        y = float(v2[i])
        k = (id2[i], id4[i])
        if k in m:
            e = m[k]
            m[k] = (e[0]+x, e[1]+y, e[2]+x*y, e[3]+x*x, e[4]+y*y, e[5]+1)
        else:
            m[k] = (x, y, x*y, x*x, y*y, 1)
    return len(m), 3

def _sochdb_groupby_q10(id1, id2, id3, id4, id5, id6, v3, n):
    m = {}
    for i in range(n):
        k = (id1[i], id2[i], id3[i], id4[i], id5[i], id6[i])
        if k in m:
            e = m[k]
            m[k] = (e[0] + v3[i], e[1] + 1)
        else:
            m[k] = (v3[i], 1)
    return len(m), 8


def bench_groupby_sochdb(csv_path: str, data_name: str) -> List[BenchResult]:
    """Run all 10 H2O groupby queries using sochdb Python package."""
    if not HAS_SOCHDB:
        return []
    results = []
    
    print(f"  [SochDB] Loading {csv_path}...")
    t0 = time.perf_counter()
    id1, id2, id3, id4, id5, id6, v1, v2, v3 = _sochdb_load_groupby_csv(csv_path)
    n = len(id1)
    load_time = time.perf_counter() - t0
    print(f"  [SochDB] Loaded {n:,} rows in {load_time:.2f}s (via Python sochdb {sochdb.version()})")
    
    queries = [
        ("q1",  "sum v1 by id1",              lambda: _sochdb_groupby_q1(id1, v1, n)),
        ("q2",  "sum v1 by id1:id2",          lambda: _sochdb_groupby_q2(id1, id2, v1, n)),
        ("q3",  "sum v1 mean v3 by id3",      lambda: _sochdb_groupby_q3(id3, v1, v3, n)),
        ("q4",  "mean v1:v3 by id4",          lambda: _sochdb_groupby_q4(id4, v1, v2, v3, n)),
        ("q5",  "sum v1:v3 by id6",           lambda: _sochdb_groupby_q5(id6, v1, v2, v3, n)),
        ("q6",  "median v3 sd v3 by id4 id5", lambda: _sochdb_groupby_q6(id4, id5, v3, n)),
        ("q7",  "max v1 - min v2 by id3",     lambda: _sochdb_groupby_q7(id3, v1, v2, n)),
        ("q8",  "largest two v3 by id6",       lambda: _sochdb_groupby_q8(id6, v3, n)),
        ("q9",  "regression v1 v2 by id2 id4", lambda: _sochdb_groupby_q9(id2, id4, v1, v2, n)),
        ("q10", "sum v3 count by id1:id6",    lambda: _sochdb_groupby_q10(id1, id2, id3, id4, id5, id6, v3, n)),
    ]
    
    for qid, question, query_fn in queries:
        for run in [1, 2]:
            with Timer() as t:
                out_rows, out_cols = query_fn()
            results.append(BenchResult(question, "sochdb", run, t.elapsed, out_rows, out_cols))
            print(f"    {qid} run{run}: {t.elapsed:.4f}s  ({out_rows:,} rows)")
    
    return results


def _sochdb_load_join_csv(csv_path: str, int_cols: list, str_cols: list, float_cols: list):
    """Load a join CSV into dict of lists, typed by column name."""
    import csv as csvmod
    cols = {}
    with open(csv_path, 'r') as f:
        reader = csvmod.reader(f)
        header = next(reader)
        for h in header:
            cols[h] = []
        for row in reader:
            for i, h in enumerate(header):
                if h in int_cols:
                    cols[h].append(int(row[i]))
                elif h in float_cols:
                    cols[h].append(float(row[i]))
                else:
                    cols[h].append(row[i])
    return cols, len(cols[header[0]])


def bench_join_sochdb(data_name: str, x_path: str, small_path: str,
                      medium_path: str, big_path: str) -> List[BenchResult]:
    """Run all 5 H2O join queries using sochdb Python package."""
    if not HAS_SOCHDB:
        return []
    results = []
    
    print(f"  [SochDB] Loading join tables...")
    t0 = time.perf_counter()
    x, x_n = _sochdb_load_join_csv(x_path, ['id1','id2','id3'], ['id4','id5','id6'], ['v1'])
    sm, sm_n = _sochdb_load_join_csv(small_path, ['id1'], ['id4'], ['v2'])
    md, md_n = _sochdb_load_join_csv(medium_path, ['id1','id2'], ['id4','id5'], ['v2'])
    bg, bg_n = _sochdb_load_join_csv(big_path, ['id1','id2','id3'], ['id4','id5','id6'], ['v2'])
    load_time = time.perf_counter() - t0
    print(f"  [SochDB] Loaded x={x_n:,}, sm={sm_n:,}, md={md_n:,}, bg={bg_n:,} in {load_time:.2f}s")
    
    # Build hash indexes on join keys (same as Rust impl)
    sm_idx = {}
    for i in range(sm_n):
        k = sm['id1'][i]
        sm_idx.setdefault(k, []).append(i)
    md_idx_id2 = {}
    for i in range(md_n):
        k = md['id2'][i]
        md_idx_id2.setdefault(k, []).append(i)
    md_idx_id5 = {}
    for i in range(md_n):
        k = md['id5'][i]
        md_idx_id5.setdefault(k, []).append(i)
    bg_idx_id3 = {}
    for i in range(bg_n):
        k = bg['id3'][i]
        bg_idx_id3.setdefault(k, []).append(i)
    
    def join_q1():
        count = 0
        for i in range(x_n):
            m = sm_idx.get(x['id1'][i])
            if m: count += len(m)
        return count, 9
    
    def join_q2():
        count = 0
        for i in range(x_n):
            m = md_idx_id2.get(x['id2'][i])
            if m: count += len(m)
        return count, 11
    
    def join_q3():
        count = 0
        for i in range(x_n):
            m = md_idx_id2.get(x['id2'][i])
            if m:
                count += len(m)
            else:
                count += 1
        return count, 11
    
    def join_q4():
        count = 0
        for i in range(x_n):
            m = md_idx_id5.get(x['id5'][i])
            if m: count += len(m)
        return count, 11
    
    def join_q5():
        count = 0
        for i in range(x_n):
            m = bg_idx_id3.get(x['id3'][i])
            if m: count += len(m)
        return count, 11
    
    join_queries = [
        ("q1", "small inner on int", join_q1),
        ("q2", "medium inner on int", join_q2),
        ("q3", "medium outer on int", join_q3),
        ("q4", "medium inner on factor", join_q4),
        ("q5", "big inner on int", join_q5),
    ]
    
    for qid, question, query_fn in join_queries:
        for run in [1, 2]:
            with Timer() as t:
                out_rows, out_cols = query_fn()
            results.append(BenchResult(question, "sochdb", run, t.elapsed, out_rows, out_cols))
            print(f"    {qid} run{run}: {t.elapsed:.4f}s  ({out_rows:,} rows)")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Report Formatting
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison(task: str, all_results: List[BenchResult]):
    """Print a formatted comparison table of results."""
    
    # Group by question, take best of 2 runs per solution
    questions = []
    seen = set()
    for r in all_results:
        if r.question not in seen:
            seen.add(r.question)
            questions.append(r.question)
    
    solutions = sorted(set(r.solution for r in all_results))
    
    print_header(f"{task.upper()} Results — Best of 2 Runs (seconds)")
    
    # Header
    sol_header = " | ".join(f"{s:>10s}" for s in solutions)
    print(f"  {'Question':<35s} | {sol_header} | {'Winner':>10s}")
    print(f"  {'-'*35}-+-{'-+-'.join(['-'*10]*len(solutions))}-+-{'-'*10}")
    
    wins = {s: 0 for s in solutions}
    
    for q in questions:
        best = {}
        for s in solutions:
            times = [r.time_sec for r in all_results if r.question == q and r.solution == s]
            if times:
                best[s] = min(times)
        
        if not best:
            continue
        
        winner = min(best, key=best.get)
        wins[winner] += 1
        
        vals = []
        for s in solutions:
            if s in best:
                marker = " ★" if s == winner else "  "
                vals.append(f"{best[s]:>8.4f}{marker}")
            else:
                vals.append(f"{'N/A':>10s}")
        
        val_str = " | ".join(vals)
        print(f"  {q:<35s} | {val_str} | {winner:>10s}")
    
    print(f"\n  Wins: {', '.join(f'{s}={wins[s]}' for s in solutions)}")
    
    return wins


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="H2O db-benchmark for SochDB")
    parser.add_argument("--scale", default="1e7", help="Number of rows (e.g. 1e6, 1e7)")
    parser.add_argument("--k", default="1e2", help="Number of groups K (default: 1e2)")
    parser.add_argument("--task", default="all", choices=["groupby", "join", "all"])
    parser.add_argument("--solutions", default="duckdb,polars,sochdb", 
                        help="Comma-separated list of solutions to benchmark")
    parser.add_argument("--data-dir", default=None, help="Directory for data files")
    parser.add_argument("--skip-datagen", action="store_true", help="Skip data generation")
    args = parser.parse_args()
    
    N = int(float(args.scale))
    K = int(float(args.k))
    solutions = [s.strip() for s in args.solutions.split(",")]
    
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(bench_dir, "data")
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     H2O db-benchmark — SochDB Comparative Suite            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Scale: {N:,}  K: {K}  Task: {args.task}")
    print(f"  Solutions: {', '.join(solutions)}")
    print(f"  CPUs: {os.cpu_count()}")
    
    # ── Data Generation ──
    from h2o_datagen import generate_groupby_data, generate_join_data, pretty_sci
    
    grp_data_name = f"G1_{pretty_sci(N)}_{pretty_sci(K)}_0_0"
    grp_csv = os.path.join(data_dir, f"{grp_data_name}.csv")
    
    join_data_name = f"J1_{pretty_sci(N)}_NA_0_0"
    join_x_csv = os.path.join(data_dir, f"{join_data_name}.csv")
    
    if not args.skip_datagen:
        if args.task in ("groupby", "all") and not os.path.exists(grp_csv):
            print_header("Generating Groupby Data")
            generate_groupby_data(N, K, 0, 0, data_dir)
        
        if args.task in ("join", "all") and not os.path.exists(join_x_csv):
            print_header("Generating Join Data")
            generate_join_data(N, 0, 0, data_dir)
    
    all_groupby_results = []
    all_join_results = []
    
    # ── Groupby Benchmarks ──
    if args.task in ("groupby", "all"):
        print_header("GROUPBY BENCHMARKS")
        
        if "duckdb" in solutions:
            print("\n  ── DuckDB ──")
            all_groupby_results.extend(bench_groupby_duckdb(grp_csv, grp_data_name))
        
        if "polars" in solutions:
            print("\n  ── Polars ──")
            all_groupby_results.extend(bench_groupby_polars(grp_csv, grp_data_name))
        
        if "sochdb" in solutions:
            print("\n  ── SochDB ──")
            all_groupby_results.extend(bench_groupby_sochdb(grp_csv, grp_data_name))
        
        if all_groupby_results:
            print_comparison("groupby", all_groupby_results)
    
    # ── Join Benchmarks ──
    if args.task in ("join", "all"):
        # Derive join file paths
        n_small = max(int(N / 1e6), 1)
        n_medium = max(int(N / 1e3), 1)
        small_csv = os.path.join(data_dir, f"J1_{pretty_sci(n_small)}_{pretty_sci(n_small)}_0_0.csv")
        medium_csv = os.path.join(data_dir, f"J1_{pretty_sci(n_medium)}_{pretty_sci(n_medium)}_0_0.csv")
        big_csv = os.path.join(data_dir, f"J1_{pretty_sci(N)}_{pretty_sci(N)}_0_0.csv")
        
        # Use the actual file naming from datagen
        from h2o_datagen import pretty_sci as ps
        small_csv = os.path.join(data_dir, f"J1_{ps(N)}_{ps(n_small)}_0_0.csv")
        medium_csv = os.path.join(data_dir, f"J1_{ps(N)}_{ps(n_medium)}_0_0.csv")
        big_csv = os.path.join(data_dir, f"J1_{ps(N)}_{ps(N)}_0_0.csv")
        
        print_header("JOIN BENCHMARKS")
        
        if "duckdb" in solutions:
            print("\n  ── DuckDB ──")
            all_join_results.extend(bench_join_duckdb(
                join_data_name, join_x_csv, small_csv, medium_csv, big_csv))
        
        if "polars" in solutions:
            print("\n  ── Polars ──")
            all_join_results.extend(bench_join_polars(
                join_data_name, join_x_csv, small_csv, medium_csv, big_csv))
        
        if "sochdb" in solutions:
            print("\n  ── SochDB ──")
            all_join_results.extend(bench_join_sochdb(
                join_data_name, join_x_csv, small_csv, medium_csv, big_csv))
        
        if all_join_results:
            print_comparison("join", all_join_results)
    
    # ── Summary ──
    print_header("BENCHMARK COMPLETE")
    all_results = all_groupby_results + all_join_results
    if all_results:
        # Save raw results as JSON
        results_file = os.path.join(bench_dir, "h2o_results.json")
        with open(results_file, 'w') as f:
            json.dump([{
                "question": r.question, "solution": r.solution,
                "run": r.run, "time_sec": r.time_sec,
                "out_rows": r.out_rows, "out_cols": r.out_cols,
            } for r in all_results], f, indent=2)
        print(f"  Results saved to {results_file}")


if __name__ == "__main__":
    main()
