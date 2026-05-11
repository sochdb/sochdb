#!/usr/bin/env python3
"""
H2O db-benchmark data generator (Python port of _data/groupby-datagen.R).

Generates the same schema as the H2O benchmark:
  id1 VARCHAR, id2 VARCHAR, id3 VARCHAR, id4 INT, id5 INT, id6 INT, v1 INT, v2 INT, v3 FLOAT

Usage:
  python h2o_datagen.py [N] [K] [NAs] [Sort]
  python h2o_datagen.py 1e7 1e2 0 0   # 10M rows, K=100, no NAs, random order
  python h2o_datagen.py 1e6 1e2 0 0   # 1M rows (quick test)
"""

import os
import sys
import time
import numpy as np


def pretty_sci(n: int) -> str:
    s = f"{n:.0e}"
    base, exp = s.split("e+") if "e+" in s else s.split("e")
    exp = str(int(exp))
    return f"{base}e{exp}"


def generate_groupby_data(N: int, K: int, nas: int = 0, sort: int = 0, out_dir: str = "data"):
    """Generate H2O groupby benchmark data and write to CSV."""
    os.makedirs(out_dir, exist_ok=True)
    
    rng = np.random.default_rng(108)
    
    print(f"Generating {pretty_sci(N)} rows, K={pretty_sci(K)}, NAs={nas}%, sort={sort}")
    t0 = time.time()
    
    # id1: large groups (char), e.g. "id001"..."id100"
    id1_vals = np.array([f"id{i+1:03d}" for i in range(K)])
    id1 = rng.choice(id1_vals, N)
    
    # id2: small groups (char)
    id2_vals = np.array([f"id{i+1:03d}" for i in range(K)])
    id2 = rng.choice(id2_vals, N)
    
    # id3: large groups (char), e.g. "id0000000001" 
    n_id3 = N // K
    id3_vals = np.array([f"id{i+1:010d}" for i in range(n_id3)])
    id3 = rng.choice(id3_vals, N)
    
    # id4: large groups (int)
    id4 = rng.integers(1, K + 1, N)
    
    # id5: small groups (int)
    id5 = rng.integers(1, K + 1, N)
    
    # id6: small groups (int)
    id6 = rng.integers(1, N // K + 1, N)
    
    # v1: int in [1, 5]
    v1 = rng.integers(1, 6, N)
    
    # v2: int in [1, 15]
    v2 = rng.integers(1, 16, N)
    
    # v3: float in [0, 100) rounded to 6 decimal places
    v3 = np.round(rng.random(N) * 100, 6)
    
    elapsed_gen = time.time() - t0
    print(f"  Data generated in {elapsed_gen:.1f}s")
    
    # Write CSV
    file_name = f"G1_{pretty_sci(N)}_{pretty_sci(K)}_{nas}_{sort}.csv"
    file_path = os.path.join(out_dir, file_name)
    
    print(f"  Writing to {file_path}...")
    t0 = time.time()
    
    with open(file_path, 'w') as f:
        f.write("id1,id2,id3,id4,id5,id6,v1,v2,v3\n")
        for i in range(N):
            f.write(f"{id1[i]},{id2[i]},{id3[i]},{id4[i]},{id5[i]},{id6[i]},{v1[i]},{v2[i]},{v3[i]}\n")
    
    elapsed_write = time.time() - t0
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  Written {file_size_mb:.1f} MB in {elapsed_write:.1f}s")
    
    return file_path, file_name.replace(".csv", "")


def generate_join_data(N: int, nas: int = 0, sort: int = 0, out_dir: str = "data"):
    """Generate H2O join benchmark data: x table + small/medium/big RHS tables."""
    os.makedirs(out_dir, exist_ok=True)
    
    rng = np.random.default_rng(108)
    
    data_name = f"J1_{pretty_sci(N)}_NA_{nas}_{sort}"
    print(f"Generating join data: {data_name}")
    t0 = time.time()
    
    # x table: id1 INT, id2 INT, id3 INT, id4 VARCHAR, id5 VARCHAR, id6 VARCHAR, v1 DOUBLE
    id1 = rng.integers(1, int(N / 1e6) + 1, N)
    id2 = rng.integers(1, int(N / 1e3) + 1, N)
    id3 = rng.integers(1, N + 1, N)
    id4_vals = np.array([f"id{i+1}" for i in range(int(N / 1e6))])
    id4 = rng.choice(id4_vals, N) if len(id4_vals) > 0 else np.array(["id1"] * N)
    id5_vals = np.array([f"id{i+1}" for i in range(int(N / 1e3))])
    id5 = rng.choice(id5_vals, N) if len(id5_vals) > 0 else np.array(["id1"] * N)
    id6_vals = np.array([f"id{i+1}" for i in range(N)])
    id6 = np.array([f"id{i+1}" for i in rng.integers(0, N, N)])
    v1 = np.round(rng.random(N) * 100, 6)
    
    # Write x table
    x_path = os.path.join(out_dir, f"{data_name}.csv")
    with open(x_path, 'w') as f:
        f.write("id1,id2,id3,id4,id5,id6,v1\n")
        for i in range(N):
            f.write(f"{id1[i]},{id2[i]},{id3[i]},{id4[i]},{id5[i]},{id6[i]},{v1[i]}\n")
    
    # small: N/1e6 rows — id1 INT, id4 VARCHAR, v2 DOUBLE
    n_small = max(int(N / 1e6), 1)
    sm_id1 = np.arange(1, n_small + 1)
    sm_id4 = np.array([f"id{i+1}" for i in range(n_small)])
    sm_v2 = np.round(rng.random(n_small) * 100, 6)
    sm_name = data_name.replace("NA", pretty_sci(n_small))
    sm_path = os.path.join(out_dir, f"{sm_name}.csv")
    with open(sm_path, 'w') as f:
        f.write("id1,id4,v2\n")
        for i in range(n_small):
            f.write(f"{sm_id1[i]},{sm_id4[i]},{sm_v2[i]}\n")
    
    # medium: N/1e3 rows — id1 INT, id2 INT, id4 VARCHAR, id5 VARCHAR, v2 DOUBLE
    n_medium = max(int(N / 1e3), 1)
    md_id1 = rng.integers(1, n_small + 1, n_medium)
    md_id2 = np.arange(1, n_medium + 1)
    md_id4 = rng.choice(sm_id4, n_medium) if len(sm_id4) > 0 else np.array(["id1"] * n_medium)
    md_id5 = np.array([f"id{i+1}" for i in range(n_medium)])
    md_v2 = np.round(rng.random(n_medium) * 100, 6)
    md_name = data_name.replace("NA", pretty_sci(n_medium))
    md_path = os.path.join(out_dir, f"{md_name}.csv")
    with open(md_path, 'w') as f:
        f.write("id1,id2,id4,id5,v2\n")
        for i in range(n_medium):
            f.write(f"{md_id1[i]},{md_id2[i]},{md_id4[i]},{md_id5[i]},{md_v2[i]}\n")
    
    # big: N rows — id1 INT, id2 INT, id3 INT, id4 VARCHAR, id5 VARCHAR, id6 VARCHAR, v2 DOUBLE
    bg_id1 = rng.integers(1, n_small + 1, N)
    bg_id2 = rng.integers(1, n_medium + 1, N)
    bg_id3 = rng.integers(1, N + 1, N)
    bg_id4 = rng.choice(sm_id4, N) if len(sm_id4) > 0 else np.array(["id1"] * N)
    bg_id5 = rng.choice(md_id5, N) if len(md_id5) > 0 else np.array(["id1"] * N)
    bg_id6 = np.array([f"id{i+1}" for i in rng.integers(0, N, N)])
    bg_v2 = np.round(rng.random(N) * 100, 6)
    bg_name = data_name.replace("NA", pretty_sci(N))
    bg_path = os.path.join(out_dir, f"{bg_name}.csv")
    with open(bg_path, 'w') as f:
        f.write("id1,id2,id3,id4,id5,id6,v2\n")
        for i in range(N):
            f.write(f"{bg_id1[i]},{bg_id2[i]},{bg_id3[i]},{bg_id4[i]},{bg_id5[i]},{bg_id6[i]},{bg_v2[i]}\n")
    
    elapsed = time.time() - t0
    print(f"  Join data generated in {elapsed:.1f}s")
    print(f"    x: {x_path} ({N} rows)")
    print(f"    small: {sm_path} ({n_small} rows)")
    print(f"    medium: {md_path} ({n_medium} rows)")
    print(f"    big: {bg_path} ({N} rows)")
    
    return data_name, x_path, sm_path, md_path, bg_path


if __name__ == "__main__":
    args = sys.argv[1:]
    N = int(float(args[0])) if len(args) > 0 else int(1e6)
    K = int(float(args[1])) if len(args) > 1 else 100
    nas = int(args[2]) if len(args) > 2 else 0
    sort = int(args[3]) if len(args) > 3 else 0
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    print("=" * 60)
    print("H2O db-benchmark Data Generator")
    print("=" * 60)
    
    # Generate groupby data
    file_path, data_name = generate_groupby_data(N, K, nas, sort, out_dir)
    print(f"Groupby data: {data_name}")
    
    # Generate join data
    join_data_name, *join_paths = generate_join_data(N, nas, sort, out_dir)
    print(f"Join data: {join_data_name}")
    
    print("\nDone!")
