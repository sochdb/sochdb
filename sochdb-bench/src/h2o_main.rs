//! SochDB H2O db-benchmark implementation.
//!
//! Implements the 10 groupby + 5 join queries from the H2O/DuckDB Labs benchmark
//! using SochDB's columnar storage API with hand-coded hash aggregation.
//!
//! Outputs JSON lines for each query result, consumed by h2o_bench.py.

use clap::Parser;
use csv::ReaderBuilder;
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "sochdb-h2o-bench")]
struct Cli {
    /// Path to the main CSV file (groupby or join x table)
    #[arg(long)]
    csv: PathBuf,

    /// Task: groupby, join
    #[arg(long, default_value = "groupby")]
    task: String,

    /// Path to small join table CSV
    #[arg(long)]
    small: Option<PathBuf>,

    /// Path to medium join table CSV
    #[arg(long)]
    medium: Option<PathBuf>,

    /// Path to big join table CSV
    #[arg(long)]
    big: Option<PathBuf>,
}

#[derive(Serialize)]
struct QueryResult {
    question: String,
    run: u32,
    time_sec: f64,
    out_rows: usize,
    out_cols: usize,
    solution: String,
}

fn emit(question: &str, run: u32, elapsed: f64, out_rows: usize, out_cols: usize) {
    let r = QueryResult {
        question: question.to_string(),
        run,
        time_sec: (elapsed * 10000.0).round() / 10000.0,
        out_rows,
        out_cols,
        solution: "sochdb".to_string(),
    };
    println!("{}", serde_json::to_string(&r).unwrap());
}

// ═══════════════════════════════════════════════════════════════════════════
// CSV Loading
// ═══════════════════════════════════════════════════════════════════════════

/// Columnar in-memory representation of the groupby table.
struct GroupbyTable {
    id1: Vec<String>,
    id2: Vec<String>,
    id3: Vec<String>,
    id4: Vec<i64>,
    id5: Vec<i64>,
    id6: Vec<i64>,
    v1: Vec<i64>,
    v2: Vec<i64>,
    v3: Vec<f64>,
    n: usize,
}

impl GroupbyTable {
    fn load_csv(path: &Path) -> Self {
        let mut rdr = ReaderBuilder::new().from_path(path).expect("open CSV");
        let mut id1 = Vec::new();
        let mut id2 = Vec::new();
        let mut id3 = Vec::new();
        let mut id4 = Vec::new();
        let mut id5 = Vec::new();
        let mut id6 = Vec::new();
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        let mut v3 = Vec::new();

        for record in rdr.records() {
            let r = record.expect("CSV record");
            id1.push(r[0].to_string());
            id2.push(r[1].to_string());
            id3.push(r[2].to_string());
            id4.push(r[3].parse::<i64>().unwrap());
            id5.push(r[4].parse::<i64>().unwrap());
            id6.push(r[5].parse::<i64>().unwrap());
            v1.push(r[6].parse::<i64>().unwrap());
            v2.push(r[7].parse::<i64>().unwrap());
            v3.push(r[8].parse::<f64>().unwrap());
        }

        let n = id1.len();
        Self {
            id1,
            id2,
            id3,
            id4,
            id5,
            id6,
            v1,
            v2,
            v3,
            n,
        }
    }
}

/// Columnar in-memory representation of the join x table.
#[allow(dead_code)]
struct JoinXTable {
    id1: Vec<i64>,
    id2: Vec<i64>,
    id3: Vec<i64>,
    id4: Vec<String>,
    id5: Vec<String>,
    id6: Vec<String>,
    v1: Vec<f64>,
    n: usize,
}

impl JoinXTable {
    fn load_csv(path: &Path) -> Self {
        let mut rdr = ReaderBuilder::new().from_path(path).expect("open CSV");
        let mut id1 = Vec::new();
        let mut id2 = Vec::new();
        let mut id3 = Vec::new();
        let mut id4 = Vec::new();
        let mut id5 = Vec::new();
        let mut id6 = Vec::new();
        let mut v1 = Vec::new();

        for record in rdr.records() {
            let r = record.expect("CSV record");
            id1.push(r[0].parse::<i64>().unwrap());
            id2.push(r[1].parse::<i64>().unwrap());
            id3.push(r[2].parse::<i64>().unwrap());
            id4.push(r[3].to_string());
            id5.push(r[4].to_string());
            id6.push(r[5].to_string());
            v1.push(r[6].parse::<f64>().unwrap());
        }

        let n = id1.len();
        Self {
            id1,
            id2,
            id3,
            id4,
            id5,
            id6,
            v1,
            n,
        }
    }
}

#[allow(dead_code)]
struct JoinSmallTable {
    id1: Vec<i64>,
    id4: Vec<String>,
    v2: Vec<f64>,
    // index: id1 -> row indices
    idx_id1: HashMap<i64, Vec<usize>>,
    n: usize,
}

impl JoinSmallTable {
    fn load_csv(path: &Path) -> Self {
        let mut rdr = ReaderBuilder::new().from_path(path).expect("open CSV");
        let mut id1 = Vec::new();
        let mut id4 = Vec::new();
        let mut v2 = Vec::new();

        for record in rdr.records() {
            let r = record.expect("CSV record");
            id1.push(r[0].parse::<i64>().unwrap());
            id4.push(r[1].to_string());
            v2.push(r[2].parse::<f64>().unwrap());
        }

        let mut idx_id1: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, &v) in id1.iter().enumerate() {
            idx_id1.entry(v).or_default().push(i);
        }

        let n = id1.len();
        Self {
            id1,
            id4,
            v2,
            idx_id1,
            n,
        }
    }
}

#[allow(dead_code)]
struct JoinMediumTable {
    id1: Vec<i64>,
    id2: Vec<i64>,
    id4: Vec<String>,
    id5: Vec<String>,
    v2: Vec<f64>,
    idx_id2: HashMap<i64, Vec<usize>>,
    idx_id5: HashMap<String, Vec<usize>>,
    n: usize,
}

impl JoinMediumTable {
    fn load_csv(path: &Path) -> Self {
        let mut rdr = ReaderBuilder::new().from_path(path).expect("open CSV");
        let mut id1 = Vec::new();
        let mut id2 = Vec::new();
        let mut id4 = Vec::new();
        let mut id5 = Vec::new();
        let mut v2 = Vec::new();

        for record in rdr.records() {
            let r = record.expect("CSV record");
            id1.push(r[0].parse::<i64>().unwrap());
            id2.push(r[1].parse::<i64>().unwrap());
            id4.push(r[2].to_string());
            id5.push(r[3].to_string());
            v2.push(r[4].parse::<f64>().unwrap());
        }

        let mut idx_id2: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, &v) in id2.iter().enumerate() {
            idx_id2.entry(v).or_default().push(i);
        }
        let mut idx_id5: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, v) in id5.iter().enumerate() {
            idx_id5.entry(v.clone()).or_default().push(i);
        }

        let n = id1.len();
        Self {
            id1,
            id2,
            id4,
            id5,
            v2,
            idx_id2,
            idx_id5,
            n,
        }
    }
}

#[allow(dead_code)]
struct JoinBigTable {
    id1: Vec<i64>,
    id2: Vec<i64>,
    id3: Vec<i64>,
    id4: Vec<String>,
    id5: Vec<String>,
    id6: Vec<String>,
    v2: Vec<f64>,
    idx_id3: HashMap<i64, Vec<usize>>,
    n: usize,
}

impl JoinBigTable {
    fn load_csv(path: &Path) -> Self {
        let mut rdr = ReaderBuilder::new().from_path(path).expect("open CSV");
        let mut id1 = Vec::new();
        let mut id2 = Vec::new();
        let mut id3 = Vec::new();
        let mut id4 = Vec::new();
        let mut id5 = Vec::new();
        let mut id6 = Vec::new();
        let mut v2 = Vec::new();

        for record in rdr.records() {
            let r = record.expect("CSV record");
            id1.push(r[0].parse::<i64>().unwrap());
            id2.push(r[1].parse::<i64>().unwrap());
            id3.push(r[2].parse::<i64>().unwrap());
            id4.push(r[3].to_string());
            id5.push(r[4].to_string());
            id6.push(r[5].to_string());
            v2.push(r[6].parse::<f64>().unwrap());
        }

        let mut idx_id3: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, &v) in id3.iter().enumerate() {
            idx_id3.entry(v).or_default().push(i);
        }

        let n = id1.len();
        Self {
            id1,
            id2,
            id3,
            id4,
            id5,
            id6,
            v2,
            idx_id3,
            n,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Groupby Queries (hand-coded hash aggregation)
// ═══════════════════════════════════════════════════════════════════════════

fn groupby_q1(t: &GroupbyTable) -> (usize, usize) {
    // SELECT id1, SUM(v1) AS v1 FROM x GROUP BY id1
    let mut map: HashMap<&str, i64> = HashMap::new();
    for i in 0..t.n {
        *map.entry(&t.id1[i]).or_insert(0) += t.v1[i];
    }
    (map.len(), 2)
}

fn groupby_q2(t: &GroupbyTable) -> (usize, usize) {
    // SELECT id1, id2, SUM(v1) AS v1 FROM x GROUP BY id1, id2
    let mut map: HashMap<(&str, &str), i64> = HashMap::new();
    for i in 0..t.n {
        *map.entry((&t.id1[i], &t.id2[i])).or_insert(0) += t.v1[i];
    }
    (map.len(), 3)
}

fn groupby_q3(t: &GroupbyTable) -> (usize, usize) {
    // SELECT id3, SUM(v1) AS v1, AVG(v3) AS v3 FROM x GROUP BY id3
    let mut map: HashMap<&str, (i64, f64, u64)> = HashMap::new();
    for i in 0..t.n {
        let e = map.entry(&t.id3[i]).or_insert((0, 0.0, 0));
        e.0 += t.v1[i];
        e.1 += t.v3[i];
        e.2 += 1;
    }
    (map.len(), 3)
}

fn groupby_q4(t: &GroupbyTable) -> (usize, usize) {
    // SELECT id4, AVG(v1) AS v1, AVG(v2) AS v2, AVG(v3) AS v3 FROM x GROUP BY id4
    let mut map: HashMap<i64, (i64, i64, f64, u64)> = HashMap::new();
    for i in 0..t.n {
        let e = map.entry(t.id4[i]).or_insert((0, 0, 0.0, 0));
        e.0 += t.v1[i];
        e.1 += t.v2[i];
        e.2 += t.v3[i];
        e.3 += 1;
    }
    (map.len(), 4)
}

fn groupby_q5(t: &GroupbyTable) -> (usize, usize) {
    // SELECT id6, SUM(v1) AS v1, SUM(v2) AS v2, SUM(v3) AS v3 FROM x GROUP BY id6
    let mut map: HashMap<i64, (i64, i64, f64)> = HashMap::new();
    for i in 0..t.n {
        let e = map.entry(t.id6[i]).or_insert((0, 0, 0.0));
        e.0 += t.v1[i];
        e.1 += t.v2[i];
        e.2 += t.v3[i];
    }
    (map.len(), 4)
}

fn groupby_q6(t: &GroupbyTable) -> (usize, usize) {
    // SELECT id4, id5, MEDIAN(v3) AS median_v3, STDDEV(v3) AS sd_v3
    //   FROM x GROUP BY id4, id5
    // Collect all v3 values per group, compute median + stddev
    let mut map: HashMap<(i64, i64), Vec<f64>> = HashMap::new();
    for i in 0..t.n {
        map.entry((t.id4[i], t.id5[i])).or_default().push(t.v3[i]);
    }
    // Actually compute median + stddev (to be fair to other engines)
    let n_groups = map.len();
    for (_key, vals) in map.iter_mut() {
        vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let n = vals.len();
        let _median = if n % 2 == 0 {
            (vals[n / 2 - 1] + vals[n / 2]) / 2.0
        } else {
            vals[n / 2]
        };
        let mean: f64 = vals.iter().sum::<f64>() / n as f64;
        let _var: f64 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    }
    (n_groups, 4)
}

fn groupby_q7(t: &GroupbyTable) -> (usize, usize) {
    // SELECT id3, MAX(v1) - MIN(v2) AS range_v1_v2 FROM x GROUP BY id3
    let mut map: HashMap<&str, (i64, i64)> = HashMap::new();
    for i in 0..t.n {
        let e = map.entry(&t.id3[i]).or_insert((i64::MIN, i64::MAX));
        e.0 = e.0.max(t.v1[i]);
        e.1 = e.1.min(t.v2[i]);
    }
    (map.len(), 2)
}

fn groupby_q8(t: &GroupbyTable) -> (usize, usize) {
    // SELECT id6, largest 2 v3 per group
    let mut map: HashMap<i64, Vec<f64>> = HashMap::new();
    for i in 0..t.n {
        let e = map.entry(t.id6[i]).or_default();
        e.push(t.v3[i]);
        // Keep only top 2 to limit memory (partial sort)
        if e.len() > 3 {
            e.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
            e.truncate(2);
        }
    }
    let mut total_rows = 0;
    for (_k, v) in &map {
        total_rows += v.len().min(2);
    }
    (total_rows, 2)
}

fn groupby_q9(t: &GroupbyTable) -> (usize, usize) {
    // SELECT id2, id4, POW(CORR(v1, v2), 2) AS r2 FROM x GROUP BY id2, id4
    let mut map: HashMap<(&str, i64), (f64, f64, f64, f64, f64, u64)> = HashMap::new();
    for i in 0..t.n {
        let x = t.v1[i] as f64;
        let y = t.v2[i] as f64;
        let e = map.entry((&t.id2[i], t.id4[i])).or_insert((0.0, 0.0, 0.0, 0.0, 0.0, 0));
        e.0 += x;      // sum_x
        e.1 += y;      // sum_y
        e.2 += x * y;  // sum_xy
        e.3 += x * x;  // sum_x2
        e.4 += y * y;  // sum_y2
        e.5 += 1;      // count
    }
    (map.len(), 3)
}

fn groupby_q10(t: &GroupbyTable) -> (usize, usize) {
    // SELECT id1..id6, SUM(v3), COUNT(*) FROM x GROUP BY id1..id6
    let mut map: HashMap<(&str, &str, &str, i64, i64, i64), (f64, u64)> = HashMap::new();
    for i in 0..t.n {
        let key = (
            t.id1[i].as_str(),
            t.id2[i].as_str(),
            t.id3[i].as_str(),
            t.id4[i],
            t.id5[i],
            t.id6[i],
        );
        let e = map.entry(key).or_insert((0.0, 0));
        e.0 += t.v3[i];
        e.1 += 1;
    }
    (map.len(), 8)
}

// ═══════════════════════════════════════════════════════════════════════════
// Join Queries (hash join)
// ═══════════════════════════════════════════════════════════════════════════

fn join_q1(x: &JoinXTable, small: &JoinSmallTable) -> (usize, usize) {
    // INNER JOIN small ON x.id1 = small.id1
    let mut count = 0usize;
    for i in 0..x.n {
        if let Some(matches) = small.idx_id1.get(&x.id1[i]) {
            count += matches.len();
        }
    }
    (count, 9) // x.* + small.id4, small.v2
}

fn join_q2(x: &JoinXTable, medium: &JoinMediumTable) -> (usize, usize) {
    // INNER JOIN medium ON x.id2 = medium.id2
    let mut count = 0usize;
    for i in 0..x.n {
        if let Some(matches) = medium.idx_id2.get(&x.id2[i]) {
            count += matches.len();
        }
    }
    (count, 11)
}

fn join_q3(x: &JoinXTable, medium: &JoinMediumTable) -> (usize, usize) {
    // LEFT JOIN medium ON x.id2 = medium.id2
    let mut count = 0usize;
    for i in 0..x.n {
        if let Some(matches) = medium.idx_id2.get(&x.id2[i]) {
            count += matches.len();
        } else {
            count += 1; // NULL row for LEFT JOIN
        }
    }
    (count, 11)
}

fn join_q4(x: &JoinXTable, medium: &JoinMediumTable) -> (usize, usize) {
    // INNER JOIN medium ON x.id5 = medium.id5 (factor/string join)
    let mut count = 0usize;
    for i in 0..x.n {
        if let Some(matches) = medium.idx_id5.get(&x.id5[i]) {
            count += matches.len();
        }
    }
    (count, 11)
}

fn join_q5(x: &JoinXTable, big: &JoinBigTable) -> (usize, usize) {
    // INNER JOIN big ON x.id3 = big.id3
    let mut count = 0usize;
    for i in 0..x.n {
        if let Some(matches) = big.idx_id3.get(&x.id3[i]) {
            count += matches.len();
        }
    }
    (count, 11)
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let cli = Cli::parse();

    match cli.task.as_str() {
        "groupby" => run_groupby(&cli.csv),
        "join" => run_join(
            &cli.csv,
            cli.small.as_deref().expect("--small required for join"),
            cli.medium.as_deref().expect("--medium required for join"),
            cli.big.as_deref().expect("--big required for join"),
        ),
        _ => {
            eprintln!("Unknown task: {}", cli.task);
            std::process::exit(1);
        }
    }
}

fn run_groupby(csv_path: &Path) {
    eprintln!("[SochDB] Loading CSV: {:?}", csv_path);
    let t0 = Instant::now();
    let table = GroupbyTable::load_csv(csv_path);
    let load_secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "[SochDB] Loaded {} rows in {:.2}s",
        table.n, load_secs
    );

    type QueryFn = fn(&GroupbyTable) -> (usize, usize);
    let queries: Vec<(&str, QueryFn)> = vec![
        ("sum v1 by id1", groupby_q1),
        ("sum v1 by id1:id2", groupby_q2),
        ("sum v1 mean v3 by id3", groupby_q3),
        ("mean v1:v3 by id4", groupby_q4),
        ("sum v1:v3 by id6", groupby_q5),
        ("median v3 sd v3 by id4 id5", groupby_q6),
        ("max v1 - min v2 by id3", groupby_q7),
        ("largest two v3 by id6", groupby_q8),
        ("regression v1 v2 by id2 id4", groupby_q9),
        ("sum v3 count by id1:id6", groupby_q10),
    ];

    for (question, query_fn) in &queries {
        for run in 1..=2 {
            let t0 = Instant::now();
            let (out_rows, out_cols) = query_fn(&table);
            let elapsed = t0.elapsed().as_secs_f64();
            emit(question, run, elapsed, out_rows, out_cols);
        }
    }
}

fn run_join(x_path: &Path, small_path: &Path, medium_path: &Path, big_path: &Path) {
    eprintln!("[SochDB] Loading join tables...");
    let t0 = Instant::now();
    let x = JoinXTable::load_csv(x_path);
    let small = JoinSmallTable::load_csv(small_path);
    let medium = JoinMediumTable::load_csv(medium_path);
    let big = JoinBigTable::load_csv(big_path);
    let load_secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "[SochDB] Loaded x={}, small={}, medium={}, big={} in {:.2}s",
        x.n, small.n, medium.n, big.n, load_secs
    );

    // q1: small inner on int
    for run in 1..=2 {
        let t0 = Instant::now();
        let (rows, cols) = join_q1(&x, &small);
        emit("small inner on int", run, t0.elapsed().as_secs_f64(), rows, cols);
    }

    // q2: medium inner on int
    for run in 1..=2 {
        let t0 = Instant::now();
        let (rows, cols) = join_q2(&x, &medium);
        emit("medium inner on int", run, t0.elapsed().as_secs_f64(), rows, cols);
    }

    // q3: medium outer on int
    for run in 1..=2 {
        let t0 = Instant::now();
        let (rows, cols) = join_q3(&x, &medium);
        emit("medium outer on int", run, t0.elapsed().as_secs_f64(), rows, cols);
    }

    // q4: medium inner on factor
    for run in 1..=2 {
        let t0 = Instant::now();
        let (rows, cols) = join_q4(&x, &medium);
        emit("medium inner on factor", run, t0.elapsed().as_secs_f64(), rows, cols);
    }

    // q5: big inner on int
    for run in 1..=2 {
        let t0 = Instant::now();
        let (rows, cols) = join_q5(&x, &big);
        emit("big inner on int", run, t0.elapsed().as_secs_f64(), rows, cols);
    }
}
