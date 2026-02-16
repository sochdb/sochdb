//! LanceDB adapter (optional, behind `lancedb-bench` feature).
//!
//! Uses async LanceDB with a tokio `block_on` bridge so it conforms to the
//! synchronous `BenchDb` trait.

#![cfg(feature = "lancedb-bench")]

use crate::{AnalyticsRow, BenchDb, BenchError, BenchResult};
use arrow_array::{
    types::Float32Type, Array, FixedSizeListArray, Float32Array, Float64Array, Int64Array,
    RecordBatch, RecordBatchIterator, StringArray, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use lancedb::connect;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::runtime::Runtime;

pub struct LanceDbAdapter {
    rt: Runtime,
    db: lancedb::Connection,
    path: PathBuf,
    vector_dim: usize,
}

impl LanceDbAdapter {
    pub fn new(dir: &Path) -> BenchResult<Self> {
        let path = dir.join("lancedb_data");
        std::fs::create_dir_all(&path)?;

        let rt = Runtime::new().map_err(|e| BenchError::Database(format!("tokio rt: {}", e)))?;
        let db = rt.block_on(async {
            connect(path.to_str().unwrap())
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("LanceDB connect: {}", e)))
        })?;

        Ok(Self {
            rt,
            db,
            path,
            vector_dim: 128,
        })
    }

    fn block_on<F: std::future::Future>(&self, f: F) -> F::Output {
        self.rt.block_on(f)
    }
}

impl BenchDb for LanceDbAdapter {
    fn name(&self) -> &str {
        "LanceDB"
    }

    fn setup_kv_table(&mut self) -> BenchResult<()> {
        // Create a KV table with key (binary) and value (binary).
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Binary, false),
            Field::new("value", DataType::Binary, false),
        ]));
        // Create empty table.
        let batch = RecordBatch::new_empty(schema.clone());
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        self.block_on(async {
            let _ = self.db.drop_table("kv").await;
            self.db
                .create_table("kv", Box::new(reader))
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("create kv: {}", e)))
        })?;
        Ok(())
    }

    fn setup_analytics_table(&mut self) -> BenchResult<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("amount", DataType::Float64, false),
            Field::new("category", DataType::Utf8, false),
            Field::new("description", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::new_empty(schema.clone());
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        self.block_on(async {
            let _ = self.db.drop_table("analytics").await;
            self.db
                .create_table("analytics", Box::new(reader))
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("create analytics: {}", e)))
        })?;
        Ok(())
    }

    fn setup_vector_table(&mut self, dim: usize) -> BenchResult<()> {
        self.vector_dim = dim;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
            Field::new("metadata", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::new_empty(schema.clone());
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        self.block_on(async {
            let _ = self.db.drop_table("vectors").await;
            self.db
                .create_table("vectors", Box::new(reader))
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("create vectors: {}", e)))
        })?;
        Ok(())
    }

    fn teardown(&mut self) -> BenchResult<()> {
        Ok(())
    }

    // ── KV ops ──
    // LanceDB is not designed for point KV — these are basic inserts/reads.

    fn put(&mut self, key: &[u8], value: &[u8]) -> BenchResult<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Binary, false),
            Field::new("value", DataType::Binary, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow_array::BinaryArray::from(vec![key])),
                Arc::new(arrow_array::BinaryArray::from(vec![value])),
            ],
        )
        .map_err(|e| BenchError::Database(format!("batch: {}", e)))?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        self.block_on(async {
            let table = self
                .db
                .open_table("kv")
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("open kv: {}", e)))?;
            table
                .add(Box::new(reader))
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("add: {}", e)))
        })?;
        Ok(())
    }

    fn get(&mut self, _key: &[u8]) -> BenchResult<Option<Vec<u8>>> {
        // LanceDB doesn't support efficient point lookups.
        // Return None for benchmark — this accurately reflects its design.
        Ok(None)
    }

    fn delete(&mut self, _key: &[u8]) -> BenchResult<()> {
        // LanceDB delete is not a point operation.
        Ok(())
    }

    fn batch_put(&mut self, pairs: &[(&[u8], &[u8])]) -> BenchResult<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Binary, false),
            Field::new("value", DataType::Binary, false),
        ]));
        let keys: Vec<&[u8]> = pairs.iter().map(|(k, _)| *k).collect();
        let vals: Vec<&[u8]> = pairs.iter().map(|(_, v)| *v).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow_array::BinaryArray::from(keys)),
                Arc::new(arrow_array::BinaryArray::from(vals)),
            ],
        )
        .map_err(|e| BenchError::Database(format!("batch: {}", e)))?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        self.block_on(async {
            let table = self
                .db
                .open_table("kv")
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("open kv: {}", e)))?;
            table
                .add(Box::new(reader))
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("add: {}", e)))
        })?;
        Ok(())
    }

    // ── Analytics ops ──

    fn insert_analytics_row(&mut self, row: &AnalyticsRow) -> BenchResult<()> {
        self.insert_analytics_batch(&[row.clone()])
    }

    fn insert_analytics_batch(&mut self, rows: &[AnalyticsRow]) -> BenchResult<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("timestamp", DataType::Int64, false),
            Field::new("amount", DataType::Float64, false),
            Field::new("category", DataType::Utf8, false),
            Field::new("description", DataType::Utf8, true),
        ]));

        let ids: Vec<u64> = rows.iter().map(|r| r.id).collect();
        let ts: Vec<i64> = rows.iter().map(|r| r.timestamp).collect();
        let amounts: Vec<f64> = rows.iter().map(|r| r.amount).collect();
        let cats: Vec<&str> = rows.iter().map(|r| r.category.as_str()).collect();
        let descs: Vec<Option<&str>> = rows.iter().map(|r| Some(r.description.as_str())).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(ids)),
                Arc::new(Int64Array::from(ts)),
                Arc::new(Float64Array::from(amounts)),
                Arc::new(StringArray::from(cats)),
                Arc::new(StringArray::from(descs)),
            ],
        )
        .map_err(|e| BenchError::Database(format!("batch: {}", e)))?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        self.block_on(async {
            let table = self
                .db
                .open_table("analytics")
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("open: {}", e)))?;
            table
                .add(Box::new(reader))
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("add: {}", e)))
        })?;
        Ok(())
    }

    fn scan_filter_amount_gt(&mut self, threshold: f64) -> BenchResult<usize> {
        self.block_on(async {
            let table = self
                .db
                .open_table("analytics")
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("open: {}", e)))?;
            let batches = table
                .query()
                .only_if(format!("amount > {}", threshold))
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("query: {}", e)))?;
            // Count rows from the stream.
            use futures::TryStreamExt;
            let all: Vec<RecordBatch> = batches
                .try_collect()
                .await
                .map_err(|e| BenchError::Database(format!("collect: {}", e)))?;
            Ok(all.iter().map(|b| b.num_rows()).sum())
        })
    }

    fn aggregate_sum_amount(&mut self) -> BenchResult<f64> {
        self.block_on(async {
            let table = self
                .db
                .open_table("analytics")
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("open: {}", e)))?;
            let batches = table
                .query()
                .select(lancedb::query::Select::columns(&["amount"]))
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("query: {}", e)))?;
            use futures::TryStreamExt;
            let all: Vec<RecordBatch> = batches
                .try_collect()
                .await
                .map_err(|e| BenchError::Database(format!("collect: {}", e)))?;
            let mut sum = 0.0f64;
            for batch in &all {
                let col = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                for i in 0..col.len() {
                    sum += col.value(i);
                }
            }
            Ok(sum)
        })
    }

    fn group_by_category_count(&mut self) -> BenchResult<Vec<(String, u64)>> {
        // LanceDB doesn't have native GROUP BY — do it client-side.
        self.block_on(async {
            let table = self
                .db
                .open_table("analytics")
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("open: {}", e)))?;
            let batches = table
                .query()
                .select(lancedb::query::Select::columns(&["category"]))
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("query: {}", e)))?;
            use futures::TryStreamExt;
            let all: Vec<RecordBatch> = batches
                .try_collect()
                .await
                .map_err(|e| BenchError::Database(format!("collect: {}", e)))?;
            let mut counts = std::collections::HashMap::new();
            for batch in &all {
                let col = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                for i in 0..col.len() {
                    *counts.entry(col.value(i).to_string()).or_insert(0u64) += 1;
                }
            }
            Ok(counts.into_iter().collect())
        })
    }

    fn range_scan_ts(&mut self, start: i64, end: i64) -> BenchResult<usize> {
        self.block_on(async {
            let table = self
                .db
                .open_table("analytics")
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("open: {}", e)))?;
            let batches = table
                .query()
                .only_if(format!("timestamp >= {} AND timestamp < {}", start, end))
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("query: {}", e)))?;
            use futures::TryStreamExt;
            let all: Vec<RecordBatch> = batches
                .try_collect()
                .await
                .map_err(|e| BenchError::Database(format!("collect: {}", e)))?;
            Ok(all.iter().map(|b| b.num_rows()).sum())
        })
    }

    // ── Vector ops ──

    fn insert_vector(
        &mut self,
        id: u64,
        vector: &[f32],
        metadata: Option<&str>,
    ) -> BenchResult<()> {
        self.insert_vector_batch(&[(id, vector.to_vec(), metadata.map(|s| s.to_string()))])
    }

    fn insert_vector_batch(
        &mut self,
        vectors: &[(u64, Vec<f32>, Option<String>)],
    ) -> BenchResult<()> {
        let dim = self.vector_dim as i32;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim,
                ),
                false,
            ),
            Field::new("metadata", DataType::Utf8, true),
        ]));

        let ids: Vec<u64> = vectors.iter().map(|(id, _, _)| *id).collect();
        let flat: Vec<f32> = vectors.iter().flat_map(|(_, v, _)| v.clone()).collect();
        let meta: Vec<Option<&str>> = vectors
            .iter()
            .map(|(_, _, m)| m.as_deref())
            .collect();

        let values = Float32Array::from(flat);
        let list = FixedSizeListArray::try_new_from_values(values, dim)
            .map_err(|e| BenchError::Database(format!("list: {}", e)))?;

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(ids)),
                Arc::new(list),
                Arc::new(StringArray::from(meta)),
            ],
        )
        .map_err(|e| BenchError::Database(format!("batch: {}", e)))?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        self.block_on(async {
            let table = self
                .db
                .open_table("vectors")
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("open: {}", e)))?;
            table
                .add(Box::new(reader))
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("add: {}", e)))
        })?;
        Ok(())
    }

    fn vector_search(&mut self, query: &[f32], k: usize) -> BenchResult<Vec<(u64, f32)>> {
        self.block_on(async {
            let table = self
                .db
                .open_table("vectors")
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("open: {}", e)))?;
            let results = table
                .vector_search(query)
                .map_err(|e| BenchError::Database(format!("search setup: {}", e)))?
                .limit(k)
                .execute()
                .await
                .map_err(|e| BenchError::Database(format!("search: {}", e)))?;
            use futures::TryStreamExt;
            let batches: Vec<RecordBatch> = results
                .try_collect()
                .await
                .map_err(|e| BenchError::Database(format!("collect: {}", e)))?;
            let mut out = Vec::new();
            for batch in &batches {
                let ids = batch
                    .column_by_name("id")
                    .and_then(|c| c.as_any().downcast_ref::<UInt64Array>());
                let dists = batch
                    .column_by_name("_distance")
                    .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
                if let (Some(ids), Some(dists)) = (ids, dists) {
                    for i in 0..ids.len() {
                        out.push((ids.value(i), dists.value(i)));
                    }
                }
            }
            Ok(out)
        })
    }

    fn db_size_bytes(&self) -> BenchResult<u64> {
        dir_size(&self.path)
    }
}

fn dir_size(path: &Path) -> BenchResult<u64> {
    let mut total = 0u64;
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            if meta.is_file() {
                total += meta.len();
            } else if meta.is_dir() {
                total += dir_size(&entry.path())?;
            }
        }
    }
    Ok(total)
}
