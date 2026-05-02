// SPDX-License-Identifier: AGPL-3.0-or-later
// SochDB - LLM-Optimized Embedded Database
// Copyright (C) 2026 Sushanth Reddy Vanagala (https://github.com/sushanthpy)

//! # Subscription Service (Task 2 — Delta-evaluation engine)
//!
//! Provides real-time change notifications via CDC events streamed over gRPC.
//! Subscribers receive filtered change events as they occur, enabling
//! reactive patterns without polling.
//!
//! ## Architecture
//!
//! ```text
//! Client ──Subscribe()──▶ SubscriptionServer
//!                              │
//!                     creates CdcSubscriber
//!                              │
//!                     ┌────────▼────────┐
//!                     │   CdcLog        │ ◄── CdcEmitter (on tx commit)
//!                     │  (ring buffer)  │
//!                     └─────────────────┘
//!                              │
//!                     poll / next_batch
//!                              │
//!                     ┌────────▼────────┐
//!                     │  filter + eval  │ (table, op-type, WHERE predicate)
//!                     └────────┬────────┘
//!                              │
//!              stream SubscribeEvent ──▶ Client
//! ```

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, Mutex};
use tonic::{Request, Response, Status};

use crate::proto::{
    subscription_service_server::SubscriptionService,
    CancelSubscriptionRequest, CancelSubscriptionResponse, ListSubscriptionsRequest,
    ListSubscriptionsResponse, OperationType, SubscribeEvent, SubscribeRequest, SubscriptionInfo,
    WatchKeyEvent, WatchKeyRequest,
};

use sochdb_storage::cdc::{CdcLog, CdcOperation, CdcSubscriber};

/// Subscription server — streams CDC events to gRPC clients.
pub struct SubscriptionServer {
    /// CDC log shared with the storage engine
    cdc_log: Arc<CdcLog>,
    /// Active subscriptions (id → metadata)
    active_subs: Arc<Mutex<HashMap<String, SubMeta>>>,
    /// Subscription ID counter
    next_sub_id: AtomicU64,
}

struct SubMeta {
    namespace: String,
    tables: Vec<String>,
    start_sequence: u64,
    current_sequence: Arc<AtomicU64>,
    created_at_us: u64,
    /// Cancellation signal
    cancel_tx: mpsc::Sender<()>,
}

type GrpcStream<T> =
    Pin<Box<dyn tokio_stream::Stream<Item = Result<T, Status>> + Send + 'static>>;

impl SubscriptionServer {
    /// Create a new subscription server backed by a CDC log.
    pub fn new(cdc_log: Arc<CdcLog>) -> Self {
        Self {
            cdc_log,
            active_subs: Arc::new(Mutex::new(HashMap::new())),
            next_sub_id: AtomicU64::new(1),
        }
    }

    /// Convert a CdcOperation to proto OperationType.
    fn op_to_proto(op: &CdcOperation) -> i32 {
        match op {
            CdcOperation::Insert { .. } => OperationType::OperationInsert as i32,
            CdcOperation::Update { .. } => OperationType::OperationUpdate as i32,
            CdcOperation::Delete { .. } => OperationType::OperationDelete as i32,
            CdcOperation::SchemaChange { .. } => OperationType::OperationSchemaChange as i32,
        }
    }

    /// Convert a CdcOperation to proto OperationType enum.
    fn op_type(op: &CdcOperation) -> OperationType {
        match op {
            CdcOperation::Insert { .. } => OperationType::OperationInsert,
            CdcOperation::Update { .. } => OperationType::OperationUpdate,
            CdcOperation::Delete { .. } => OperationType::OperationDelete,
            CdcOperation::SchemaChange { .. } => OperationType::OperationSchemaChange,
        }
    }

    /// Get the current timestamp in microseconds.
    fn now_us() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }

    /// Generate a unique subscription ID.
    fn gen_sub_id(&self) -> String {
        let id = self.next_sub_id.fetch_add(1, Ordering::Relaxed);
        format!("sub-{}", id)
    }

    /// Create a `SubscribeEvent` proto from a CDC event.
    fn cdc_to_proto(event: &sochdb_storage::cdc::CdcEvent) -> SubscribeEvent {
        let (after_value, before_value, ddl) = match &event.operation {
            CdcOperation::Insert { after } => (after.clone(), vec![], String::new()),
            CdcOperation::Update { before, after } => {
                (after.clone(), before.clone().unwrap_or_default(), String::new())
            }
            CdcOperation::Delete { before } => {
                (vec![], before.clone().unwrap_or_default(), String::new())
            }
            CdcOperation::SchemaChange { ddl } => (vec![], vec![], ddl.clone()),
        };

        SubscribeEvent {
            sequence: event.sequence,
            timestamp_us: event.timestamp_us,
            txn_id: event.txn_id,
            table: event.table.clone(),
            key: event.key.clone(),
            operation: Self::op_to_proto(&event.operation),
            after_value,
            before_value,
            ddl,
        }
    }

    /// Build a set of allowed operation types from request filters.
    fn op_filter(ops: &[i32]) -> Option<Vec<OperationType>> {
        if ops.is_empty() {
            return None; // no filter → allow all
        }
        let filtered: Vec<OperationType> = ops
            .iter()
            .filter_map(|&v| match v {
                1 => Some(OperationType::OperationInsert),
                2 => Some(OperationType::OperationUpdate),
                3 => Some(OperationType::OperationDelete),
                4 => Some(OperationType::OperationSchemaChange),
                _ => None,
            })
            .collect();
        if filtered.is_empty() {
            None
        } else {
            Some(filtered)
        }
    }
}

#[tonic::async_trait]
impl SubscriptionService for SubscriptionServer {
    type SubscribeStream = GrpcStream<SubscribeEvent>;
    type WatchKeyStream = GrpcStream<WatchKeyEvent>;

    async fn subscribe(
        &self,
        request: Request<SubscribeRequest>,
    ) -> Result<Response<Self::SubscribeStream>, Status> {
        let req = request.into_inner();

        // Create CDC subscriber
        let mut subscriber = if req.start_sequence > 0 {
            CdcSubscriber::new(self.cdc_log.clone(), req.start_sequence)
        } else {
            CdcSubscriber::from_latest(self.cdc_log.clone())
        };

        // Apply table filter
        if !req.tables.is_empty() {
            subscriber = subscriber.with_tables(req.tables.clone());
        }

        let op_filter = Self::op_filter(&req.operations);
        let batch_size = if req.batch_size > 0 {
            req.batch_size as usize
        } else {
            64
        };

        // Set up cancellation
        let (cancel_tx, mut cancel_rx) = mpsc::channel::<()>(1);
        let sub_id = self.gen_sub_id();
        let current_seq = Arc::new(AtomicU64::new(req.start_sequence));

        // Register subscription metadata
        {
            let mut subs = self.active_subs.lock().await;
            subs.insert(
                sub_id.clone(),
                SubMeta {
                    namespace: req.namespace.clone(),
                    tables: req.tables.clone(),
                    start_sequence: req.start_sequence,
                    current_sequence: current_seq.clone(),
                    created_at_us: Self::now_us(),
                    cancel_tx,
                },
            );
        }

        // Streaming channel
        let (tx, rx) = mpsc::channel(256);
        let active_subs = self.active_subs.clone();
        let sub_id_for_cleanup = sub_id.clone();

        // Spawn streaming task
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = cancel_rx.recv() => {
                        // Subscription cancelled
                        break;
                    }
                    should_break = async {
                        // Poll for events with timeout
                        let events = match subscriber.next_batch(batch_size, Duration::from_millis(100)) {
                            Ok(events) => events,
                            Err(_) => return false, // timeout or overrun, retry
                        };

                        for event in events {
                            // Apply operation type filter
                            if let Some(ref filter) = op_filter {
                                let event_op = SubscriptionServer::op_type(&event.operation);
                                if !filter.contains(&event_op) {
                                    continue;
                                }
                            }

                            let proto_event = SubscriptionServer::cdc_to_proto(&event);
                            current_seq.store(event.sequence, Ordering::Relaxed);

                            if tx.send(Ok(proto_event)).await.is_err() {
                                // Client disconnected
                                return true; // signal break
                            }
                        }
                        false // continue
                    } => {
                        if should_break {
                            break;
                        }
                    }
                }
            }

            // Clean up subscription on exit
            let mut subs = active_subs.lock().await;
            subs.remove(&sub_id_for_cleanup);
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream) as Self::SubscribeStream))
    }

    async fn watch_key(
        &self,
        request: Request<WatchKeyRequest>,
    ) -> Result<Response<Self::WatchKeyStream>, Status> {
        let req = request.into_inner();
        let target_table = req.table.clone();
        let target_key = req.key.clone();

        let mut subscriber = CdcSubscriber::from_latest(self.cdc_log.clone());
        if !target_table.is_empty() {
            subscriber = subscriber.with_tables(vec![target_table.clone()]);
        }

        let (tx, rx) = mpsc::channel(64);

        tokio::spawn(async move {
            loop {
                let events = match subscriber.next_batch(32, Duration::from_millis(200)) {
                    Ok(events) => events,
                    Err(_) => continue, // timeout or overrun, retry
                };
                for event in events {
                    // Filter to exact key
                    if !target_key.is_empty() && event.key != target_key {
                        continue;
                    }

                    let (after_value, before_value) = match &event.operation {
                        CdcOperation::Insert { after } => (after.clone(), vec![]),
                        CdcOperation::Update { before, after } => {
                            (after.clone(), before.clone().unwrap_or_default())
                        }
                        CdcOperation::Delete { before } => {
                            (vec![], before.clone().unwrap_or_default())
                        }
                        CdcOperation::SchemaChange { .. } => continue,
                    };

                    let watch_event = WatchKeyEvent {
                        sequence: event.sequence,
                        timestamp_us: event.timestamp_us,
                        operation: SubscriptionServer::op_to_proto(&event.operation),
                        after_value,
                        before_value,
                    };

                    if tx.send(Ok(watch_event)).await.is_err() {
                        break;
                    }
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream) as Self::WatchKeyStream))
    }

    async fn list_subscriptions(
        &self,
        request: Request<ListSubscriptionsRequest>,
    ) -> Result<Response<ListSubscriptionsResponse>, Status> {
        let req = request.into_inner();
        let subs = self.active_subs.lock().await;

        let infos: Vec<SubscriptionInfo> = subs
            .iter()
            .filter(|(_, meta)| req.namespace.is_empty() || meta.namespace == req.namespace)
            .map(|(id, meta)| SubscriptionInfo {
                subscription_id: id.clone(),
                namespace: meta.namespace.clone(),
                tables: meta.tables.clone(),
                start_sequence: meta.start_sequence,
                current_sequence: meta.current_sequence.load(Ordering::Relaxed),
                created_at_us: meta.created_at_us,
            })
            .collect();

        Ok(Response::new(ListSubscriptionsResponse {
            subscriptions: infos,
        }))
    }

    async fn cancel_subscription(
        &self,
        request: Request<CancelSubscriptionRequest>,
    ) -> Result<Response<CancelSubscriptionResponse>, Status> {
        let req = request.into_inner();
        let mut subs = self.active_subs.lock().await;

        if let Some(meta) = subs.remove(&req.subscription_id) {
            let _ = meta.cancel_tx.send(()).await;
            Ok(Response::new(CancelSubscriptionResponse {
                success: true,
                error: String::new(),
            }))
        } else {
            Ok(Response::new(CancelSubscriptionResponse {
                success: false,
                error: format!("Subscription '{}' not found", req.subscription_id),
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sochdb_storage::cdc::{CdcConfig, CdcEmitter, CdcLog, CdcOperation};

    fn make_cdc_log() -> Arc<CdcLog> {
        CdcLog::new(CdcConfig {
            enabled: true,
            capacity: 1024,
        })
    }

    #[test]
    fn test_op_to_proto() {
        assert_eq!(
            SubscriptionServer::op_to_proto(&CdcOperation::Insert {
                after: vec![1, 2, 3]
            }),
            OperationType::OperationInsert as i32
        );
        assert_eq!(
            SubscriptionServer::op_to_proto(&CdcOperation::Update {
                before: None,
                after: vec![]
            }),
            OperationType::OperationUpdate as i32
        );
        assert_eq!(
            SubscriptionServer::op_to_proto(&CdcOperation::Delete { before: None }),
            OperationType::OperationDelete as i32
        );
        assert_eq!(
            SubscriptionServer::op_to_proto(&CdcOperation::SchemaChange {
                ddl: "ALTER TABLE x".into()
            }),
            OperationType::OperationSchemaChange as i32
        );
    }

    #[test]
    fn test_cdc_to_proto_insert() {
        let event = sochdb_storage::cdc::CdcEvent {
            sequence: 42,
            timestamp_us: 1000000,
            txn_id: 7,
            table: "users".into(),
            key: vec![1, 2],
            operation: CdcOperation::Insert {
                after: vec![10, 20, 30],
            },
        };

        let proto = SubscriptionServer::cdc_to_proto(&event);
        assert_eq!(proto.sequence, 42);
        assert_eq!(proto.timestamp_us, 1000000);
        assert_eq!(proto.txn_id, 7);
        assert_eq!(proto.table, "users");
        assert_eq!(proto.key, vec![1, 2]);
        assert_eq!(proto.operation, OperationType::OperationInsert as i32);
        assert_eq!(proto.after_value, vec![10, 20, 30]);
        assert!(proto.before_value.is_empty());
    }

    #[test]
    fn test_cdc_to_proto_update() {
        let event = sochdb_storage::cdc::CdcEvent {
            sequence: 43,
            timestamp_us: 2000000,
            txn_id: 8,
            table: "orders".into(),
            key: vec![3],
            operation: CdcOperation::Update {
                before: Some(vec![1, 2]),
                after: vec![3, 4],
            },
        };

        let proto = SubscriptionServer::cdc_to_proto(&event);
        assert_eq!(proto.operation, OperationType::OperationUpdate as i32);
        assert_eq!(proto.after_value, vec![3, 4]);
        assert_eq!(proto.before_value, vec![1, 2]);
    }

    #[test]
    fn test_cdc_to_proto_schema_change() {
        let event = sochdb_storage::cdc::CdcEvent {
            sequence: 1,
            timestamp_us: 100,
            txn_id: 1,
            table: "".into(),
            key: vec![],
            operation: CdcOperation::SchemaChange {
                ddl: "CREATE TABLE foo".into(),
            },
        };

        let proto = SubscriptionServer::cdc_to_proto(&event);
        assert_eq!(proto.operation, OperationType::OperationSchemaChange as i32);
        assert_eq!(proto.ddl, "CREATE TABLE foo");
    }

    #[test]
    fn test_op_filter_empty() {
        assert!(SubscriptionServer::op_filter(&[]).is_none());
    }

    #[test]
    fn test_op_filter_specific() {
        let filter = SubscriptionServer::op_filter(&[1, 3]).unwrap();
        assert_eq!(filter.len(), 2);
        assert!(filter.contains(&OperationType::OperationInsert));
        assert!(filter.contains(&OperationType::OperationDelete));
    }

    #[test]
    fn test_gen_sub_id() {
        let log = make_cdc_log();
        let server = SubscriptionServer::new(log);
        let id1 = server.gen_sub_id();
        let id2 = server.gen_sub_id();
        assert_ne!(id1, id2);
        assert!(id1.starts_with("sub-"));
    }

    #[tokio::test]
    async fn test_list_subscriptions_empty() {
        let log = make_cdc_log();
        let server = SubscriptionServer::new(log);

        let resp = server
            .list_subscriptions(Request::new(ListSubscriptionsRequest {
                namespace: String::new(),
            }))
            .await
            .unwrap();

        assert!(resp.into_inner().subscriptions.is_empty());
    }

    #[tokio::test]
    async fn test_cancel_nonexistent_subscription() {
        let log = make_cdc_log();
        let server = SubscriptionServer::new(log);

        let resp = server
            .cancel_subscription(Request::new(CancelSubscriptionRequest {
                subscription_id: "no-such-sub".into(),
            }))
            .await
            .unwrap();

        let inner = resp.into_inner();
        assert!(!inner.success);
        assert!(inner.error.contains("not found"));
    }
}
