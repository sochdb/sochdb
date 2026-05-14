# PR 11: fix/kernel-atomic-claim-queue-id

## Title

fix(kernel): add `queue_id` to `ClaimToken` and fix `release()` to use correct key

## What Happened

The `AtomicClaimManager::release()` method was using `token.owner` (a worker identifier like "worker1") as the queue lookup key, instead of `token.queue_id`. This caused `release()` to always fail silently because the claims HashMap is keyed by queue_id, not owner.

Additionally, `ClaimToken` lacked a `queue_id` field entirely, making it impossible to know which queue a claim belonged to.

## Lines that are causing errors

```
sochdb-kernel/src/atomic_claim.rs:360 —  let lock = self.get_claim_lock(&token.owner, &token.task_id);
sochdb-kernel/src/atomic_claim.rs:365 —  if let Some(queue_claims) = claims.get_mut(&token.owner) {
```

The `ClaimToken` struct at lines 107-123 lacked a `queue_id` field.

## Fix

1. Added `queue_id: String` field to `ClaimToken`
2. Updated `ClaimEntry::to_token()` to accept and include `queue_id`
3. Fixed `release()` to use `token.queue_id` instead of `token.owner`
4. Updated all call sites of `to_token()` to pass `queue_id`
5. Simplified `LeaseManager::release()` to delegate to `AtomicClaimManager::release()`

```diff
 /// A token proving ownership of a claimed task
 pub struct ClaimToken {
+    /// Queue containing the task
+    pub queue_id: String,
     /// Task being claimed
     pub task_id: String,
     /// Owner identity
     pub owner: String,
     // ...
 }

     fn to_token(&self, queue_id: &str, task_id: &str) -> ClaimToken {
         ClaimToken {
+            queue_id: queue_id.to_string(),
             task_id: task_id.to_string(),
             // ...
         }
     }

     pub fn release(&self, token: &ClaimToken) -> Result<(), String> {
-        let lock = self.get_claim_lock(&token.owner, &token.task_id);
+        let lock = self.get_claim_lock(&token.queue_id, &token.task_id);
         let _guard = lock.lock();
         
         let mut claims = self.claims.write();
         
-        if let Some(queue_claims) = claims.get_mut(&token.owner) {
+        if let Some(queue_claims) = claims.get_mut(&token.queue_id) {
```

## Impact

| Issue | Severity | Description |
|-------|----------|-------------|
| Claim leakage | High | Tasks never removed after processing |
| Memory growth | High | HashMap entries accumulate forever |
| Queue starvation | High | Other workers can't process claimed tasks |
| Test workaround | Medium | Tests used `cleanup_expired()` instead of proper `release()` |

## Validation

```bash
cargo test --package sochdb-kernel atomic_claim
# test result: ok. 9 passed; 0 failed

cargo build --package sochdb-kernel
# Finished `dev` profile
```

New tests added:
- `test_claim_release_wrong_queue`: Verifies token has correct `queue_id`
- `test_multiple_queue_isolation`: Verifies same task_id in different queues are independent