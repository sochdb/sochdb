// Copyright 2025 Sushanth (https://github.com/sushanthpy)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Embedding Integration Module
//!
//! This module provides a comprehensive embedding system for semantic search
//! across agent traces. It supports local inference (fastembed ONNX) and
//! external providers via the `llm` crate (OpenAI, Ollama, etc.).
//!
//! ## Architecture
//!
//! ```text
//! Text Input → Tokenization → Batching → Provider Selection →
//! Inference → Normalization → Quantization → Storage → Index Update
//! ```
//!
//! ## Providers
//!
//! - **Local (FastEmbed)**: Uses ONNX runtime for offline inference with models
//!   like all-MiniLM-L6-v2 (384 dimensions). Zero cost, works offline.
//!   Requires the `fastembed-embeddings` feature.
//!
//! - **LLM Providers**: Uses the `llm` crate for external providers like OpenAI,
//!   Ollama, Azure OpenAI, etc. Requires the `llm-embeddings` feature.
//!
//! - **Local (Default)**: Improved text-to-vector using TF-IDF-like features.
//!   Works offline without additional dependencies.
//!
//! - **Mock**: Deterministic fake embeddings for testing.
//!
//! ## Memory Efficiency
//!
//! With Product Quantization (PQ):
//! - 384-dim vector (1536 bytes) → 48 bytes (32x compression)
//! - 10M vectors fit in ~480 MB RAM
//!
//! ## Usage
//!
//! ```rust,ignore
//! use toondb_index::embedding::{EmbeddingProvider, LocalEmbeddingProvider, EmbeddingRegistry};
//!
//! // Create local provider (offline)
//! let local = LocalEmbeddingProvider::new(Default::default())?;
//!
//! // Embed single text
//! let vector = local.embed("Find traces with errors")?;
//!
//! // Batch embedding (more efficient)
//! let texts = vec!["query 1", "query 2", "query 3"];
//! let vectors = local.embed_batch(&texts)?;
//! ```
//!
//! ## Feature Flags
//!
//! - `local-embeddings`: Default, uses improved local algorithm
//! - `fastembed-embeddings`: Uses fastembed-rs for ONNX inference
//! - `llm-embeddings`: Uses llm crate for external providers (OpenAI, Ollama, etc.)

pub mod config;
#[cfg(feature = "fastembed-embeddings")]
pub mod fastembed;
pub mod hierarchical;
pub mod index_integration;
#[cfg(feature = "llm-embeddings")]
pub mod llm_provider;
pub mod model_manager;
pub mod normalize;
pub mod pipeline;
pub mod provider;
pub mod storage;

// Re-export main types
pub use config::{
    EmbeddingConfig, EmbeddingProviderType, EmbeddingTrigger, LlmProviderConfig, LocalModelConfig,
    ResourceLimits,
};
#[cfg(feature = "fastembed-embeddings")]
pub use fastembed::FastEmbedProvider;
pub use hierarchical::{
    AggregationStrategy, ContextualEmbedding, EmbeddingLevel, HierarchicalEmbedder,
};
pub use index_integration::{
    EmbeddingIntegration, IntegrationConfig, IntegrationError, SemanticSearchResult,
};
#[cfg(feature = "llm-embeddings")]
pub use llm_provider::LlmEmbeddingProvider;
pub use model_manager::{ModelError, ModelManager, ModelManifest};
pub use normalize::{normalize_l2, normalize_l2_simd};
pub use pipeline::{EmbeddingPipeline, EmbeddingRequest, PipelineConfig};
pub use provider::{
    EmbeddingError, EmbeddingProvider, EmbeddingRegistry, LocalEmbeddingConfig,
    LocalEmbeddingProvider, MockEmbeddingProvider,
};
pub use storage::{EmbeddingMetadata, EmbeddingStorage, EmbeddingStorageConfig};

/// Create an embedding provider from configuration
///
/// This is the main entry point for creating providers based on settings.
/// It automatically selects the best available provider based on features
/// and configuration.
pub fn create_provider_from_config(
    config: &EmbeddingConfig,
) -> Result<std::sync::Arc<dyn EmbeddingProvider>, EmbeddingError> {
    use std::sync::Arc;

    match &config.provider {
        EmbeddingProviderType::Disabled => Err(EmbeddingError::ConfigError(
            "Embeddings disabled".to_string(),
        )),

        EmbeddingProviderType::Local => {
            // Check if fastembed is available
            #[cfg(feature = "fastembed-embeddings")]
            {
                let provider = FastEmbedProvider::new(&config.local)?;
                return Ok(Arc::new(provider));
            }

            #[cfg(not(feature = "fastembed-embeddings"))]
            {
                // Use default local provider
                let local_config = LocalEmbeddingConfig {
                    model_name: config.local.model.clone(),
                    dimension: 384,
                    max_tokens: 512,
                    num_threads: config.local.max_threads,
                    batch_size: config.local.batch_size,
                };
                let provider = LocalEmbeddingProvider::new(local_config)?;
                Ok(Arc::new(provider))
            }
        }

        #[cfg(feature = "llm-embeddings")]
        EmbeddingProviderType::Llm(llm_config) => {
            let provider = LlmEmbeddingProvider::new(llm_config.clone())?;
            Ok(Arc::new(provider))
        }

        #[cfg(not(feature = "llm-embeddings"))]
        EmbeddingProviderType::Llm(_) => Err(EmbeddingError::ConfigError(
            "LLM embeddings feature not enabled. Compile with --features llm-embeddings"
                .to_string(),
        )),

        EmbeddingProviderType::LlmWithLocalFallback(llm_config) => {
            // Create registry with LLM as primary and local as fallback
            let mut registry = EmbeddingRegistry::new();

            // Add local provider
            #[cfg(feature = "fastembed-embeddings")]
            {
                let local = Arc::new(FastEmbedProvider::new(&config.local)?);
                registry.register("local", local)?;
            }

            #[cfg(not(feature = "fastembed-embeddings"))]
            {
                let local_config = LocalEmbeddingConfig::default();
                let local = Arc::new(LocalEmbeddingProvider::new(local_config)?);
                registry.register("local", local)?;
            }

            // Add LLM provider if feature enabled
            #[cfg(feature = "llm-embeddings")]
            {
                let llm = Arc::new(LlmEmbeddingProvider::new(llm_config.clone())?);
                registry.register("llm", llm)?;
                registry.set_default("llm")?;
                registry.set_fallback_chain(vec!["local".to_string()])?;
            }

            #[cfg(not(feature = "llm-embeddings"))]
            {
                let _ = llm_config;
                registry.set_default("local")?;
            }

            // Return wrapped registry
            Ok(Arc::new(RegistryProvider { registry }))
        }
    }
}

/// Wrapper to make EmbeddingRegistry implement EmbeddingProvider
struct RegistryProvider {
    registry: EmbeddingRegistry,
}

impl EmbeddingProvider for RegistryProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.registry.embed(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let provider = self.registry.get_default()?;
        provider.embed_batch(texts)
    }

    fn dimension(&self) -> usize {
        self.registry.dimension().unwrap_or(384)
    }

    fn max_tokens(&self) -> usize {
        self.registry
            .get_default()
            .map(|p| p.max_tokens())
            .unwrap_or(512)
    }

    fn provider_id(&self) -> &str {
        "registry"
    }

    fn is_offline(&self) -> bool {
        self.registry
            .get_default()
            .map(|p| p.is_offline())
            .unwrap_or(true)
    }
}
