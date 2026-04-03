//! OpenRouter embeddings provider.

use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::errors::ClientError;
use crate::http::HttpClient;
use crate::models::{EmbedRequest, EmbedResponse, Usage};
use crate::providers::EmbeddingProvider;

const OPENROUTER_API_URL: &str = "https://openrouter.ai/api/v1/embeddings";

/// OpenRouter embeddings provider.
#[derive(Debug)]
pub struct OpenRouterProvider {
    api_key: String,
    http_client: HttpClient,
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider with the given API key.
    pub fn new(api_key: String, http_client: HttpClient) -> Self {
        Self {
            api_key,
            http_client,
        }
    }

    /// Create a new OpenRouter provider from environment variable.
    pub fn from_env(http_client: HttpClient) -> Result<Self, ClientError> {
        let api_key =
            std::env::var("OPENROUTER_API_KEY").map_err(|_| ClientError::MissingApiKey {
                provider: "openrouter".to_string(),
            })?;
        Ok(Self::new(api_key, http_client))
    }
}

/// OpenRouter API request body (OpenAI-compatible).
#[derive(Debug, Serialize)]
struct OpenRouterEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

/// OpenRouter API response.
#[derive(Debug, Deserialize)]
struct OpenRouterEmbeddingResponse {
    data: Vec<OpenRouterEmbedding>,
    model: String,
    usage: OpenRouterUsage,
}

#[derive(Debug, Deserialize)]
struct OpenRouterEmbedding {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct OpenRouterUsage {
    total_tokens: u64,
}

/// OpenRouter API error response.
#[derive(Debug, Deserialize)]
struct OpenRouterErrorResponse {
    error: OpenRouterError,
}

#[derive(Debug, Deserialize)]
struct OpenRouterError {
    message: String,
}

#[async_trait]
impl EmbeddingProvider for OpenRouterProvider {
    fn name(&self) -> &'static str {
        "openrouter"
    }

    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, ClientError> {
        debug!(
            model = %request.model,
            inputs = request.inputs.len(),
            "Sending OpenRouter embedding request"
        );

        // OpenRouter doesn't support input_type
        let input_type_value = request.input_type;
        if input_type_value.is_some() {
            debug!("OpenRouter doesn't use input_type parameter, ignoring");
        }

        let input_count = request.inputs.len();
        let body = OpenRouterEmbeddingRequest {
            model: &request.model,
            input: &request.inputs,
            dimensions: request.dimensions,
        };

        let body_json = serde_json::to_string(&body)?;
        let api_key = request.api_key.as_ref().unwrap_or(&self.api_key).clone();

        let start = Instant::now();
        let response = self
            .http_client
            .send_with_retry(|client| {
                client
                    .post(OPENROUTER_API_URL)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .body(body_json.clone())
            })
            .await?;

        let status = response.status().as_u16();
        let response_text = response.text().await?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if status != 200 {
            if let Ok(error_response) =
                serde_json::from_str::<OpenRouterErrorResponse>(&response_text)
            {
                return Err(ClientError::Api {
                    status,
                    message: error_response.error.message,
                });
            }
            return Err(ClientError::Api {
                status,
                message: response_text,
            });
        }

        let or_response: OpenRouterEmbeddingResponse = serde_json::from_str(&response_text)?;

        // Sort embeddings by index
        let mut embeddings: Vec<_> = or_response.data.into_iter().collect();
        embeddings.sort_by_key(|e| e.index);

        let embedding_vectors: Vec<Vec<f32>> =
            embeddings.into_iter().map(|e| e.embedding).collect();

        let dimensions = embedding_vectors.first().map(|e| e.len()).unwrap_or(0);
        let total_tokens = or_response.usage.total_tokens;

        let cost = calculate_cost(&request.model, total_tokens);

        Ok(EmbedResponse {
            embeddings: embedding_vectors,
            model: or_response.model,
            provider: "openrouter".to_string(),
            dimensions,
            input_count,
            input_type: input_type_value,
            latency_ms,
            usage: Usage {
                tokens: total_tokens,
                cost,
            },
        })
    }
}

fn calculate_cost(model: &str, tokens: u64) -> Option<f64> {
    // OpenRouter embedding pricing (per 1M tokens)
    let price_per_million = match model {
        m if m.contains("text-embedding-3-small") => 0.02,
        m if m.contains("text-embedding-3-large") => 0.13,
        m if m.contains("text-embedding-ada-002") => 0.10,
        m if m.contains("gemini-embedding-001") => 0.15,
        m if m.contains("codestral-embed") => 0.15,
        m if m.contains("mistral-embed") => 0.10,
        m if m.contains("pplx-embed-v1-4b") => 0.03,
        m if m.contains("pplx-embed-v1-0.6b") => 0.004,
        m if m.contains("qwen3-embedding-8b") => 0.01,
        m if m.contains("qwen3-embedding-4b") => 0.02,
        m if m.contains("nemotron-embed") => 0.0,
        m if m.contains("bge-large") || m.contains("gte-large") || m.contains("e5-large") => 0.01,
        _ => 0.005, // Default for small/base open-source models
    };

    Some((tokens as f64 / 1_000_000.0) * price_per_million)
}
