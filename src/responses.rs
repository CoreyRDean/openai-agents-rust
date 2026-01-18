use crate::config::Config;
use crate::error::AgentError;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub input: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesResponse {
    pub id: Option<String>,
    #[serde(default)]
    pub output: Vec<Value>,
}

pub struct ResponsesClient {
    http: Client,
    config: Arc<Config>,
}

impl ResponsesClient {
    pub fn new(config: Config) -> Self {
        let http = Client::builder()
            .user_agent("openai-agents-rust")
            .build()
            .expect("Failed to build reqwest client");
        Self {
            http,
            config: Arc::new(config),
        }
    }

    fn url(&self) -> String {
        format!("{}/responses", self.config.base_url.trim_end_matches('/'))
    }

    pub async fn create(&self, req: &ResponsesRequest) -> Result<ResponsesResponse, AgentError> {
        let mut rb = self.http.post(self.url());
        if !self.config.api_key.is_empty() {
            rb = rb.bearer_auth(&self.config.api_key);
        }
        let response = rb.json(req).send().await.map_err(AgentError::from)?;
        let status = response.status();
        let body_text = response.text().await.map_err(AgentError::from)?;
        if !status.is_success() {
            return Err(AgentError::Other(format!(
                "Responses API HTTP {} error: {}",
                status, body_text
            )));
        }
        let parsed: ResponsesResponse =
            serde_json::from_str(&body_text).map_err(AgentError::from)?;
        Ok(parsed)
    }

    pub async fn create_stream<F>(
        &self,
        req: &ResponsesRequest,
        mut on_event: F,
    ) -> Result<(), AgentError>
    where
        F: FnMut(Value) -> Result<(), AgentError>,
    {
        let mut req = req.clone();
        req.stream = Some(true);
        let mut rb = self.http.post(self.url());
        if !self.config.api_key.is_empty() {
            rb = rb.bearer_auth(&self.config.api_key);
        }
        let response = rb.json(&req).send().await.map_err(AgentError::from)?;
        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.map_err(AgentError::from)?;
            return Err(AgentError::Other(format!(
                "Responses API HTTP {} error: {}",
                status, body_text
            )));
        }

        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(AgentError::from)?;
            buffer.extend_from_slice(&chunk);
            while let Some(idx) = find_event_boundary(&buffer) {
                let event = buffer.drain(..idx).collect::<Vec<u8>>();
                let event = String::from_utf8_lossy(&event);
                for line in event.lines() {
                    let line = line.trim();
                    if !line.starts_with("data:") {
                        continue;
                    }
                    let data = line.trim_start_matches("data:").trim();
                    if data == "[DONE]" {
                        return Ok(());
                    }
                    let value: Value = serde_json::from_str(data).map_err(AgentError::from)?;
                    on_event(value)?;
                }
            }
        }
        Ok(())
    }
}

fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    buf.windows(2)
        .position(|w| w == b"\n\n")
        .map(|pos| pos + 2)
}
