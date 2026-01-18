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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::extract::Json;
    use axum::http::StatusCode;
    use axum::response::Response;
    use axum::routing::post;
    use axum::Router;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use tokio::net::TcpListener;

    async fn spawn_test_server(router: Router) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let addr = listener.local_addr().expect("server local addr");
        tokio::spawn(async move {
            axum::serve(listener, router).await.expect("serve test");
        });
        format!("http://{}", addr)
    }

    fn test_config(base_url: String) -> Config {
        Config {
            api_key: String::new(),
            model: "gpt-4o-mini".to_string(),
            base_url,
            log_level: "info".to_string(),
            plugins_path: PathBuf::from("."),
            max_concurrent_requests: None,
        }
    }

    #[tokio::test]
    async fn create_parses_success_response() {
        let seen_body: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
        let seen_body_clone = Arc::clone(&seen_body);
        let app = Router::new().route(
            "/responses",
            post(move |Json(payload): Json<Value>| async move {
                *seen_body_clone.lock().expect("lock body") = Some(payload);
                (StatusCode::OK, Json(json!({ "id": "resp_123", "output": [] })))
            }),
        );

        let base_url = spawn_test_server(app).await;
        let client = ResponsesClient::new(test_config(base_url));

        let req = ResponsesRequest {
            instructions: Some("be concise".to_string()),
            model: Some("gpt-4o-mini".to_string()),
            input: json!([{"role":"user","content":"hello"}]),
            tools: None,
            reasoning: None,
            max_output_tokens: None,
            truncation: None,
            previous_response_id: None,
            store: None,
            stream: None,
        };

        let response = client.create(&req).await.expect("create response");
        assert_eq!(response.id.as_deref(), Some("resp_123"));

        let captured = seen_body.lock().expect("lock body").clone();
        let captured = captured.expect("capture request body");
        assert_eq!(captured["input"], req.input);
        assert_eq!(captured["model"], "gpt-4o-mini");
    }

    #[tokio::test]
    async fn create_returns_error_on_http_failure() {
        let app = Router::new().route(
            "/responses",
            post(|| async { (StatusCode::UNAUTHORIZED, "nope") }),
        );
        let base_url = spawn_test_server(app).await;
        let client = ResponsesClient::new(test_config(base_url));

        let req = ResponsesRequest {
            instructions: None,
            model: Some("gpt-4o-mini".to_string()),
            input: json!([{"role":"user","content":"hello"}]),
            tools: None,
            reasoning: None,
            max_output_tokens: None,
            truncation: None,
            previous_response_id: None,
            store: None,
            stream: None,
        };

        let err = client.create(&req).await.expect_err("expected error");
        let err_text = err.to_string();
        assert!(
            err_text.contains("Responses API HTTP 401"),
            "unexpected error: {}",
            err_text
        );
    }

    #[tokio::test]
    async fn create_stream_emits_events_until_done() {
        let body = [
            r#"data: {"type":"response.output_text.delta","delta":"hello"}"#,
            "",
            "data: [DONE]",
            "",
        ]
        .join("\n");
        let app = Router::new().route(
            "/responses",
            post(move || async move {
                Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "text/event-stream")
                    .body(Body::from(body))
                    .expect("sse response")
            }),
        );
        let base_url = spawn_test_server(app).await;
        let client = ResponsesClient::new(test_config(base_url));

        let events: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
        let events_clone = Arc::clone(&events);

        let req = ResponsesRequest {
            instructions: None,
            model: Some("gpt-4o-mini".to_string()),
            input: json!([{"role":"user","content":"stream"}]),
            tools: None,
            reasoning: None,
            max_output_tokens: None,
            truncation: None,
            previous_response_id: None,
            store: None,
            stream: None,
        };

        client
            .create_stream(&req, move |value| {
                events_clone
                    .lock()
                    .expect("lock events")
                    .push(value);
                Ok(())
            })
            .await
            .expect("stream ok");

        let captured = events.lock().expect("lock events");
        assert_eq!(captured.len(), 1);
        assert_eq!(
            captured[0],
            json!({"type":"response.output_text.delta","delta":"hello"})
        );
    }
}
