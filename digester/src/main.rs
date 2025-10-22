// Digester: File system watcher and document uploader
//
// This service watches a directory for new files and automatically uploads them
// to the polars-worker API for ingestion into the RAG system. It handles:
// - File system event monitoring using the notify crate
// - Concurrent file uploads with retry logic
// - Deduplication to prevent processing the same file multiple times
// - Graceful shutdown handling

use std::{
    collections::HashSet,
    env,
    path::{Path, PathBuf},
    sync::{mpsc, Arc, Mutex},
    thread,
    time::Duration,
};

use anyhow::{Context, Result};
use notify::{
    event::{CreateKind, EventAttributes},
    Config as NotifyConfig, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
};
use reqwest::blocking::{multipart, Client};
use thiserror::Error;
use time::{macros::format_description, OffsetDateTime};

/// Configuration settings loaded from environment variables
#[derive(Debug, Clone)]
struct Settings {
    digest_path: PathBuf,       // Directory to watch for new files
    polars_api: String,         // Base URL for the polars-worker API
    max_retries: u32,           // Number of upload retry attempts
    retry_delay: Duration,      // Delay between retry attempts
    upload_timeout: Duration,   // HTTP request timeout
}

impl Settings {
    /// Load configuration from environment variables with sensible defaults
    fn load() -> Result<Self> {
        // Watch directory - defaults to /digestion (typical Docker mount point)
        let digest_path = env::var("DIGEST_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/digestion"));

        // API endpoint - defaults to polars-worker service name in Docker network
        let polars_api =
            env::var("POLARS_API").unwrap_or_else(|_| "http://polars-worker:8080".to_string());

        // Number of times to retry failed uploads before giving up (default: 3)
        let max_retries = env::var("MAX_RETRIES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(3);

        // Seconds to wait between retry attempts (default: 30s)
        let retry_delay = env::var("RETRY_DELAY")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(30));

        // HTTP request timeout - 5 minutes to handle large file uploads
        let upload_timeout = Duration::from_secs(300);

        Ok(Self {
            digest_path,
            polars_api,
            max_retries,
            retry_delay,
            upload_timeout,
        })
    }
}

/// Error types for file upload operations
#[derive(Debug, Error)]
enum UploadError {
    #[error("request failed with status {0}")]
    BadStatus(reqwest::StatusCode),  // HTTP error response from server
    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),  // Network or request errors
    #[error(transparent)]
    Io(#[from] std::io::Error),  // File I/O errors
}

fn main() -> Result<()> {
    // Load configuration and ensure watch directory exists
    let settings = Arc::new(Settings::load()?);
    std::fs::create_dir_all(&settings.digest_path)
        .with_context(|| format!("ensure watch dir {}", settings.digest_path.display()))?;

    // Display startup configuration
    println!("👀 Watching {}", settings.digest_path.display());
    println!("🔗 Polars API {}", settings.polars_api);
    println!(
        "🔄 Max retries {}, delay {}s",
        settings.max_retries,
        settings.retry_delay.as_secs()
    );

    // Track files we've already processed to prevent duplicates
    // Uses Arc<Mutex<>> for thread-safe sharing across spawned upload threads
    let processed = Arc::new(Mutex::new(HashSet::<PathBuf>::new()));

    // Create HTTP client with timeout configured for large file uploads
    let client = Arc::new(
        Client::builder()
            .timeout(settings.upload_timeout)
            .build()
            .context("build HTTP client")?,
    );

    // Set up channel for file system events
    // tx (sender) is cloned for the watcher callback
    // rx (receiver) is used in the main event loop
    let (tx, rx) = mpsc::channel::<Event>();
    let watcher_tx = tx.clone();
    let watch_dir = settings.digest_path.clone();

    // Create file system watcher with callback that sends events to channel
    // The callback runs in a separate thread managed by the notify crate
    let mut watcher = RecommendedWatcher::new(
        move |res| {
            if let Ok(event) = res {
                if watcher_tx.send(event).is_err() {
                    eprintln!("watch channel closed");
                }
            }
        },
        NotifyConfig::default(),
    )
    .context("start file watcher")?;

    // Start watching the directory (non-recursive - only files in root)
    watcher
        .watch(&watch_dir, RecursiveMode::NonRecursive)
        .context("watch directory")?;

    // Process any files that already exist in the directory at startup
    // This ensures we don't miss files that were added while the service was down
    enqueue_existing_files(&settings.digest_path, &tx)?;

    // Set up graceful shutdown handler for Ctrl+C
    let shutdown = Arc::new(Mutex::new(false));
    {
        let shutdown = shutdown.clone();
        ctrlc::set_handler(move || {
            println!("\n🛑 Shutdown signal received.");
            if let Ok(mut flag) = shutdown.lock() {
                *flag = true;
            }
        })
        .context("install ctrlc handler")?;
    }

    // Main event loop: process file system events until shutdown
    loop {
        // Check for shutdown signal
        if *shutdown.lock().unwrap() {
            println!("👋 Digester shutdown complete");
            break;
        }

        // Wait for events with timeout to allow periodic shutdown checks
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(event) => {
                // Filter event to get the path of a newly created/modified file
                if let Some(path) = select_path(&settings.digest_path, &event) {
                    // Check if we've already processed this file (deduplication)
                    // Lock is scoped to minimize contention
                    let should_process = {
                        let mut seen = processed.lock().unwrap();
                        if seen.contains(&path) {
                            false  // Already processed - skip it
                        } else {
                            seen.insert(path.clone());  // Mark as processing
                            true
                        }
                    };

                    if should_process {
                        // Spawn a new thread to handle the upload concurrently
                        // This allows the main loop to continue watching for new files
                        // while uploads are in progress
                        let client = client.clone();
                        let settings = settings.clone();
                        let processed = processed.clone();
                        thread::spawn(move || {
                            if let Err(err) = handle_file(settings.as_ref(), client.as_ref(), &path)
                            {
                                // Upload failed after all retries - remove from processed set
                                // so it can be retried if the file is modified again
                                eprintln!("💥 {} error: {err}", display_name(&path));
                                processed.lock().unwrap().remove(&path);
                            } else {
                                // Upload succeeded - log with timestamp
                                let timestamp = OffsetDateTime::now_utc()
                                    .format(format_description!(
                                        "[year]-[month]-[day] [hour]:[minute]:[second]"
                                    ))
                                    .unwrap_or_else(|_| OffsetDateTime::now_utc().to_string());
                                println!("✅ {} processed at {}", display_name(&path), timestamp);
                            }
                        });
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // No events received - periodic no-op to allow shutdown check
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Watcher channel closed unexpectedly - exit
                eprintln!("watcher disconnected");
                break;
            }
        }
    }

    Ok(())
}

/// Scan directory for existing files and queue them for processing
/// This is called at startup to handle files added while service was down
fn enqueue_existing_files(dir: &Path, tx: &mpsc::Sender<Event>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let path = entry.path();
            // Create a synthetic file creation event for each existing file
            let event = Event {
                kind: EventKind::Create(CreateKind::File),
                paths: vec![path.clone()],
                attrs: EventAttributes::default(),
            };
            tx.send(event)
                .with_context(|| format!("queue {}", path.display()))?;
        }
    }
    Ok(())
}

/// Extract file path from a file system event, filtering for relevant events
/// Returns Some(path) only for Create/Modify events on files within the watch directory
fn select_path(root: &Path, event: &Event) -> Option<PathBuf> {
    // Only process creation and modification events
    if !matches!(
        event.kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Any
    ) {
        return None;
    }

    // Find the first path that is a file within our watch directory
    event
        .paths
        .iter()
        .find(|p| p.starts_with(root) && p.is_file())
        .cloned()
}

/// Handle a file upload with retry logic
/// Attempts upload up to max_retries times, then deletes the file on success
fn handle_file(settings: &Settings, client: &Client, path: &Path) -> Result<()> {
    // Retry loop with exponential backoff
    for attempt in 0..=settings.max_retries {
        match upload_file(client, &settings.polars_api, path) {
            Ok(_) => {
                // Upload succeeded - delete the original file
                std::fs::remove_file(path).with_context(|| format!("remove {}", path.display()))?;
                return Ok(());
            }
            Err(err) => {
                eprintln!(
                    "⚠️ {} attempt {} failed: {err}",
                    display_name(path),
                    attempt + 1
                );
                if attempt == settings.max_retries {
                    // All retries exhausted - return error
                    return Err(err.into());
                }
                // Wait before retrying (allows transient issues to resolve)
                thread::sleep(settings.retry_delay);
            }
        }
    }
    Ok(())
}

/// Upload a file to the polars-worker API using multipart form data
fn upload_file(client: &Client, base_url: &str, path: &Path) -> Result<(), UploadError> {
    // Skip if file was deleted between detection and upload
    if !path.exists() {
        return Ok(());
    }

    // Build multipart form with the file
    let form = multipart::Form::new()
        .file("file", path)
        .map_err(UploadError::from)?;

    // POST to the /ingest/file endpoint
    let response = client
        .post(format!("{}/ingest/file", base_url))
        .multipart(form)
        .send()
        .map_err(UploadError::from)?;

    // Check for successful status code (2xx)
    if response.status().is_success() {
        Ok(())
    } else {
        Err(UploadError::BadStatus(response.status()))
    }
}

/// Extract just the filename from a path for cleaner logging
fn display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_else(|| path.to_string_lossy().as_ref())
        .to_string()
}
