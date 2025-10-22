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

#[derive(Debug, Clone)]
struct Settings {
    digest_path: PathBuf,
    polars_api: String,
    max_retries: u32,
    retry_delay: Duration,
    upload_timeout: Duration,
}

impl Settings {
    fn load() -> Result<Self> {
        let digest_path = env::var("DIGEST_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/digestion"));

        let polars_api =
            env::var("POLARS_API").unwrap_or_else(|_| "http://polars-worker:8080".to_string());

        let max_retries = env::var("MAX_RETRIES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(3);

        let retry_delay = env::var("RETRY_DELAY")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(30));

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

#[derive(Debug, Error)]
enum UploadError {
    #[error("request failed with status {0}")]
    BadStatus(reqwest::StatusCode),
    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

fn main() -> Result<()> {
    let settings = Arc::new(Settings::load()?);
    std::fs::create_dir_all(&settings.digest_path)
        .with_context(|| format!("ensure watch dir {}", settings.digest_path.display()))?;

    println!("👀 Watching {}", settings.digest_path.display());
    println!("🔗 Polars API {}", settings.polars_api);
    println!(
        "🔄 Max retries {}, delay {}s",
        settings.max_retries,
        settings.retry_delay.as_secs()
    );

    let processed = Arc::new(Mutex::new(HashSet::<PathBuf>::new()));
    let client = Arc::new(
        Client::builder()
            .timeout(settings.upload_timeout)
            .build()
            .context("build HTTP client")?,
    );

    let (tx, rx) = mpsc::channel::<Event>();
    let watcher_tx = tx.clone();
    let watch_dir = settings.digest_path.clone();

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

    watcher
        .watch(&watch_dir, RecursiveMode::NonRecursive)
        .context("watch directory")?;

    // Process any files that already exist before watching.
    enqueue_existing_files(&settings.digest_path, &tx)?;

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

    loop {
        if *shutdown.lock().unwrap() {
            println!("👋 Digester shutdown complete");
            break;
        }

        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(event) => {
                if let Some(path) = select_path(&settings.digest_path, &event) {
                    let should_process = {
                        let mut seen = processed.lock().unwrap();
                        if seen.contains(&path) {
                            false
                        } else {
                            seen.insert(path.clone());
                            true
                        }
                    };

                    if should_process {
                        let client = client.clone();
                        let settings = settings.clone();
                        let processed = processed.clone();
                        thread::spawn(move || {
                            if let Err(err) = handle_file(settings.as_ref(), client.as_ref(), &path)
                            {
                                eprintln!("💥 {} error: {err}", display_name(&path));
                                processed.lock().unwrap().remove(&path);
                            } else {
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
                // periodic no-op to allow shutdown check
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                eprintln!("watcher disconnected");
                break;
            }
        }
    }

    Ok(())
}

fn enqueue_existing_files(dir: &Path, tx: &mpsc::Sender<Event>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let path = entry.path();
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

fn select_path(root: &Path, event: &Event) -> Option<PathBuf> {
    if !matches!(
        event.kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Any
    ) {
        return None;
    }

    event
        .paths
        .iter()
        .find(|p| p.starts_with(root) && p.is_file())
        .cloned()
}

fn handle_file(settings: &Settings, client: &Client, path: &Path) -> Result<()> {
    for attempt in 0..=settings.max_retries {
        match upload_file(client, &settings.polars_api, path) {
            Ok(_) => {
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
                    return Err(err.into());
                }
                thread::sleep(settings.retry_delay);
            }
        }
    }
    Ok(())
}

fn upload_file(client: &Client, base_url: &str, path: &Path) -> Result<(), UploadError> {
    if !path.exists() {
        return Ok(());
    }

    let form = multipart::Form::new()
        .file("file", path)
        .map_err(UploadError::from)?;

    let response = client
        .post(format!("{}/ingest/file", base_url))
        .multipart(form)
        .send()
        .map_err(UploadError::from)?;

    if response.status().is_success() {
        Ok(())
    } else {
        Err(UploadError::BadStatus(response.status()))
    }
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_else(|| path.to_string_lossy().as_ref())
        .to_string()
}
