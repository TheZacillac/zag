const DEFAULT_API_BASE = "http://localhost:8080";
const STORAGE_KEY = "rag:web-wrapper:api-base";
const MODEL_STORAGE_KEY = "rag:web-wrapper:model";

const form = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const kInput = document.getElementById("k");
const modelSelect = document.getElementById("model");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const answerSection = document.getElementById("answer");
const answerCard = document.getElementById("answer-card");

const baseUrlControlId = "api-base";

function getStoredApiBase() {
  return localStorage.getItem(STORAGE_KEY) ?? DEFAULT_API_BASE;
}

function setStoredApiBase(url) {
  localStorage.setItem(STORAGE_KEY, url);
}

function getStoredModel() {
  return localStorage.getItem(MODEL_STORAGE_KEY) ?? "";
}

function setStoredModel(model) {
  if (model) {
    localStorage.setItem(MODEL_STORAGE_KEY, model);
  } else {
    localStorage.removeItem(MODEL_STORAGE_KEY);
  }
}

function ensureBaseUrlControl() {
  if (document.getElementById(baseUrlControlId)) {
    return;
  }

  const optionsRow = document.querySelector(".options");
  if (!optionsRow) {
    return;
  }

  const label = document.createElement("label");
  label.setAttribute("for", baseUrlControlId);
  label.textContent = "API Base:";

  const input = document.createElement("input");
  input.type = "url";
  input.id = baseUrlControlId;
  input.name = baseUrlControlId;
  input.placeholder = DEFAULT_API_BASE;
  input.value = getStoredApiBase();
  input.addEventListener("change", () => {
    const value = input.value.trim() || DEFAULT_API_BASE;
    setStoredApiBase(value);
    refreshModels(value).catch((error) => {
      console.warn("Failed to refresh model list:", error);
    });
  });

  optionsRow.appendChild(label);
  optionsRow.appendChild(input);
}

async function loadModelData(baseUrl) {
  const sanitizedBase = (baseUrl || DEFAULT_API_BASE).replace(/\/$/, "");
  const response = await fetch(`${sanitizedBase}/models`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to load models (${response.status}): ${errorText}`);
  }

  return response.json();
}

function populateModelOptions(models = [], defaultModel = "") {
  if (!modelSelect) {
    return;
  }

  modelSelect.innerHTML = "";

  const autoOption = document.createElement("option");
  autoOption.value = "";
  autoOption.textContent = defaultModel
    ? `Default (${defaultModel})`
    : "Auto (env default)";
  modelSelect.appendChild(autoOption);

  const seen = new Set();
  if (defaultModel) {
    seen.add(defaultModel);
  }

  models.forEach((name) => {
    if (!name || seen.has(name)) {
      return;
    }
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    modelSelect.appendChild(option);
    seen.add(name);
  });

  const storedModel = getStoredModel();
  if (storedModel && !seen.has(storedModel)) {
    const customOption = document.createElement("option");
    customOption.value = storedModel;
    customOption.textContent = `${storedModel} (saved)`;
    modelSelect.appendChild(customOption);
    seen.add(storedModel);
  }

  if (storedModel && seen.has(storedModel)) {
    modelSelect.value = storedModel;
  } else {
    modelSelect.value = "";
  }
}

async function refreshModels(baseUrl = getStoredApiBase()) {
  if (!modelSelect) {
    return;
  }

  try {
    const data = await loadModelData(baseUrl);
    populateModelOptions(data.models || [], data.default || "");
  } catch (error) {
    console.warn("Model list unavailable:", error);
    // Keep existing options; ensure at least auto option exists
    if (!modelSelect.options.length) {
      populateModelOptions([], "");
    }
  }
}

function setStatus(message, type = "info") {
  statusEl.textContent = message;
  statusEl.dataset.status = type;
}

function clearAnswer() {
  if (answerCard) {
    answerCard.textContent = "";
  }
  if (answerSection) {
    answerSection.hidden = true;
  }
}

function renderAnswer(answer, model, chunkCount) {
  if (!answerSection || !answerCard) {
    return;
  }

  if (!answer) {
    clearAnswer();
    return;
  }

  const heading = answerSection.querySelector("h2");
  if (heading) {
    heading.textContent = model ? `Answer (${model})` : "Answer";
  }

  answerCard.textContent = answer;
  answerSection.hidden = false;

  const modelLabel = model || "default model";
  const chunkLabel = typeof chunkCount === "number" ? ` using ${chunkCount} chunk${chunkCount === 1 ? "" : "s"}` : "";
  setStatus(`Answer generated with ${modelLabel}${chunkLabel}.`, "success");
}

function formatSimilarity(distance) {
  if (typeof distance !== "number" || Number.isNaN(distance)) {
    return "–";
  }

  const similarity = Math.max(0, 1 - distance);
  return similarity.toFixed(3);
}

function renderResultCard(chunk, index) {
  const card = document.createElement("article");
  card.className = "result-card";

  const heading = document.createElement("h2");
  const prefix = typeof index === "number" ? `[${index + 1}] ` : "";
  heading.textContent = `${prefix}${chunk.title || "Untitled Document"}`;
  card.appendChild(heading);

  const meta = document.createElement("div");
  meta.className = "result-meta";

  if (typeof index === "number") {
    const order = document.createElement("span");
    order.className = "badge";
    order.textContent = `Context [${index + 1}]`;
    meta.appendChild(order);
  }

  if (chunk.chunk_id != null) {
    const chunkId = document.createElement("span");
    chunkId.textContent = `Chunk #${chunk.chunk_id}`;
    meta.appendChild(chunkId);
  }

  if (chunk.source_uri) {
    const source = document.createElement("span");
    source.textContent = chunk.source_uri;
    meta.appendChild(source);
  }

  const score = document.createElement("span");
  score.textContent = `Similarity: ${formatSimilarity(chunk.distance)}`;
  meta.appendChild(score);

  card.appendChild(meta);

  const text = document.createElement("p");
  text.className = "result-text";
  text.textContent = chunk.text || "(empty chunk)";
  card.appendChild(text);

  return card;
}

function renderResults(chunks) {
  resultsEl.innerHTML = "";

  if (!chunks?.length) {
    clearAnswer();
    setStatus("No matches found.", "empty");
    return;
  }

  const fragment = document.createDocumentFragment();
  chunks.forEach((chunk, index) => {
    fragment.appendChild(renderResultCard(chunk, index));
  });

  resultsEl.appendChild(fragment);
}

async function fetchAnswer(query, k, baseUrl, model) {
  const sanitizedBase = (baseUrl || DEFAULT_API_BASE).replace(/\/$/, "");
  const payload = { query, k: Number.parseInt(k, 10) || 5 };

  if (model) {
    payload.model = model;
  }

  const response = await fetch(`${sanitizedBase}/answer`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Request failed (${response.status}): ${errorText}`);
  }

  return response.json();
}

function handleError(error) {
  console.error(error);
  clearAnswer();
  setStatus(error.message || "Unexpected error. Check browser console.", "error");
}

function parseKValue(value) {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed) || parsed <= 0) {
    return 5;
  }
  return Math.min(parsed, 20);
}

function init() {
  ensureBaseUrlControl();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const query = queryInput.value.trim();
    const k = parseKValue(kInput.value);
    const baseUrlInput = document.getElementById(baseUrlControlId);
    const baseUrl = baseUrlInput?.value.trim() || getStoredApiBase();
    setStoredApiBase(baseUrl);
    const selectedModel = modelSelect?.value || "";
    setStoredModel(selectedModel);

    if (!query) {
      setStatus("Enter a query to search.", "warn");
      return;
    }

    form.querySelectorAll("button, input, select").forEach((el) => {
      el.disabled = true;
    });

    clearAnswer();
    resultsEl.innerHTML = "";
    setStatus("Searching & generating…", "pending");

    try {
      const data = await fetchAnswer(query, k, baseUrl, selectedModel);

      if (data.error) {
        throw new Error(data.error);
      }

      renderResults(data.chunks);

      if (Array.isArray(data.chunks) && data.chunks.length) {
        if (data.answer) {
          renderAnswer(data.answer, data.model, data.chunks.length);
        } else {
          clearAnswer();
          setStatus(
            `Retrieved ${data.chunks.length} chunk${data.chunks.length === 1 ? "" : "s"}, but no answer was returned.`,
            "warn"
          );
        }
      } else {
        clearAnswer();
      }
    } catch (error) {
      handleError(error);
    } finally {
      form.querySelectorAll("button, input, select").forEach((el) => {
        el.disabled = false;
      });
    }
  });

  if (modelSelect) {
    modelSelect.addEventListener("change", () => {
      setStoredModel(modelSelect.value);
    });
  }

  refreshModels().catch((error) => {
    console.warn("Initial model fetch failed:", error);
  });

  setStatus("Ready to search.");
}

document.addEventListener("DOMContentLoaded", init);

