// SunGrid Cooperative Copilot -- frontend
// Vanilla JS, no build step. Talks to the FastAPI backend mounted at the
// same origin (/api/*), so no CORS configuration is needed in production.

const API_BASE = "/api";

const els = {
  messages: document.getElementById("messages"),
  emptyState: document.getElementById("emptyState"),
  composer: document.getElementById("composer"),
  input: document.getElementById("questionInput"),
  sendBtn: document.getElementById("sendBtn"),
  threadList: document.getElementById("threadList"),
  threadTitle: document.getElementById("threadTitle"),
  newChatBtn: document.getElementById("newChatBtn"),
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
};

let state = {
  threadId: localStorage.getItem("sungrid_current_thread") || null,
  threads: [],
  sending: false,
};

// ---------------------------------------------------------------- utils --

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// Render "[1]" / "[2, 3]" style citation markers as schematic-style chips.
function renderAnswerHtml(text) {
  const escaped = escapeHtml(text);
  return escaped.replace(/\[(\d+(?:,\s*\d+)*)\]/g, (match, nums) => {
    return `<span class="cite" title="source reference ${nums}">${match}</span>`;
  });
}

function autoGrowTextarea() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 160) + "px";
}

// ------------------------------------------------------------ rendering --

function clearMessages() {
  els.messages.innerHTML = "";
}

function showEmptyState() {
  clearMessages();
  els.messages.appendChild(els.emptyState);
}

function appendMessage(role, content, sources = []) {
  if (els.emptyState.parentElement) els.emptyState.remove();

  const row = document.createElement("div");
  row.className = `msg-row ${role === "human" ? "human" : "ai"}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = role === "human" ? escapeHtml(content) : renderAnswerHtml(content);

  if (role !== "human" && sources.length) {
    const line = document.createElement("div");
    line.className = "sources-line";
    line.innerHTML = sources
      .map((s) => `<span class="source-chip">${escapeHtml(s)}</span>`)
      .join("");
    bubble.appendChild(line);
  }

  row.appendChild(bubble);
  els.messages.appendChild(row);
  els.messages.scrollTop = els.messages.scrollHeight;
  return row;
}

function appendThinkingRow() {
  const row = document.createElement("div");
  row.className = "msg-row ai thinking";
  row.innerHTML = `<div class="bubble"><span class="sun-spinner"></span> thinking…</div>`;
  els.messages.appendChild(row);
  els.messages.scrollTop = els.messages.scrollHeight;
  return row;
}

function renderThreadList() {
  els.threadList.innerHTML = "";
  for (const t of state.threads) {
    const btn = document.createElement("button");
    btn.className = "thread-item" + (t.thread_id === state.threadId ? " active" : "");
    btn.textContent = t.title || "New chat";
    btn.title = t.title || "New chat";
    btn.addEventListener("click", () => selectThread(t.thread_id));
    els.threadList.appendChild(btn);
  }
}

// -------------------------------------------------------------- backend --

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

async function checkHealth() {
  try {
    await fetchJson("/health");
    els.statusDot.className = "dot ok";
    els.statusText.textContent = "backend connected";
  } catch (e) {
    els.statusDot.className = "dot err";
    els.statusText.textContent = "backend unreachable";
  }
}

async function loadThreadList() {
  try {
    state.threads = await fetchJson(`${API_BASE}/threads`);
  } catch (e) {
    state.threads = [];
  }
  renderThreadList();
}

async function selectThread(threadId) {
  state.threadId = threadId;
  localStorage.setItem("sungrid_current_thread", threadId);
  renderThreadList();

  if (!threadId) {
    els.threadTitle.textContent = "New chat";
    showEmptyState();
    return;
  }

  try {
    const detail = await fetchJson(`${API_BASE}/threads/${threadId}`);
    els.threadTitle.textContent = detail.title || "Chat";
    clearMessages();
    if (detail.messages.length === 0) {
      showEmptyState();
    } else {
      for (const m of detail.messages) {
        appendMessage(m.role, m.content);
      }
    }
  } catch (e) {
    // Thread might not exist yet on the backend (e.g. brand-new local id) --
    // fall back to a fresh, empty conversation rather than erroring out.
    els.threadTitle.textContent = "New chat";
    showEmptyState();
  }
}

function startNewChat() {
  state.threadId = null;
  localStorage.removeItem("sungrid_current_thread");
  els.threadTitle.textContent = "New chat";
  renderThreadList();
  showEmptyState();
  els.input.focus();
}

async function sendQuestion(question) {
  state.sending = true;
  els.sendBtn.disabled = true;

  appendMessage("human", question);
  const thinkingRow = appendThinkingRow();

  try {
    const body = { question };
    if (state.threadId) body.thread_id = state.threadId;

    const result = await fetchJson(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    thinkingRow.remove();
    appendMessage("ai", result.answer, result.sources);

    const isNewThread = state.threadId !== result.thread_id;
    state.threadId = result.thread_id;
    localStorage.setItem("sungrid_current_thread", state.threadId);

    if (isNewThread) {
      els.threadTitle.textContent = question.slice(0, 60);
      await loadThreadList();
    }
  } catch (e) {
    thinkingRow.remove();
    appendMessage(
      "ai",
      `Something went wrong reaching the copilot: ${e.message}. Check that the server is running and GROQ_API_KEY is configured.`
    );
  } finally {
    state.sending = false;
    els.sendBtn.disabled = false;
  }
}

// --------------------------------------------------------------- events --

els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = els.input.value.trim();
  if (!question || state.sending) return;
  els.input.value = "";
  autoGrowTextarea();
  sendQuestion(question);
});

els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});

els.input.addEventListener("input", autoGrowTextarea);

els.newChatBtn.addEventListener("click", startNewChat);

// ----------------------------------------------------------------- init --

(async function init() {
  await checkHealth();
  await loadThreadList();
  if (state.threadId) {
    await selectThread(state.threadId);
  } else {
    showEmptyState();
  }
})();
