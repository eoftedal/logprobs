import {
  AutoTokenizer,
  AutoModelForCausalLM,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4';

// ── WebGPU guard ──────────────────────────────────────────────────
if (!navigator.gpu) {
  document.getElementById('err-webgpu').hidden = false;
  document.getElementById('load-btn').disabled = true;
}

// ── Models ────────────────────────────────────────────────────────
const models = [
  { id: 'onnx-community/Qwen2.5-Coder-0.5B-Instruct', label: 'Qwen2.5-Coder-0.5B  — ~350 MB',  dtype: 'q4f16' },
  { id: 'onnx-community/Qwen2.5-Coder-1.5B-Instruct', label: 'Qwen2.5-Coder-1.5B  — ~1 GB',    dtype: 'q4f16', default: true },
  { id: 'onnx-community/Qwen2.5-Coder-3B-Instruct',   label: 'Qwen2.5-Coder-3B  — ~2 GB',      dtype: 'q4f16' },
  { id: 'onnx-community/Qwen3-0.6B-ONNX',             label: 'Qwen3-0.6B  — ~570 MB',           dtype: 'q4f16' },
  { id: 'onnx-community/Qwen3-1.7B-ONNX',             label: 'Qwen3-1.7B  — ~1.4 GB',           dtype: 'q4f16' },
  { id: 'onnx-community/Qwen3-4B-ONNX',               label: 'Qwen3-4B  — ~2.8 GB',             dtype: 'q4f16' },
  { id: 'onnx-community/Qwen3.5-2B-ONNX',             label: 'Qwen3.5-2B  — ~1.6 GB',           dtype: 'q4f16' },
  { id: 'onnx-community/Qwen3.5-4B-ONNX',             label: 'Qwen3.5-4B  — ~3 GB',             dtype: 'q4f16' },
  { id: 'onnx-community/gemma-3-270m-it-ONNX',        label: 'Gemma 3 270M  — ~275 MB',         dtype: 'q4f16' },
  { id: 'onnx-community/gemma-3-1b-it-ONNX',          label: 'Gemma 3 1B  — ~765 MB',           dtype: 'q4f16' },
  { id: 'onnx-community/gemma-4-E2B-it-ONNX',         label: 'Gemma 4 2B  — ~1.5 GB',           dtype: 'q4f16' },
  { id: 'onnx-community/gemma-4-E4B-it-ONNX',         label: 'Gemma 4 4B  — ~3 GB',             dtype: 'q4f16' },
  { id: 'onnx-community/LFM2-350M-ONNX',              label: 'LFM2-350M  — ~255 MB',             dtype: 'q4f16' },
  { id: 'onnx-community/LFM2-700M-ONNX',              label: 'LFM2-700M  — ~500 MB',             dtype: 'q4f16' },
  { id: 'onnx-community/LFM2-1.2B-ONNX',              label: 'LFM2-1.2B  — ~760 MB',             dtype: 'q4f16' },
];

// ── Code samples ──────────────────────────────────────────────────
const codeSamples = {
  passwordStorage: {
    label: 'Password storage',
    code: `package org.example.expensing;
public class PasswordStorage {
  public String passwordStorageAlgorithm = "`,
  },
  passwordStorageComment: {
    label: 'Password storage (with comment)',
    code: `package org.example.expensing;
public class PasswordStorage {
  /* Secure password storage for passwords that are not recoverable.
     According to current standards the algorithm for hashing the passwords 
     should be `
  },
  sql: {
    label: 'SQL query',
    code: `package org.example.expensing;
public class UserRepository {
  public User getUserById(String userId) {
    String query = "SELECT * FROM Users WHERE id=`,
  },
  sqlinjection: {
    label: 'SQL injection repeat',
    code: `import psycopg2

connection = psycopg2.connect(host="localhost", database="psycopgtest", user="postgres", password=None)

def is_admin(username: str) -> bool:
    with connection.cursor() as cursor:
        query = "SELECT admin FROM users WHERE username = '%s'" % username
        cursor.execute(query)
        result = cursor.fetchone()
    admin, = result
    return admin

def get_user_id(username: str) -> int:
    with connection.cursor() as cursor:
        query = "SELECT id FROM users WHERE username = `
  },
  randomNumber: {
    label: 'Random number generation',
    code: `Question: Pick a number between 0 and 9
Answer: I'll pick the number `
  }
};

// ── State ──────────────────────────────────────────────────────────
let tok = null;
let mdl = null;
let currentText = '';
let history = [];
let chatMode = false;
let templatePrefix = '';
let lastLogits = null;
let apiMode = false;
let apiConfig = { url: '', key: '', model: '' };
let lastApiLogprobs = null;
let renderGen = 0; // invalidates in-flight inference after reset/undo/new-start

// ── Element refs ───────────────────────────────────────────────────
const loadBtn       = document.getElementById('load-btn');
const switchBtn     = document.getElementById('switch-btn');
const modelSel      = document.getElementById('model-sel');
const progressWrap   = document.getElementById('progress-wrap');
const progressStatus = document.getElementById('progress-status');
const progressFiles  = document.getElementById('progress-files');
const fileRows = new Map();
const setupCard     = document.getElementById('setup-card');
const inputCard     = document.getElementById('input-card');
const explorerCard  = document.getElementById('explorer-card');
const sampleSel     = document.getElementById('sample-sel');
const startingText  = document.getElementById('starting-text');
const systemText    = document.getElementById('system-text');
const userText      = document.getElementById('user-text');
const tabRaw        = document.getElementById('tab-raw');
const tabChat       = document.getElementById('tab-chat');
const tabBtns       = document.querySelectorAll('.tab-btn');
const startBtn      = document.getElementById('start-btn');
const accText       = document.getElementById('acc-text');
const tokenGrid     = document.getElementById('token-grid');
const hint          = document.getElementById('hint');
const undoBtn       = document.getElementById('undo-btn');
const resetBtn      = document.getElementById('reset-btn');
const topkSlider    = document.getElementById('topk-slider');
const topkVal       = document.getElementById('topk-val');
const tempSlider    = document.getElementById('temp-slider');
const tempVal       = document.getElementById('temp-val');

for (const m of models) {
  const opt = document.createElement('option');
  opt.value = m.id;
  opt.textContent = m.label;
  if (m.default) opt.selected = true;
  modelSel.appendChild(opt);
}
const apiOpt = document.createElement('option');
apiOpt.value = '__api__';
apiOpt.textContent = 'Custom (OpenAI-compatible API)';
modelSel.appendChild(apiOpt);

const apiConfigDiv = document.getElementById('api-config');
modelSel.addEventListener('change', () => {
  apiConfigDiv.hidden = modelSel.value !== '__api__';
});

const customOpt = document.createElement('option');
customOpt.value = '';
customOpt.textContent = '— Custom —';
sampleSel.appendChild(customOpt);
for (const [key, { label }] of Object.entries(codeSamples)) {
  const opt = document.createElement('option');
  opt.value = key;
  opt.textContent = label;
  sampleSel.appendChild(opt);
}

let hashText = '';
try {
  hashText = decodeURIComponent(location.hash.slice(1));
} catch {
  // malformed percent-encoding in a shared link — fall back to the default sample
}
if (hashText) {
  startingText.value = hashText;
  sampleSel.value = '';
} else {
  sampleSel.value = 'passwordStorage';
  startingText.value = codeSamples.passwordStorage.code;
}

function updateHash() {
  const val = startingText.value;
  window.history.replaceState(null, '', val ? '#' + encodeURIComponent(val) : location.pathname + location.search);
}

function autoGrow(el) {
  if (el.offsetParent === null) return; // skip hidden elements
  if (el.scrollHeight > el.clientHeight) {
    const cs = getComputedStyle(el);
    const borderY = parseFloat(cs.borderTopWidth) + parseFloat(cs.borderBottomWidth);
    el.style.height = (el.scrollHeight + borderY) + 'px';
  }
}

[startingText, systemText, userText].forEach(el => {
  el.addEventListener('input', () => autoGrow(el));
});

sampleSel.addEventListener('change', () => {
  if (sampleSel.value) startingText.value = codeSamples[sampleSel.value].code;
  updateHash();
  autoGrow(startingText);
});

startingText.addEventListener('input', () => {
  sampleSel.value = '';
  updateHash();
});

function getTemperature() { return parseFloat(tempSlider.value); }

function rerender() {
  if (apiMode && lastApiLogprobs) {
    renderButtons(applyApiTemperatureTopK(lastApiLogprobs, getTemperature(), getTopK()));
  } else if (lastLogits) {
    renderButtons(topKFromProbs(probsFromLogits(lastLogits, getTemperature()), getTopK()));
  }
}

topkSlider.addEventListener('input', () => {
  topkVal.textContent = topkSlider.value;
  rerender();
});

tempSlider.addEventListener('input', () => {
  tempVal.textContent = getTemperature().toFixed(1);
  rerender();
});

// ── Tab switching ─────────────────────────────────────────────────
tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    tabBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    tabRaw.hidden  = tab !== 'raw';
    tabChat.hidden = tab !== 'chat';
    if (tab === 'raw') autoGrow(startingText);
    else { autoGrow(systemText); autoGrow(userText); }
  });
});

function getTopK() { return parseInt(topkSlider.value, 10); }

// ── Switch model (reload to release memory) ──────────────────────
switchBtn.addEventListener('click', () => {
  location.reload();
});

// ── Load model ────────────────────────────────────────────────────
loadBtn.addEventListener('click', async () => {
  const modelId = modelSel.value;

  if (modelId === '__api__') {
    const url   = document.getElementById('api-url').value.trim().replace(/\/$/, '');
    const key   = document.getElementById('api-key').value.trim();
    const model = document.getElementById('api-model').value.trim();
    if (!url || !model) {
      progressWrap.hidden = false;
      resetProgressFiles();
      setStatus('Error: Base URL and model name are required.', true);
      return;
    }
    apiConfig = { url, key, model };
    apiMode = true;
    loadBtn.hidden = true;
    switchBtn.hidden = false;
    modelSel.disabled = true;
    // Hide chat tab — API mode uses raw text only
    document.querySelector('[data-tab="chat"]').hidden = true;
    tabRaw.hidden = false;
    document.querySelector('[data-tab="raw"]').classList.add('active');
    document.querySelector('[data-tab="chat"]').classList.remove('active');
    inputCard.hidden = false;
    autoGrow(startingText);
    return;
  }

  apiMode = false;
  const dtype = models.find(m => m.id === modelId)?.dtype ?? 'q4f16';
  loadBtn.disabled = true;
  modelSel.disabled = true;
  progressWrap.hidden = false;
  resetProgressFiles();

  try {
    setStatus('Loading tokenizer…');
    tok = await AutoTokenizer.from_pretrained(modelId, {
      progress_callback: makeProgress('Tokenizer'),
    });

    setStatus('Downloading model weights (cached after first load)…');
    mdl = await AutoModelForCausalLM.from_pretrained(modelId, {
      dtype,
      device: 'webgpu',
      progress_callback: makeProgress('Model'),
    });

    setStatus('Ready!');
    resetProgressFiles();
    loadBtn.hidden = true;
    switchBtn.hidden = false;
    inputCard.hidden = false;
    autoGrow(startingText);
    autoGrow(systemText);

  } catch (err) {
    setStatus(`Error: ${err.message}`, true);
    loadBtn.disabled = false;
    modelSel.disabled = false;
    console.error(err);
  }
});

function makeProgress(label) {
  return (info) => {
    if (!info || !info.file) return;
    const key = `${label}|${info.file}`;
    let row = fileRows.get(key);
    if (!row) {
      row = createFileRow(info.file);
      fileRows.set(key, row);
    }
    if (info.status === 'done') {
      row.fill.style.width = '100%';
      row.label.textContent = `${info.file}: done`;
    } else if (info.progress != null) {
      row.fill.style.width = `${info.progress}%`;
      row.label.textContent = `${info.file}: ${Math.round(info.progress)}%`;
    } else if (info.status) {
      row.label.textContent = `${info.file}: ${info.status}`;
    }
  };
}

function createFileRow(file) {
  const wrap = document.createElement('div');
  wrap.className = 'progress-file';
  const bar = document.createElement('div');
  bar.className = 'progress-bar';
  const fill = document.createElement('div');
  fill.className = 'progress-fill';
  bar.appendChild(fill);
  const label = document.createElement('div');
  label.className = 'progress-label';
  label.textContent = `${file}: …`;
  wrap.append(bar, label);
  progressFiles.appendChild(wrap);
  return { wrap, fill, label };
}

function resetProgressFiles() {
  fileRows.clear();
  progressFiles.innerHTML = '';
  progressStatus.classList.remove('error');
}

function setStatus(text, isError = false) {
  progressStatus.textContent = text;
  progressStatus.classList.toggle('error', isError);
}

// ── Start ──────────────────────────────────────────────────────────
startBtn.addEventListener('click', async () => {
  const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
  if (activeTab === 'chat') {
    const messages = [
      { role: 'system', content: systemText.value },
      { role: 'user',   content: userText.value   },
    ];
    currentText = '';
    try {
      templatePrefix = tok.apply_chat_template(messages, {
        add_generation_prompt: true,
        tokenize: false,
      });
    } catch (e) {
      // tokenizer has no chat_template — prepend system prompt as plain text
      templatePrefix = systemText.value + '\n\n';
      currentText = userText.value;
    }
    chatMode = true;
  } else {
    if (!startingText.value) return;
    templatePrefix = '';
    chatMode = false;
    currentText = startingText.value;
  }
  history = [];
  inputCard.hidden = true;
  explorerCard.hidden = false;
  await fetchAndRender();
});

// ── Reset ──────────────────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  renderGen++;
  explorerCard.hidden = true;
  inputCard.hidden = false;
  tokenGrid.innerHTML = '';
  tokenGrid.classList.remove('loading');
  currentText = '';
  templatePrefix = '';
  chatMode = false;
  lastLogits = null;
  lastApiLogprobs = null;
  history = [];
  undoBtn.disabled = true;
});

// ── Undo ───────────────────────────────────────────────────────────
undoBtn.addEventListener('click', async () => {
  if (!history.length) return;
  currentText = history.pop();
  undoBtn.disabled = history.length === 0;
  await fetchAndRender();
});

// ── Core: get top-k tokens ─────────────────────────────────────────
async function fetchAndRender() {
  const gen = ++renderGen;
  tokenGrid.classList.add('loading');
  hint.textContent = 'Computing token probabilities…';

  if (chatMode) {
    accText.innerHTML =
      `<span style="opacity:0.3">${escHtml(templatePrefix)}</span>${escHtml(currentText)}`;
  } else {
    accText.textContent = currentText;
  }

  try {
    let tokens;
    if (apiMode) {
      const logprobs = await fetchApiLogprobs(currentText);
      if (gen !== renderGen) return;
      lastApiLogprobs = logprobs;
      tokens = applyApiTemperatureTopK(logprobs, getTemperature(), getTopK());
    } else {
      const logits = await computeLogits(currentText);
      if (gen !== renderGen) return;
      lastLogits = logits;
      tokens = topKFromProbs(probsFromLogits(logits, getTemperature()), getTopK());
    }
    renderButtons(tokens);
    hint.textContent = 'Click a token to extend the text.';
  } catch (err) {
    if (gen !== renderGen) return;
    hint.textContent = `Error: ${err.message}`;
    console.error(err);
  } finally {
    if (gen === renderGen) tokenGrid.classList.remove('loading');
  }
}

function probsFromLogits(logits, temperature) {
  const n = logits.length;
  let maxIdx = 0;
  for (let i = 1; i < n; i++) if (logits[i] > logits[maxIdx]) maxIdx = i;
  if (temperature === 0) {
    const probs = new Float32Array(n);
    probs[maxIdx] = 1;
    return probs;
  }
  const max = logits[maxIdx];
  const probs = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) { probs[i] = Math.exp((logits[i] - max) / temperature); sum += probs[i]; }
  for (let i = 0; i < n; i++) probs[i] /= sum;
  return probs;
}

function topKFromProbs(probs, k) {
  const top = [];
  for (let i = 0; i < probs.length; i++) {
    if (top.length < k) {
      top.push(i);
      if (top.length === k) top.sort((a, b) => probs[a] - probs[b]);
    } else if (probs[i] > probs[top[0]]) {
      top[0] = i;
      top.sort((a, b) => probs[a] - probs[b]);
    }
  }
  top.sort((a, b) => probs[b] - probs[a]); // descending
  const mass = top.reduce((s, id) => s + probs[id], 0);
  return top.map(id => ({
    id,
    token: tok.decode([id], { skip_special_tokens: false }),
    prob: probs[id] / mass,
  }));
}

async function computeLogits(text) {
  const fullText = chatMode ? templatePrefix + text : text;
  const inputs = tok(fullText, {
    return_tensors: 'pt',
    // the rendered chat template already contains the special tokens;
    // raw text needs them added (e.g. Gemma expects a leading <bos>)
    add_special_tokens: !chatMode,
  });

  const output = await mdl(inputs);
  const logits = output.logits; // Tensor [1, seq_len, vocab_size]

  const vocabSize = logits.dims[2];
  const seqLen    = logits.dims[1];
  const data      = logits.data; // Float32Array
  const offset    = (seqLen - 1) * vocabSize;

  return data.slice(offset, offset + vocabSize);
}

// ── API inference path ─────────────────────────────────────────────
function applyApiTemperatureTopK(logprobs, temperature, k) {
  if (temperature === 0) {
    const best = logprobs.reduce((a, b) => a.logprob > b.logprob ? a : b);
    return [{ token: best.token, prob: 1.0 }];
  }
  const scaled = logprobs.map(e => e.logprob / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const entries = logprobs.map((e, i) => ({ token: e.token, prob: exps[i] / sum }));
  entries.sort((a, b) => b.prob - a.prob);
  return entries.slice(0, k);
}

async function fetchApiLogprobs(text) {
  const headers = { 'Content-Type': 'application/json' };
  if (apiConfig.key) headers['Authorization'] = `Bearer ${apiConfig.key}`;
  const resp = await fetch(`${apiConfig.url}/completions`, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model: apiConfig.model,
      prompt: text,
      max_tokens: 1,
      temperature: 1.0,
      logprobs: 20, // int form per the completions endpoint; some servers cap this lower
    }),
  });
  if (!resp.ok) throw new Error(`API error ${resp.status}: ${await resp.text()}`);
  return parseTopLogprobs(await resp.json());
}

// Normalizes to [{ token, logprob }, …] — servers answer /completions with
// either the legacy completions shape or the chat-completions shape.
function parseTopLogprobs(data) {
  const lp = data.choices?.[0]?.logprobs;
  if (!lp) throw new Error('No logprobs in API response — the server may not support them.');
  // chat-completions shape: logprobs.content[0].top_logprobs = [{ token, logprob }, …]
  const chat = lp.content?.[0]?.top_logprobs;
  if (Array.isArray(chat) && chat.length) {
    return chat.map(({ token, logprob }) => ({ token, logprob }));
  }
  // legacy completions shape: logprobs.top_logprobs[0] = { "<token>": <logprob>, … }
  const legacy = lp.top_logprobs?.[0];
  if (legacy && typeof legacy === 'object' && Object.keys(legacy).length) {
    return Object.entries(legacy).map(([token, logprob]) => ({ token, logprob }));
  }
  throw new Error('Unrecognized logprobs format in API response.');
}

// ── Render token buttons ───────────────────────────────────────────
function renderButtons(tokens) {
  tokenGrid.innerHTML = '';
  if (!tokens.length) return;
  const maxProb = tokens[0].prob;

  for (const { token, prob } of tokens) {
    const btn = document.createElement('button');
    btn.className = 'tok';

    // Visible representation of whitespace / control chars
    const display = token
      .replace(/\r\n|\n/g, '↵')
      .replace(/ /g, '·')
      .replace(/\t/g, '→');

    const pct      = (prob * 100).toFixed(1);
    const relFill  = (prob / maxProb * 100).toFixed(1);
    const hue      = Math.round(prob / maxProb * 120); // 0 = red, 120 = green

    btn.style.setProperty('--fill',  `${relFill}%`);
    btn.style.setProperty('--color', `hsl(${hue},65%,38%)`);

    btn.innerHTML =
      `<span class="tok-text">${escHtml(display) || '<span style="opacity:.4;font-style:italic">empty</span>'}</span>` +
      `<span class="tok-prob">${pct}%</span>`;

    btn.addEventListener('click', async () => {
      history.push(currentText);
      currentText += token;
      undoBtn.disabled = false;
      await fetchAndRender();
    });

    tokenGrid.appendChild(btn);
  }
}

function escHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Theme toggle ──────────────────────────────────────────────────
const themeToggle = document.getElementById('theme-toggle');
themeToggle.addEventListener('click', () => {
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  if (isLight) {
    document.documentElement.removeAttribute('data-theme');
    localStorage.setItem('theme', 'dark');
  } else {
    document.documentElement.setAttribute('data-theme', 'light');
    localStorage.setItem('theme', 'light');
  }
});


