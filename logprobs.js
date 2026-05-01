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
  sql: {
    label: 'SQL query',
    code: `package org.example.expensing;
public class UserRepository {
  public User getUserById(String userId) {
    String query = "SELECT * FROM Users WHERE id=`,
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

// ── Element refs ───────────────────────────────────────────────────
const loadBtn       = document.getElementById('load-btn');
const modelSel      = document.getElementById('model-sel');
const progressWrap  = document.getElementById('progress-wrap');
const progressFill  = document.getElementById('progress-fill');
const progressLabel = document.getElementById('progress-label');
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

const hashCode = location.hash.slice(1);
if (hashCode) {
  startingText.value = decodeURIComponent(hashCode);
  sampleSel.value = '';
} else {
  sampleSel.value = 'passwordStorage';
  startingText.value = codeSamples.passwordStorage.code;
}

function updateHash() {
  const val = startingText.value;
  window.history.replaceState(null, '', val ? '#' + encodeURIComponent(val) : location.pathname + location.search);
}

sampleSel.addEventListener('change', () => {
  if (sampleSel.value) startingText.value = codeSamples[sampleSel.value].code;
  updateHash();
});

startingText.addEventListener('input', () => {
  sampleSel.value = '';
  updateHash();
});

function getTemperature() { return parseFloat(tempSlider.value); }

function rerender() {
  if (lastLogits) renderButtons(topKFromProbs(probsFromLogits(lastLogits, getTemperature()), getTopK()));
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
  });
});

function getTopK() { return parseInt(topkSlider.value, 10); }

// ── Load model ────────────────────────────────────────────────────
loadBtn.addEventListener('click', async () => {
  const modelId = modelSel.value;
  const dtype = models.find(m => m.id === modelId)?.dtype ?? 'q4f16';
  loadBtn.disabled = true;
  modelSel.disabled = true;
  progressWrap.hidden = false;

  try {
    setProgress(0, 'Loading tokenizer…');
    tok = await AutoTokenizer.from_pretrained(modelId, {
      progress_callback: makeProgress('Tokenizer'),
    });

    setProgress(0, 'Downloading model weights (cached after first load)…');
    mdl = await AutoModelForCausalLM.from_pretrained(modelId, {
      dtype,
      device: 'webgpu',
      progress_callback: makeProgress('Model'),
    });

    setProgress(100, 'Ready!');
    inputCard.hidden = false;

  } catch (err) {
    setProgress(0, `Error: ${err.message}`);
    progressFill.style.background = '#f85149';
    loadBtn.disabled = false;
    modelSel.disabled = false;
    console.error(err);
  }
});

function makeProgress(label) {
  return (info) => {
    if (!info) return;
    const file = info.file ? ` — ${info.file}` : '';
    if (info.progress != null) {
      setProgress(info.progress, `${label}${file}: ${Math.round(info.progress)}%`);
    } else if (info.status) {
      setProgress(null, `${label}${file}: ${info.status}`);
    }
  };
}

function setProgress(pct, label) {
  if (pct != null) progressFill.style.width = `${pct}%`;
  if (label != null) progressLabel.textContent = label;
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
  explorerCard.hidden = true;
  inputCard.hidden = false;
  tokenGrid.innerHTML = '';
  currentText = '';
  templatePrefix = '';
  chatMode = false;
  lastLogits = null;
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
  tokenGrid.classList.add('loading');
  hint.textContent = 'Computing token probabilities…';

  if (chatMode) {
    accText.innerHTML =
      `<span style="opacity:0.3">${escHtml(templatePrefix)}</span>${escHtml(currentText)}`;
  } else {
    accText.textContent = currentText;
  }

  try {
    const tokens = await getTopTokens(currentText, getTopK());
    renderButtons(tokens);
    hint.textContent = 'Click a token to extend the text.';
  } catch (err) {
    hint.textContent = `Error: ${err.message}`;
    console.error(err);
  } finally {
    tokenGrid.classList.remove('loading');
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

async function getTopTokens(text, k) {
  const fullText = chatMode ? templatePrefix + text : text;
  const inputs = tok(fullText, {
    return_tensors: 'pt',
    add_special_tokens: false,
  });

  const output = await mdl(inputs);
  const logits = output.logits; // Tensor [1, seq_len, vocab_size]

  const vocabSize = logits.dims[2];
  const seqLen    = logits.dims[1];
  const data      = logits.data; // Float32Array
  const offset    = (seqLen - 1) * vocabSize;

  lastLogits = data.slice(offset, offset + vocabSize);
  const probs = probsFromLogits(lastLogits, getTemperature());
  return topKFromProbs(probs, k);
}

// ── Render token buttons ───────────────────────────────────────────
function renderButtons(tokens) {
  tokenGrid.innerHTML = '';
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


