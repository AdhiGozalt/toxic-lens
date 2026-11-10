// Dashboard UI. User text reaches the page through textContent only.
import { createModel, prepare, score, verdict } from './pipeline.js';

const EXAMPLES = [
  ['Insult', 'Dasar bodoh, kamu tidak berguna!!!'],
  ['Friendly', 'Terima kasih banyak ya kak, sangat membantu'],
  ['Slang', 'gue gak suka sama lo bangeeet'],
  ['Political', 'Pemerintah harus lebih peduli pada rakyat kecil'],
];
const MAX_BARS = 12;
const SVG = 'http://www.w3.org/2000/svg';

const $ = (id) => document.getElementById(id);
const percent = (p) => `${(p * 100).toFixed(1)}%`;
const signed = (n) => `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(3)}`;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

let model = null;
let prepared = null;
let fullResult = null;
const removed = new Set();

function renderChanges(changes) {
  const items = changes.map(({ from, to }) => el('span', 'chip', `${from} → ${to}`));
  $('changes').replaceChildren(...(items.length ? items : [el('span', 'empty', 'No slang found in this comment.')]));
}

function renderTokens(focusIndex) {
  const buttons = prepared.tokens.map((token, i) => {
    const button = el('button', 'chip', token);
    button.type = 'button';
    button.setAttribute('aria-pressed', String(removed.has(i)));
    button.addEventListener('click', () => {
      if (removed.has(i)) removed.delete(i);
      else removed.add(i);
      recalculate(i);
    });
    return button;
  });
  $('tokens').replaceChildren(...buttons);
  $('reset').hidden = removed.size === 0;
  if (focusIndex !== undefined) buttons[focusIndex]?.focus();
}

function renderLookup(result) {
  const { features, unknownTokens } = result;
  $('known').textContent = `${features.length} ${features.length === 1 ? 'term' : 'terms'} recognised`;
  $('known-note').textContent = unknownTokens.length ? `· ${unknownTokens.length} not in the vocabulary:` : '· every word is known.';
  $('unknown').replaceChildren(...unknownTokens.map((t) => el('span', 'chip', t)));
}

function renderTable(features) {
  const head = el('tr');
  for (const title of ['Term', 'Count', 'TF', 'IDF', 'Weight', 'Learned', 'Contribution']) head.append(el('th', '', title));
  const rows = features.map((f) => {
    const row = el('tr');
    for (const value of [f.term, f.count, f.tf.toFixed(3), f.idf.toFixed(3), f.weight.toFixed(3), signed(f.coefficient), signed(f.contribution)]) row.append(el('td', '', value));
    return row;
  });
  $('features').replaceChildren(el('thead'), el('tbody'));
  $('features').tHead.append(head);
  $('features').tBodies[0].append(...rows);
}

function renderBars(features) {
  const shown = features.slice(0, MAX_BARS);
  const max = Math.max(...shown.map((f) => Math.abs(f.contribution)), 0.001);
  $('bars').replaceChildren(...shown.map((f) => {
    const row = el('div', 'bar-row');
    row.title = `${f.term}: ${signed(f.contribution)}`;
    const left = el('div', 'bar-side left');
    const right = el('div', 'bar-side right');
    const up = f.contribution >= 0;
    const fill = el('i', `bar-fill ${up ? 'up' : 'down'}`);
    fill.style.width = `${(Math.abs(f.contribution) / max) * 100}%`;
    (up ? right : left).append(fill);
    row.append(el('span', 'bar-term', f.term), left, right, el('span', 'bar-value', signed(f.contribution)));
    return row;
  }));
  if (shown.length === 0) $('bars').append(el('p', 'empty', 'No recognised terms.'));
  const hidden = features.length - shown.length;
  $('bars-note').textContent = hidden > 0 ? `Showing the ${MAX_BARS} strongest of ${features.length} terms. Step 5 lists them all.` : '';
}

function svgNode(tag, attributes, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderCurve(logit, probability) {
  const left = 30;
  const right = 350;
  const top = 18;
  const bottom = 140;
  const range = 6;
  const x = (v) => left + ((Math.min(range, Math.max(-range, v)) + range) / (2 * range)) * (right - left);
  const y = (p) => bottom - p * (bottom - top);
  let path = '';
  for (let v = -range; v <= range + 1e-9; v += 0.1) path += `${path ? 'L' : 'M'}${x(v).toFixed(1)} ${y(1 / (1 + Math.exp(-v))).toFixed(1)}`;

  const cx = x(logit);
  const cy = y(probability);
  const labelOnLeft = cx > 100; // Above and to the left stays clear of a rising curve.
  $('curve').replaceChildren(
    svgNode('line', { class: 'axis', x1: left, x2: right, y1: bottom, y2: bottom }),
    svgNode('line', { class: 'threshold', x1: left, x2: right, y1: y(0.5), y2: y(0.5) }),
    svgNode('text', { x: left - 5, y: y(1) + 4, 'text-anchor': 'end' }, '100'),
    svgNode('text', { x: left - 5, y: y(0.5) + 4, 'text-anchor': 'end' }, '50'),
    svgNode('text', { x: left - 5, y: y(0) + 4, 'text-anchor': 'end' }, '0'),
    svgNode('text', { x: left, y: 156 }, `−${range}`),
    svgNode('text', { x: (left + right) / 2, y: 156, 'text-anchor': 'middle' }, 'log-odds'),
    svgNode('text', { x: right, y: 156, 'text-anchor': 'end' }, `+${range}`),
    svgNode('path', { class: 'curve-line', d: path }),
    svgNode('circle', { class: 'marker', cx, cy, r: 6 }),
    svgNode('text', { class: 'marker-label', x: cx + (labelOnLeft ? -11 : 11), y: cy - 9, 'text-anchor': labelOnLeft ? 'end' : 'start' }, percent(probability)),
  );
}

function renderSummary(result) {
  const hasSignal = result.features.length > 0;
  const summary = $('summary');
  if (!hasSignal) {
    summary.className = 'summary';
    $('final').textContent = '–';
    $('verdict').textContent = 'Not enough known words';
    $('delta').textContent = 'The model has nothing to base a score on.';
    return;
  }
  const v = verdict(result.probability);
  summary.className = `summary ${v.id}`;
  $('final').textContent = percent(result.probability);
  $('verdict').textContent = v.label;
  $('delta').textContent = removed.size ? `${percent(fullResult.probability)} with all words` : '';
}

function recalculate(focusIndex) {
  const tokens = prepared.tokens.filter((_, i) => !removed.has(i));
  const result = score(tokens, model);
  renderTokens(focusIndex);
  renderLookup(result);
  renderTable(result.features);
  renderBars(result.features);
  $('intercept').textContent = signed(result.intercept);
  $('sum').textContent = signed(result.contributionTotal);
  $('logit').textContent = signed(result.logit);
  $('probability').textContent = percent(result.probability);
  renderCurve(result.logit, result.probability);
  renderSummary(result);
  $('pill').textContent = `${$('final').textContent} · ${$('verdict').textContent}`;
  $('pill').hidden = false;
}

function analyzeInput() {
  const text = $('text').value;
  removed.clear();
  prepared = prepare(text, model);
  if (prepared.tokens.length === 0) {
    $('steps').hidden = true;
    $('pill').hidden = true;
    $('summary').className = 'summary';
    $('final').textContent = '–';
    $('verdict').textContent = 'Waiting for text';
    $('delta').textContent = '';
    return;
  }
  fullResult = score(prepared.tokens, model);
  $('raw').textContent = text;
  $('cleaned').textContent = prepared.cleaned;
  renderChanges(prepared.changes);
  recalculate();
  $('steps').hidden = false;
}

function renderReport(metrics) {
  const { test, sizes, majorityBaseline } = metrics;
  const tiles = [
    [percent(test.accuracy), `Accuracy (always guessing the majority: ${percent(majorityBaseline)})`],
    [percent(test.precision), 'Precision: flagged comments that were toxic'],
    [percent(test.recall), 'Recall: toxic comments that were flagged'],
    [test.rocAuc.toFixed(3), 'ROC AUC: ranking quality, 1 is perfect'],
  ];
  $('tiles').replaceChildren(...tiles.map(([value, label]) => {
    const tile = el('div', 'tile');
    tile.append(el('strong', '', value), el('span', '', label));
    return tile;
  }));

  const { tn, fp, fn, tp } = test.confusion;
  const head = el('tr');
  for (const title of [`${sizes.test.toLocaleString('en-US')} comments`, 'Predicted not toxic', 'Predicted toxic']) head.append(el('th', '', title));
  const rows = [['Actually not toxic', tn, fp], ['Actually toxic', fn, tp]].map((cells) => {
    const row = el('tr');
    for (const cell of cells) row.append(el('td', '', typeof cell === 'number' ? cell.toLocaleString('en-US') : cell));
    return row;
  });
  $('confusion').replaceChildren(el('thead'), el('tbody'));
  $('confusion').tHead.append(head);
  $('confusion').tBodies[0].append(...rows);
}

function setExample(text) {
  $('text').value = text;
  analyzeInput();
}

async function start() {
  try {
    const [modelResponse, metricsResponse] = await Promise.all([fetch('model/model.json'), fetch('model/metrics.json')]);
    if (!modelResponse.ok || !metricsResponse.ok) throw new Error('missing model');
    const [modelJson, metrics] = await Promise.all([modelResponse.json(), metricsResponse.json()]);
    model = createModel(modelJson);
    renderReport(metrics);
    $('status').textContent = `${metrics.sizes.features.toLocaleString('en-US')} terms loaded. Results update as you type.`;
    $('text').disabled = false;
    $('examples').replaceChildren(...EXAMPLES.map(([label, text]) => {
      const button = el('button', 'example', label);
      button.type = 'button';
      button.addEventListener('click', () => setExample(text));
      return button;
    }));
    $('text').addEventListener('input', analyzeInput);
    $('reset').addEventListener('click', () => {
      removed.clear();
      recalculate();
    });
    setExample(EXAMPLES[0][1]);
  } catch {
    $('status').textContent = 'The model could not be loaded. Please refresh the page.';
  }
}

start();
