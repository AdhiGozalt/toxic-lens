// The scoring pipeline, step by step. Must stay in step with training/train.py;
// test/pipeline.test.js checks both against the same reference sentences.

export function createModel(json) {
  return {
    intercept: json.intercept,
    ngramMax: json.ngramMax,
    vocabulary: new Map(Object.entries(json.vocabulary)),
    slang: new Map(Object.entries(json.slang)),
  };
}

export function clean(text) {
  return text
    .toLowerCase()
    .replace(/\\x[0-9a-f]{2}/g, ' ')
    .replace(/\\n/g, ' ')
    .replace(/https?:\/\/\S+|www\.\S+/g, ' ')
    .replace(/@\w+/g, ' ')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\b(rt|user|url)\b/g, ' ')
    .replace(/([a-z])\1{2,}/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
}

export function normalize(cleaned, model) {
  const tokens = [];
  const changes = [];
  for (const token of cleaned.split(' ').filter(Boolean)) {
    const replacement = model.slang.get(token);
    if (replacement) changes.push({ from: token, to: replacement });
    tokens.push(...(replacement ?? token).split(' '));
  }
  return { tokens, changes };
}

export function prepare(text, model) {
  const cleaned = clean(text);
  return { raw: text, cleaned, ...normalize(cleaned, model) };
}

function ngrams(tokens, max) {
  const terms = [...tokens];
  for (let size = 2; size <= max; size++) {
    for (let i = 0; i + size <= tokens.length; i++) terms.push(tokens.slice(i, i + size).join(' '));
  }
  return terms;
}

/** Scores a token list, keeping every intermediate number for the dashboard. */
export function score(tokens, model) {
  const counts = new Map();
  for (const term of ngrams(tokens, model.ngramMax)) counts.set(term, (counts.get(term) || 0) + 1);

  const features = [];
  for (const [term, count] of counts) {
    const entry = model.vocabulary.get(term);
    if (!entry) continue;
    const [idf, coefficient] = entry;
    const tf = 1 + Math.log(count);
    features.push({ term, count, tf, idf, tfidf: tf * idf, coefficient });
  }

  const norm = Math.sqrt(features.reduce((sum, f) => sum + f.tfidf ** 2, 0));
  let contributionTotal = 0;
  for (const f of features) {
    f.weight = norm ? f.tfidf / norm : 0;
    f.contribution = f.weight * f.coefficient;
    contributionTotal += f.contribution;
  }
  features.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  const known = new Set(features.map((f) => f.term));
  const unknownTokens = [...new Set(tokens.filter((t) => !known.has(t)))];
  const logit = model.intercept + contributionTotal;
  return { features, unknownTokens, norm, intercept: model.intercept, contributionTotal, logit, probability: 1 / (1 + Math.exp(-logit)) };
}

export function verdict(probability) {
  if (probability >= 0.6) return { id: 'toxic', label: 'Likely toxic' };
  if (probability > 0.4) return { id: 'unsure', label: 'Uncertain' };
  return { id: 'clean', label: 'Likely not toxic' };
}
