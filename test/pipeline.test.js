import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { clean, createModel, prepare, score, verdict } from '../src/pipeline.js';

const read = (path) => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const trained = createModel(read('../model/model.json'));
const fixtures = read('./fixtures.json');

const tiny = createModel({
  intercept: -1,
  ngramMax: 2,
  vocabulary: { a: [2, 1], b: [2, -1], 'a b': [3, 2] },
  slang: { gk: 'tidak', sy: 'saya kira' },
});

for (const fixture of fixtures) {
  test(`matches the Python model: "${fixture.text}"`, () => {
    const { tokens } = prepare(fixture.text, trained);
    assert.deepEqual(tokens, fixture.tokens);
    const result = score(tokens, trained);
    assert.ok(Math.abs(result.logit - fixture.logit) < 0.01, `${result.logit} vs ${fixture.logit}`);
  });
}

test('clean removes links, mentions, punctuation and stretched letters', () => {
  assert.equal(clean('RT @budi: BAGUSSS!!! https://x.co/a lihat www.a.id'), 'bagus lihat');
});

test('slang is expanded and reported', () => {
  const { tokens, changes } = prepare('gk sy', tiny);
  assert.deepEqual(tokens, ['tidak', 'saya', 'kira']);
  assert.deepEqual(changes, [{ from: 'gk', to: 'tidak' }, { from: 'sy', to: 'saya kira' }]);
});

test('the shown numbers add up to the final score', () => {
  const result = score(['a', 'b'], tiny);
  const sum = result.features.reduce((total, f) => total + f.contribution, 0);
  assert.ok(Math.abs(result.intercept + sum - result.logit) < 1e-12);
  assert.ok(Math.abs(1 / (1 + Math.exp(-result.logit)) - result.probability) < 1e-12);
  assert.ok(Math.abs(result.features.reduce((total, f) => total + f.weight ** 2, 0) - 1) < 1e-12);
});

test('repeated words use sublinear term frequency', () => {
  const feature = score(['a', 'a'], tiny).features.find((f) => f.term === 'a');
  assert.equal(feature.count, 2);
  assert.ok(Math.abs(feature.tf - (1 + Math.log(2))) < 1e-12);
});

test('words outside the vocabulary are reported and change nothing', () => {
  const result = score(['zzz'], tiny);
  assert.deepEqual(result.features, []);
  assert.deepEqual(result.unknownTokens, ['zzz']);
  assert.equal(result.logit, -1);
});

test('object property names are ordinary words', () => {
  const { tokens } = prepare('constructor', trained);
  assert.deepEqual(tokens, ['constructor']);
  assert.equal(Number.isNaN(score(tokens, tiny).logit), false);
});

test('verdict has an uncertain band', () => {
  assert.equal(verdict(0.9).id, 'toxic');
  assert.equal(verdict(0.6).id, 'toxic');
  assert.equal(verdict(0.5).id, 'unsure');
  assert.equal(verdict(0.4).id, 'clean');
});
