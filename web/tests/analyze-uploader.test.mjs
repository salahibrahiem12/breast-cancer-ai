import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import ts from 'typescript';

const loadModule = createRequire(import.meta.url);

function harness() {
  const slots = [];
  const pending = [];
  const timers = new Map();
  let cursor = 0;
  let timerId = 0;
  const hooks = {
    useState(initial) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = initial;
      return [slots[index], value => { slots[index] = value; }];
    },
    useRef(initial) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = { current: initial };
      return slots[index];
    },
    useEffect() {},
  };
  const compiledModule = { exports: {} };
  const source = ts.transpileModule(
    fs.readFileSync(new URL('../components/analyze-uploader.tsx', import.meta.url), 'utf8'),
    { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2021 } },
  ).outputText;
  vm.runInNewContext(source, {
    module: compiledModule, exports: compiledModule.exports,
    require: name => name === 'react' ? hooks : loadModule(name),
    AbortController, process,
    FormData: class { append() {} },
    fetch: (url, options) => {
      if (String(url).endsWith('/api/validate-image')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            is_valid: true,
            detected_modality: 'breast_ultrasound',
            supported_modality: 'breast_ultrasound',
            stages_passed: [
              'stage1_file_validation',
              'stage2_modality_classification',
              'stage3_quality_control',
              'stage4_ood_assessment',
            ],
            rejection_reason: null,
            message: 'Valid breast ultrasound',
            quality_metrics: {},
            dimensions: { width: 256, height: 256 },
          }),
        });
      }
      return new Promise((resolve, reject) => {
        pending.push({ resolve, reject, signal: options.signal });
      });
    },
    setTimeout: (callback, delay) => { timers.set(++timerId, { callback, delay }); return timerId; },
    clearTimeout: id => timers.delete(id),
  });
  function nodes(node) {
    if (!node || typeof node !== 'object') return [];
    return [node, ...[node.props?.children].flat(Infinity).flatMap(nodes)];
  }
  function render() {
    cursor = 0;
    return nodes(compiledModule.exports.AnalyzeUploader());
  }
  return {
    pending, timers,
    choose(name) {
      render().find(n => n.type === 'input').props.onChange({ target: { files: [{ name, type: 'image/png', size: 100 }] } });
    },
    submit: () => render().find(n => n.type === 'form').props.onSubmit({ preventDefault() {} }),
    text: () => render().flatMap(n => [n.props?.children].flat(Infinity)).filter(x => typeof x === 'string').join(' '),
    button: () => render().find(n => n.type === 'button').props,
  };
}

function respond(request, message) {
  request.resolve({
    ok: true,
    status: 200,
    json: async () => ({
      request_id: 'test',
      model_version: 'research-prototype-v2',
      detected_modality: 'breast_ultrasound',
      status: 'unable_to_assess',
      input_suitability: 'ultrasound_like',
      abstention_reason: 'classification_disabled_by_server',
      message,
      classification: null,
      confidence: null,
      probabilities: null,
      gradcam_overlay_url: null,
      explanation: null,
      segmentation: null,
      scientific_limitations: [],
      research_disclaimer: 'For research only',
    }),
  });
}

test('changing the image cancels the previous request and ignores its late response', async () => {
  const ui = harness();
  ui.choose('first.png');
  const first = ui.submit();
  ui.choose('second.png');
  assert.equal(ui.pending[0].signal.aborted, true);
  const second = ui.submit();
  respond(ui.pending[0], 'OLD IMAGE RESULT');
  await first;
  assert.equal(ui.text().includes('OLD IMAGE RESULT'), false);
  assert.equal(ui.button().disabled, true);
  respond(ui.pending[1], 'CURRENT IMAGE RESULT');
  await second;
  assert.equal(ui.text().includes('CURRENT IMAGE RESULT'), true);
  assert.equal(ui.button().disabled, false);
  assert.equal(ui.timers.size, 0);
});

test('a timed-out request releases the form and displays a retry message', async () => {
  const ui = harness();
  ui.choose('scan.png');
  const request = ui.submit();
  const timer = [...ui.timers.values()].find(value => value.delay === 60_000);
  timer.callback();
  assert.equal(ui.pending[0].signal.aborted, true);
  ui.pending[0].reject(new Error('aborted'));
  await request;
  assert.equal(ui.text().includes('timed out'), true);
  assert.equal(ui.button().disabled, false);
  assert.equal(ui.timers.size, 0);
});

test('duplicate submission starts only one request', async () => {
  const ui = harness();
  ui.choose('scan.png');
  const first = ui.submit();
  const second = ui.submit();
  assert.equal(ui.pending.length, 1);
  respond(ui.pending[0], 'READY');
  await Promise.all([first, second]);
  assert.equal(ui.button().disabled, false);
  assert.equal(ui.timers.size, 0);
});
