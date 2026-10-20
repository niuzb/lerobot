// Run with Node; no browser, HTTP server, camera or motor is accessed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../../src/lerobot/sim2real/pose_capture.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const notReady = '请点击停止跟随、停稳 1 秒，并让两路完整看到棋盘格';
const rejected = message => ({ok: false, status: 422, json: async () => ({message})});
const saved = () => ({ok: true, json: async () => ({message: '已保存第 1 个姿态（fit）'})});

async function harness(reply) {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {textContent: '', disabled: false, value: 'fit', command: ''});
    return elements.get(id);
  };
  const state = {following: false, fault: null, follower: [0, 0, 0, 0, 0, 0], capture_ready: true,
    counts: {fit: 0, check: 0}, cameras: {front: {found: true}, side: {found: true}}};
  const calls = [];
  let now = 0;
  const context = vm.createContext({
    document: {getElementById: element, addEventListener() {}},
    addEventListener() {}, setInterval() {},
    performance: {now: () => now},
    AbortSignal: {timeout: ms => ({timeoutMs: ms})},
    setTimeout(callback, delay) {
      if (callback === context.refresh) return 0; // No real polling loop in this harness.
      now += delay;
      queueMicrotask(callback);
      return 0;
    },
    async fetch(url, options) {
      if (url === '/status') return {ok: true, json: async () => state};
      const request = JSON.parse(options.body);
      calls.push(request);
      return reply(calls.length, request, options);
    },
  });
  context.window = context;
  vm.runInContext(script, context);
  await context.refresh();
  return {context, element, calls, state, now: () => now};
}

(async () => {
  const inspection = await harness(() => saved());
  assert.equal(inspection.element('inspectGripper').disabled, false);
  inspection.state.armed = true;
  await inspection.context.refresh();
  assert.equal(inspection.element('inspectGripper').disabled, true);
  inspection.state.armed = false;
  inspection.state.torque_owned = true;
  await inspection.context.refresh();
  assert.equal(inspection.element('inspectGripper').disabled, true);

  // A readiness race gets bounded retries, and exactly one successful write.
  let writes = 0;
  let h = await harness(n => n < 3 ? rejected(notReady) : (writes++, saved()));
  const inline = html.match(/id="save"[^>]*onclick="([^"]+)"/)[1];
  await vm.runInContext(`(function(){return ${inline}}).call(document.getElementById('save'))`, h.context);
  assert.equal(h.calls.length, 3);
  assert.equal(writes, 1);
  assert.match(h.element('message').textContent, /已保存第 1/);
  assert.equal(h.element('save').disabled, false);

  // An indefinitely stale camera stops; no infinite loop or stuck controls.
  h = await harness(() => rejected(notReady));
  h.state.capture_ready = false;
  await h.context.refresh();
  assert.equal(h.element('save').disabled, false); // Frame age must not make the click target flicker.
  await h.context.runPoseCommand('capture', {split: 'fit'});
  assert.equal(h.calls.length, 20);
  assert.equal(h.now(), 5000);
  assert.match(h.element('message').textContent, /本次未保存/);
  assert.equal(h.element('split').disabled, false);

  // A duplicate pose, fault, or network timeout must never be retried.
  for (const failure of [
    () => rejected('姿态与已有记录太接近'),
    () => rejected('feedback unavailable'),
    () => {const e = new Error('timeout'); e.name = 'TimeoutError'; throw e;},
    () => {throw new TypeError('connection lost');},
  ]) {
    h = await harness(failure);
    await h.context.runPoseCommand('capture', {split: 'fit'});
    assert.equal(h.calls.length, 1);
    assert.equal(h.element('save').disabled, false);
  }

  // Double click and concurrent status refresh cannot allow another command.
  let finish;
  h = await harness(() => new Promise(resolve => {finish = resolve;}));
  const pending = h.context.runPoseCommand('capture', {split: 'fit'});
  await h.context.refresh();
  assert.equal(h.element('save').disabled, true);
  assert.equal(h.element('start').disabled, true);
  assert.match(h.element('captureHint').textContent, /最多等待 5 秒/);
  await h.context.runPoseCommand('capture', {split: 'check'});
  assert.equal(h.calls.length, 1);
  finish(saved());
  await pending;
  assert.equal(h.element('save').disabled, false);
  assert.ok(h.calls.every(request => request.op === 'capture'));
  console.log('PASS: readiness race, 5-second bound, terminal failures, double clicks and busy polling');
})().catch(error => {console.error(error); process.exitCode = 1;});
