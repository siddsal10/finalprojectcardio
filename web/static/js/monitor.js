// Live "patient monitor" heartbeat: a sweeping ECG trace drawn on a canvas.
// Purely illustrative (a synthetic beat at the given heart rate), never a recording.

const WAVES = [ // phase in the beat (0-1), amplitude, width
  [0.12, 0.12, 0.028], [0.235, -0.13, 0.011], [0.26, 1.0, 0.013], [0.29, -0.28, 0.012], [0.52, 0.28, 0.055],
];
const beatShape = (ph) => WAVES.reduce((a, [c, amp, w]) => a + amp * Math.exp(-0.5 * ((ph - c) / w) ** 2), 0);

/**
 * mountMonitor(el, {bpm, color, speed, height, onBeat}) -> {setBpm, stop}
 * el gets a canvas that fills its width.
 */
export function mountMonitor(el, { bpm = 72, color = "#e5355c", speed = 120, height = 90, onBeat, glow = true } = {}) {
  const canvas = document.createElement("canvas");
  canvas.style.width = "100%";
  canvas.style.height = `${height}px`;
  canvas.style.display = "block";
  el.appendChild(canvas);
  const ctx = canvas.getContext("2d");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let W = 0, H = 0, dpr = 1, x = 0, lastY = null, t = 0, last = performance.now(), raf = 0, visible = true;
  let rate = bpm, lastBeat = -1;

  const size = () => {
    dpr = window.devicePixelRatio || 1;
    W = el.clientWidth || 300; H = height;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    x = 0; lastY = null;
    if (reduce) drawStatic();
  };
  const yAt = (time) => {
    const period = 60 / rate;
    const ph = (time % period) / period;
    return H * 0.62 - beatShape(ph) * H * 0.48;
  };
  function drawStatic() {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    for (let px = 0; px < W; px++) { const y = yAt(px / speed); px ? ctx.lineTo(px, y) : ctx.moveTo(px, y); }
    ctx.stroke();
  }
  function frame(now) {
    raf = requestAnimationFrame(frame);
    if (!el.isConnected) return stop();
    if (!visible || reduce) { last = now; return; }
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const steps = Math.max(1, Math.round(dt * speed));
    for (let i = 0; i < steps; i++) {
      t += dt / steps;
      x += (dt * speed) / steps;
      if (x > W) { x = 0; lastY = null; }
      const y = yAt(t);
      ctx.clearRect(x, 0, 14, H); // erase ahead of the beam, like a real monitor
      if (lastY !== null) {
        ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.lineCap = "round";
        ctx.beginPath(); ctx.moveTo(x - (dt * speed) / steps, lastY); ctx.lineTo(x, y); ctx.stroke();
      }
      lastY = y;
      const beat = Math.floor(t / (60 / rate));
      const ph = (t % (60 / rate)) / (60 / rate);
      if (beat !== lastBeat && ph > 0.26) { lastBeat = beat; onBeat && onBeat(); }
    }
    if (glow && lastY !== null) { // beam head
      ctx.save(); ctx.fillStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(x, lastY, 2.6, 0, Math.PI * 2); ctx.fill(); ctx.restore();
    }
  }
  function stop() { cancelAnimationFrame(raf); ro.disconnect(); io.disconnect(); }
  const ro = new ResizeObserver(size);
  ro.observe(el);
  const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; });
  io.observe(el);
  size();
  raf = requestAnimationFrame(frame);
  return { setBpm: (b) => { rate = b; }, stop };
}

/** A heart-rate readout that pulses on every beat. */
export function bpmPulse(numEl) {
  return () => { numEl.classList.remove("beat"); void numEl.offsetWidth; numEl.classList.add("beat"); };
}
