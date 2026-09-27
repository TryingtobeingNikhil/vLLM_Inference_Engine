'use client';

import { useEffect, useRef } from 'react';

const PALETTE = ['#4ADE80', '#60A5FA', '#FBBF24', '#A78BFA', '#FB7185', '#22D3EE'];
const PITCH = 22;   // cell + gap, px
const CELL = 17;
const RADIUS = 3;

interface Run {
  cells: number[];
  color: number;
  born: number;
  fillMs: number;  // time to page in every block
  holdMs: number;
  freeMs: number;
}

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
const RGB = PALETTE.map(hexToRgb);

/**
 * The hero backdrop: a KV-cache block pool you can play with. Ambient
 * "sequences" page runs of blocks in and out; the cursor allocates a fading
 * trail; a click fires an allocation ripple.
 */
export function BlockField({ className = '' }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let cols = 0, rows = 0, dpr = 1, w = 0, h = 0;
    let trail = new Float32Array(0);
    let trailColor = new Uint8Array(0);
    let base: HTMLCanvasElement | null = null;
    let runs: Run[] = [];
    const ripples: { x: number; y: number; t0: number; color: number }[] = [];
    let mouse = { x: -9999, y: -9999, active: false };
    let lastCell = -1;
    let cursorColor = 0;
    let cellsMoved = 0;
    let raf = 0;
    let visible = true;
    let lastSpawn = 0;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = rect.width; h = rect.height;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      cols = Math.ceil(w / PITCH) + 1;
      rows = Math.ceil(h / PITCH) + 1;
      trail = new Float32Array(cols * rows);
      trailColor = new Uint8Array(cols * rows);
      runs = [];

      // Pre-render the idle grid once.
      base = document.createElement('canvas');
      base.width = canvas.width; base.height = canvas.height;
      const b = base.getContext('2d')!;
      b.scale(dpr, dpr);
      b.fillStyle = 'rgba(255,255,255,0.028)';
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          b.beginPath();
          b.roundRect(c * PITCH, r * PITCH, CELL, CELL, RADIUS);
          b.fill();
        }
    };

    const spawnRun = (now: number) => {
      const len = 3 + Math.floor(Math.random() * 7);
      const r = Math.floor(Math.random() * rows);
      const c0 = Math.floor(Math.random() * Math.max(1, cols - len));
      const cells: number[] = [];
      // Mostly contiguous, occasionally wraps to the next row like a real page table.
      for (let i = 0; i < len; i++) {
        const c = c0 + i;
        cells.push(r * cols + (c % cols));
      }
      runs.push({
        cells,
        color: Math.floor(Math.random() * PALETTE.length),
        born: now,
        fillMs: 90 * len,
        holdMs: 1400 + Math.random() * 2600,
        freeMs: 700,
      });
    };

    const cellAt = (x: number, y: number) => {
      const c = Math.floor(x / PITCH), r = Math.floor(y / PITCH);
      if (c < 0 || r < 0 || c >= cols || r >= rows) return -1;
      return r * cols + c;
    };

    const drawCell = (idx: number, rgb: [number, number, number], a: number, glow = false) => {
      const c = idx % cols, r = (idx / cols) | 0;
      ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`;
      if (glow) { ctx.shadowColor = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a * 0.9})`; ctx.shadowBlur = 14; }
      ctx.beginPath();
      ctx.roundRect(c * PITCH, r * PITCH, CELL, CELL, RADIUS);
      ctx.fill();
      if (glow) ctx.shadowBlur = 0;
    };

    const frame = (now: number) => {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      // A zero-size canvas (hidden tab/pane at load) can't be drawn; wait for the ResizeObserver.
      if (base && base.width > 0 && base.height > 0) ctx.drawImage(base, 0, 0);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // Ambient sequences
      if (!reduced && now - lastSpawn > 420 && runs.length < Math.max(6, (cols * rows) / 120)) {
        spawnRun(now);
        lastSpawn = now;
      }
      runs = runs.filter((run) => now - run.born < run.fillMs + run.holdMs + run.freeMs);
      for (const run of runs) {
        const t = now - run.born;
        const filled = Math.min(run.cells.length, Math.floor((t / run.fillMs) * run.cells.length) + 1);
        const fade = t > run.fillMs + run.holdMs ? 1 - (t - run.fillMs - run.holdMs) / run.freeMs : 1;
        for (let i = 0; i < filled; i++) {
          const head = i === filled - 1 && t < run.fillMs;
          drawCell(run.cells[i], RGB[run.color], (head ? 0.55 : 0.2) * Math.max(0, fade), head);
        }
      }

      // Ripples: a ring of freshly allocated blocks expanding from the click
      for (let k = ripples.length - 1; k >= 0; k--) {
        const rp = ripples[k];
        const age = now - rp.t0;
        const radius = age * 0.55;
        if (radius > 420) { ripples.splice(k, 1); continue; }
        const r0 = Math.max(0, Math.floor((rp.y - radius) / PITCH)), r1 = Math.min(rows - 1, Math.ceil((rp.y + radius) / PITCH));
        const c0 = Math.max(0, Math.floor((rp.x - radius) / PITCH)), c1 = Math.min(cols - 1, Math.ceil((rp.x + radius) / PITCH));
        for (let r = r0; r <= r1; r++)
          for (let c = c0; c <= c1; c++) {
            const d = Math.hypot(c * PITCH + CELL / 2 - rp.x, r * PITCH + CELL / 2 - rp.y);
            if (Math.abs(d - radius) < PITCH * 0.7) {
              const idx = r * cols + c;
              trail[idx] = Math.max(trail[idx], 0.75 * (1 - radius / 420));
              trailColor[idx] = rp.color;
            }
          }
      }

      // Cursor trail + proximity lift
      for (let i = 0; i < trail.length; i++) {
        if (trail[i] > 0.01) {
          drawCell(i, RGB[trailColor[i]], trail[i] * 0.7, trail[i] > 0.6);
          trail[i] *= 0.955;
        } else trail[i] = 0;
      }
      if (mouse.active) {
        const R = 130;
        const r0 = Math.max(0, Math.floor((mouse.y - R) / PITCH)), r1 = Math.min(rows - 1, Math.ceil((mouse.y + R) / PITCH));
        const c0 = Math.max(0, Math.floor((mouse.x - R) / PITCH)), c1 = Math.min(cols - 1, Math.ceil((mouse.x + R) / PITCH));
        for (let r = r0; r <= r1; r++)
          for (let c = c0; c <= c1; c++) {
            const d = Math.hypot(c * PITCH + CELL / 2 - mouse.x, r * PITCH + CELL / 2 - mouse.y);
            if (d < R) drawCell(r * cols + c, [255, 255, 255], 0.06 * (1 - d / R));
          }
      }

      if (visible && !reduced) raf = requestAnimationFrame(frame);
    };

    const onMove = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouse = { x: e.clientX - rect.left, y: e.clientY - rect.top, active: true };
      if (mouse.y < 0 || mouse.y > h) { mouse.active = false; return; }
      const idx = cellAt(mouse.x, mouse.y);
      if (idx >= 0 && idx !== lastCell) {
        lastCell = idx;
        // Every ~28 blocks the cursor "becomes" a new sequence with its own colour.
        if (++cellsMoved % 28 === 0) cursorColor = (cursorColor + 1) % PALETTE.length;
        trail[idx] = 1;
        trailColor[idx] = cursorColor;
      }
    };
    const onLeave = () => { mouse.active = false; cursorColor = (cursorColor + 1) % PALETTE.length; };
    const onDown = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const y = e.clientY - rect.top;
      if (y < 0 || y > h) return;
      // Don't hijack clicks on real controls sitting above the field.
      if ((e.target as HTMLElement).closest('a,button,input,[role="button"]')) return;
      cursorColor = (cursorColor + 1) % PALETTE.length;
      ripples.push({ x: e.clientX - rect.left, y, t0: performance.now(), color: cursorColor });
    };

    resize();
    if (reduced) {
      // One calm static frame with a handful of owned blocks.
      for (let i = 0; i < 10; i++) spawnRun(0);
      runs.forEach((r) => (r.born = -r.fillMs));
      frame(0);
    } else {
      raf = requestAnimationFrame(frame);
    }

    const ro = new ResizeObserver(() => { resize(); if (reduced) frame(0); });
    ro.observe(canvas);
    const io = new IntersectionObserver(([e]) => {
      const was = visible;
      visible = e.isIntersecting;
      if (visible && !was && !reduced) raf = requestAnimationFrame(frame);
    });
    io.observe(canvas);
    // Listen on the whole section so the field reacts even under the headline.
    const host = (canvas.closest('section') ?? canvas.parentElement)!;
    host.addEventListener('pointermove', onMove);
    host.addEventListener('pointerleave', onLeave);
    host.addEventListener('pointerdown', onDown);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      host.removeEventListener('pointermove', onMove);
      host.removeEventListener('pointerleave', onLeave);
      host.removeEventListener('pointerdown', onDown);
    };
  }, []);

  return <canvas ref={canvasRef} className={`pointer-events-none absolute inset-0 h-full w-full ${className}`} aria-hidden="true" />;
}
