/**
 * simulation.ts — Deterministic scheduler simulation engine.
 *
 * Generates tick-by-tick state traces for all animated visualizations.
 * All timing ratios are derived from real benchmark data (baseline_metrics.json,
 * bench_phases_results.json). No randomness — the same trace always plays.
 *
 * Used by:
 *   - HeroSection (live scheduler panel)
 *   - SchedulerLiveSection (scheduler loop explainer)
 *   - Phase1vs2Section (side-by-side comparison)
 *   - KVCacheSection (block grid animation)
 *   - CPUSwapSection (swap event log)
 */

import { ENGINE_CONFIG } from './benchmarks';

// ── Types ──────────────────────────────────────────────────────────────────────

export type SeqState =
  | 'waiting'
  | 'prefill'
  | 'chunked_prefilling'
  | 'decoding'
  | 'swapped'
  | 'finished';

export interface SimSequence {
  id: string;
  shortId: string;         // e.g. "a1b2" for display
  promptTokens: number;
  maxNewTokens: number;
  state: SeqState;
  tokensGenerated: number;
  blocksAllocated: number;
  ttftMs: number;          // ms when first token was produced (0 until then)
  prefillChunksTotal: number;
  prefillChunksDone: number;
}

export interface SchedulerTick {
  tick: number;
  sequences: SimSequence[];
  queueDepth: number;
  batchSize: number;       // sequences in decoding state
  tokPerSec: number;       // rolling approximate
  freeBlocks: number;
  allocatedBlocks: number;
  swapEvents: SwapEvent[];
}

export interface SwapEvent {
  type: 'swap_out' | 'swap_in';
  seqShortId: string;
  blocksAffected: number;
  reason: string;
}

export interface BlockGridState {
  blocks: BlockState[];    // length = ENGINE_CONFIG.kv_num_blocks
  tick: number;
}

export interface BlockState {
  blockId: number;
  state: 'free' | 'allocated' | 'dirty';
  ownerShortId: string | null;
  tokensUsed: number;
}

// ── Scheduler simulation traces ───────────────────────────────────────────────

// Sequence definitions (realistic prompt/decode sizes):
const DEMO_SEQUENCES = [
  { shortId: 'a1b2', promptTokens: 48, maxNewTokens: 50 },
  { shortId: 'c3d4', promptTokens: 32, maxNewTokens: 40 },
  { shortId: 'e5f6', promptTokens: 64, maxNewTokens: 35 },
  { shortId: 'g7h8', promptTokens: 24, maxNewTokens: 45 },
  { shortId: 'i9j0', promptTokens: 56, maxNewTokens: 30 },
  { shortId: 'k1l2', promptTokens: 40, maxNewTokens: 50 },
];

function blocksNeeded(tokens: number): number {
  return Math.ceil(tokens / ENGINE_CONFIG.kv_block_size);
}

function makeSeq(def: typeof DEMO_SEQUENCES[number]): SimSequence {
  return {
    id: `seq-${def.shortId}`,
    shortId: def.shortId,
    promptTokens: def.promptTokens,
    maxNewTokens: def.maxNewTokens,
    state: 'waiting',
    tokensGenerated: 0,
    blocksAllocated: 0,
    ttftMs: 0,
    prefillChunksTotal: Math.ceil(def.promptTokens / ENGINE_CONFIG.prefill_chunk_size),
    prefillChunksDone: 0,
  };
}

/**
 * Generates N ticks of a Phase 2 continuous batching trace.
 * Sequences flow through: waiting → prefill → decoding → finished.
 * A new sequence is admitted from the "pending pool" whenever a slot opens.
 */
export function buildContinuousBatchingTrace(numTicks = 120): SchedulerTick[] {
  const pending = DEMO_SEQUENCES.map(makeSeq);
  const ticks: SchedulerTick[] = [];
  let running: SimSequence[] = [];
  let queue: SimSequence[] = [];
  let pendingPool = [...pending];
  const maxBatch = ENGINE_CONFIG.max_batch_size;
  let totalFreeBlocks: number = ENGINE_CONFIG.kv_num_blocks;

  // Pre-seed queue with first 2 sequences
  queue.push(pendingPool.shift()!);
  queue.push(pendingPool.shift()!);

  for (let tick = 0; tick < numTicks; tick++) {
    // Every 15 ticks, add a new sequence to the queue if pool has one
    if (tick > 0 && tick % 15 === 0 && pendingPool.length > 0) {
      queue.push(pendingPool.shift()!);
    }

    // Evict finished sequences
    const finished = running.filter(s => s.state === 'finished');
    for (const s of finished) {
      const freed = blocksNeeded(s.promptTokens + s.tokensGenerated);
      totalFreeBlocks = Math.min(ENGINE_CONFIG.kv_num_blocks, totalFreeBlocks + freed);
    }
    running = running.filter(s => s.state !== 'finished');

    // Admit from queue (up to maxBatch)
    while (running.length < maxBatch && queue.length > 0) {
      const seq = { ...queue.shift()! };
      seq.state = 'prefill';
      seq.blocksAllocated = blocksNeeded(seq.promptTokens);
      totalFreeBlocks = Math.max(0, totalFreeBlocks - seq.blocksAllocated);
      running.push(seq);
    }

    // Advance each running sequence
    running = running.map(s => {
      const seq = { ...s };
      if (seq.state === 'prefill') {
        // Prefill takes 2 ticks then transitions to decoding
        seq.prefillChunksDone += 1;
        if (seq.prefillChunksDone >= seq.prefillChunksTotal) {
          seq.state = 'decoding';
          seq.ttftMs = 21.0; // from bench data (approx)
        }
      } else if (seq.state === 'decoding') {
        seq.tokensGenerated += 1;
        // Allocate a new block every kv_block_size tokens
        if (seq.tokensGenerated % ENGINE_CONFIG.kv_block_size === 0) {
          seq.blocksAllocated += 1;
          totalFreeBlocks = Math.max(0, totalFreeBlocks - 1);
        }
        if (seq.tokensGenerated >= seq.maxNewTokens) {
          seq.state = 'finished';
        }
      }
      return seq;
    });

    const decodingSeqs = running.filter(s => s.state === 'decoding');
    const batchSize = decodingSeqs.length;
    const queueDepth = queue.length;
    // tok/s approximation: batchSize × (1 token per ~11ms decode step) × 1000ms
    const tokPerSec = Math.round(batchSize * (1000 / 11.1));

    ticks.push({
      tick,
      sequences: [...running.map(s => ({ ...s })), ...queue.map(s => ({ ...s }))],
      queueDepth,
      batchSize,
      tokPerSec,
      freeBlocks: totalFreeBlocks,
      allocatedBlocks: ENGINE_CONFIG.kv_num_blocks - totalFreeBlocks,
      swapEvents: [],
    });
  }

  return ticks;
}

/**
 * Phase 1 sequential trace — one request at a time.
 * Returns a flat array of ticks showing requests queued behind each other.
 */
export function buildSequentialTrace(numTicks = 120): SchedulerTick[] {
  const seqs = DEMO_SEQUENCES.slice(0, 4).map(makeSeq);
  const ticks: SchedulerTick[] = [];
  let activeIdx = 0;
  let activeSeq = { ...seqs[0], state: 'prefill' as SeqState };
  let queueSeqs = seqs.slice(1).map(s => ({ ...s }));

  for (let tick = 0; tick < numTicks; tick++) {
    if (activeSeq.state === 'prefill') {
      activeSeq = { ...activeSeq, state: 'decoding' };
    } else if (activeSeq.state === 'decoding') {
      activeSeq = {
        ...activeSeq,
        tokensGenerated: activeSeq.tokensGenerated + 1,
      };
      if (activeSeq.tokensGenerated >= activeSeq.maxNewTokens) {
        activeSeq = { ...activeSeq, state: 'finished' };
      }
    } else if (activeSeq.state === 'finished' && queueSeqs.length > 0) {
      activeIdx++;
      activeSeq = { ...queueSeqs.shift()!, state: 'prefill' };
      queueSeqs = [...queueSeqs];
    }

    ticks.push({
      tick,
      sequences: [
        { ...activeSeq },
        ...queueSeqs.map(s => ({ ...s })),
      ],
      queueDepth: queueSeqs.length,
      batchSize: activeSeq.state === 'decoding' ? 1 : 0,
      tokPerSec: activeSeq.state === 'decoding' ? 42 : 0, // from sequential baseline data
      freeBlocks: ENGINE_CONFIG.kv_num_blocks,
      allocatedBlocks: 0,
      swapEvents: [],
    });
  }

  return ticks;
}

// ── Block grid trace ─────────────────────────────────────────────────────────

/**
 * Generates block grid states for the KV-cache visualization.
 * Shows 256 blocks being allocated/freed across 4 sequences cycling.
 */
export function buildBlockGridTrace(numTicks = 80): BlockGridState[] {
  const frames: BlockGridState[] = [];
  const totalBlocks = ENGINE_CONFIG.kv_num_blocks;
  const blockSize = ENGINE_CONFIG.kv_block_size;

  // Each sequence claims a slab of blocks then releases them
  const seqDefs = [
    { shortId: 'a1b2', startBlock: 0,  numBlocks: 4, startTick: 0,  endTick: 40 },
    { shortId: 'c3d4', startBlock: 4,  numBlocks: 3, startTick: 5,  endTick: 45 },
    { shortId: 'e5f6', startBlock: 7,  numBlocks: 5, startTick: 10, endTick: 50 },
    { shortId: 'g7h8', startBlock: 12, numBlocks: 4, startTick: 15, endTick: 55 },
  ];

  for (let tick = 0; tick < numTicks; tick++) {
    const blocks: BlockState[] = Array.from({ length: totalBlocks }, (_, i) => ({
      blockId: i,
      state: 'free' as const,
      ownerShortId: null,
      tokensUsed: 0,
    }));

    for (const seq of seqDefs) {
      // Loop the sequence so animation repeats
      const period = 60;
      const localTick = tick % period;
      const seqStart = seq.startTick % period;
      const seqEnd = seq.endTick % period;
      const active = seqStart <= localTick && localTick < seqEnd;

      if (active) {
        for (let b = 0; b < seq.numBlocks; b++) {
          const blockIdx = seq.startBlock + b;
          if (blockIdx < totalBlocks) {
            const progress = (localTick - seqStart) / (seqEnd - seqStart);
            const usedTokens = Math.round(progress * blockSize);
            blocks[blockIdx] = {
              blockId: blockIdx,
              state: usedTokens >= blockSize ? 'dirty' : 'allocated',
              ownerShortId: seq.shortId,
              tokensUsed: Math.min(usedTokens, blockSize),
            };
          }
        }
      }
    }

    frames.push({ blocks, tick });
  }

  return frames;
}

// ── CPU Swap event log ────────────────────────────────────────────────────────

export interface SwapEventLogEntry {
  timeMs: number;
  type: 'admit' | 'oom' | 'swap_out' | 'alloc' | 'resume' | 'decode' | 'done';
  message: string;
  seqId?: string;
  blocks?: number;
}

/**
 * Deterministic replay of a burst-load scenario showing CPU swap path.
 * Based on the scheduler._try_swap_out_victim() logic in scheduler.py.
 */
export const CPU_SWAP_EVENT_LOG: SwapEventLogEntry[] = [
  { timeMs: 0,    type: 'admit',    message: 'seq-e5f6 arrives → request admitted to queue',    seqId: 'e5f6' },
  { timeMs: 80,   type: 'admit',    message: 'seq-e5f6 prefill start → allocate 7 blocks',       seqId: 'e5f6', blocks: 7 },
  { timeMs: 120,  type: 'oom',      message: 'OutOfBlocksError: 7 requested, 5 available',       seqId: 'e5f6' },
  { timeMs: 140,  type: 'swap_out', message: 'SWAP_OUT: victim=seq-c3d4 (12 blocks, largest)',   seqId: 'c3d4', blocks: 12 },
  { timeMs: 180,  type: 'swap_out', message: 'KV tensors copied to CPU staging pool',            seqId: 'c3d4', blocks: 12 },
  { timeMs: 210,  type: 'alloc',    message: 'seq-e5f6 retry → allocate 7 blocks (success)',     seqId: 'e5f6', blocks: 7 },
  { timeMs: 250,  type: 'decode',   message: 'seq-e5f6 decoding…',                               seqId: 'e5f6' },
  { timeMs: 1100, type: 'resume',   message: 'seq-e5f6 done → seq-c3d4 SWAP_IN: GPU blocks restored', seqId: 'c3d4', blocks: 12 },
  { timeMs: 1150, type: 'decode',   message: 'seq-c3d4 resumed decoding from token 18',          seqId: 'c3d4' },
  { timeMs: 1560, type: 'done',     message: 'seq-c3d4 finished (50 tokens). 0 requests dropped.', seqId: 'c3d4' },
];

// ── Phase 1 vs Phase 2 timeline data ──────────────────────────────────────────

export interface TimelineBar {
  reqId: string;
  label: string;
  waitStartMs: number;   // when request arrived
  prefillStartMs: number;
  decodeStartMs: number; // = first token time
  finishMs: number;
  ttftMs: number;        // relative to waitStart
  totalMs: number;
}

/**
 * Phase 1: sequential - each request waits for the prior one to fully finish.
 * Numbers derived from baseline_metrics.json.
 */
export const PHASE1_TIMELINE: TimelineBar[] = [
  { reqId: 'req-1', label: 'Req 1', waitStartMs: 0,      prefillStartMs: 0,      decodeStartMs: 1209.6, finishMs: 2357,  ttftMs: 1209.6, totalMs: 2357 },
  { reqId: 'req-2', label: 'Req 2', waitStartMs: 0,      prefillStartMs: 2357,   decodeStartMs: 3573,   finishMs: 4755,  ttftMs: 3573,   totalMs: 4755 },
  { reqId: 'req-3', label: 'Req 3', waitStartMs: 0,      prefillStartMs: 4755,   decodeStartMs: 5771,   finishMs: 6966,  ttftMs: 5771,   totalMs: 6966 },
  { reqId: 'req-4', label: 'Req 4', waitStartMs: 0,      prefillStartMs: 6966,   decodeStartMs: 7982,   finishMs: 9161,  ttftMs: 7982,   totalMs: 9161 },
];

/**
 * Phase 2: continuous batching - all requests run concurrently.
 * Numbers derived from bench_direct_results.json concurrent section.
 */
export const PHASE2_TIMELINE: TimelineBar[] = [
  { reqId: 'req-1', label: 'Req 1', waitStartMs: 0, prefillStartMs: 0,    decodeStartMs: 21.9,  finishMs: 541, ttftMs: 21.9,  totalMs: 541 },
  { reqId: 'req-2', label: 'Req 2', waitStartMs: 0, prefillStartMs: 21.9, decodeStartMs: 52.9,  finishMs: 541, ttftMs: 52.9,  totalMs: 541 },
  { reqId: 'req-3', label: 'Req 3', waitStartMs: 0, prefillStartMs: 52.9, decodeStartMs: 67.4,  finishMs: 541, ttftMs: 67.4,  totalMs: 541 },
  { reqId: 'req-4', label: 'Req 4', waitStartMs: 0, prefillStartMs: 67.4, decodeStartMs: 83.7,  finishMs: 541, ttftMs: 83.7,  totalMs: 541 },
];
