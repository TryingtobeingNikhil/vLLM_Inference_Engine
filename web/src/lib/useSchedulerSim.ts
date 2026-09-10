'use client';

import { useReducer, useCallback, useEffect, useRef } from 'react';
import {
  buildContinuousBatchingTrace,
  type SchedulerTick,
} from '@/data/simulation';

// Build the full trace once (deterministic — same every time)
const TRACE = buildContinuousBatchingTrace(120);
const TRACE_LEN = TRACE.length;

interface SchedulerSimState {
  tickIndex: number;
  current: SchedulerTick;
  isRunning: boolean;
}

type SchedulerSimAction =
  | { type: 'TICK' }
  | { type: 'PAUSE' }
  | { type: 'RESUME' }
  | { type: 'RESET' };

function reducer(state: SchedulerSimState, action: SchedulerSimAction): SchedulerSimState {
  switch (action.type) {
    case 'TICK': {
      const nextIdx = (state.tickIndex + 1) % TRACE_LEN;
      return { ...state, tickIndex: nextIdx, current: TRACE[nextIdx] };
    }
    case 'PAUSE':
      return { ...state, isRunning: false };
    case 'RESUME':
      return { ...state, isRunning: true };
    case 'RESET':
      return { tickIndex: 0, current: TRACE[0], isRunning: true };
    default:
      return state;
  }
}

/**
 * Drives the live continuous batching scheduler visualization.
 * Returns the current tick frame plus control functions.
 *
 * Tick interval: 180ms — feels real-time without being jarring.
 */
export function useSchedulerSim(tickIntervalMs = 180) {
  const [state, dispatch] = useReducer(reducer, {
    tickIndex: 0,
    current: TRACE[0],
    isRunning: true,
  });

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Stable tick callback
  const tick = useCallback(() => {
    dispatch({ type: 'TICK' });
  }, []);

  useEffect(() => {
    if (state.isRunning) {
      intervalRef.current = setInterval(tick, tickIntervalMs);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [state.isRunning, tick, tickIntervalMs]);

  const pause = useCallback(() => dispatch({ type: 'PAUSE' }), []);
  const resume = useCallback(() => dispatch({ type: 'RESUME' }), []);
  const reset = useCallback(() => dispatch({ type: 'RESET' }), []);

  return {
    tick: state.current.tick,
    sequences: state.current.sequences,
    queueDepth: state.current.queueDepth,
    batchSize: state.current.batchSize,
    tokPerSec: state.current.tokPerSec,
    freeBlocks: state.current.freeBlocks,
    allocatedBlocks: state.current.allocatedBlocks,
    isRunning: state.isRunning,
    pause,
    resume,
    reset,
  };
}
