'use client';

import { useEffect, useRef, useCallback } from 'react';

interface UseAnimationLoopOptions {
  intervalMs: number;
  onTick: () => void;
  autoStart?: boolean;
}

/**
 * Generic animation loop hook using setInterval.
 * Returns controls to start/stop the loop.
 */
export function useAnimationLoop({ intervalMs, onTick, autoStart = true }: UseAnimationLoopOptions) {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const onTickRef = useRef(onTick);

  // Keep onTick ref current without re-creating the interval
  useEffect(() => {
    onTickRef.current = onTick;
  }, [onTick]);

  const start = useCallback(() => {
    if (intervalRef.current !== null) return;
    intervalRef.current = setInterval(() => {
      onTickRef.current();
    }, intervalMs);
  }, [intervalMs]);

  const stop = useCallback(() => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    stop();
    start();
  }, [stop, start]);

  useEffect(() => {
    if (autoStart) start();
    return () => stop();
  }, [autoStart, start, stop]);

  return { start, stop, reset };
}
