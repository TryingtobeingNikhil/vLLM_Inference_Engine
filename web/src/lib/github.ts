export interface RepoStats {
  stars: number;
  forks: number;
  watchers: number;
}

export const REPO = 'TryingtobeingNikhil/vLLM_Inference_Engine';

/**
 * Server-side fetch, cached and revalidated hourly (ISR). One request per hour
 * for the whole site instead of one per visitor, so GitHub's unauthenticated
 * rate limit (60/hr per IP) never leaves visitors staring at a stale fallback.
 */
export async function getRepoStats(): Promise<RepoStats | null> {
  try {
    const res = await fetch(`https://api.github.com/repos/${REPO}`, {
      headers: { Accept: 'application/vnd.github+json' },
      next: { revalidate: 3600 },
    });
    if (!res.ok) return null;
    const data = await res.json();
    if (data.stargazers_count === undefined) return null;
    return { stars: data.stargazers_count, forks: data.forks_count, watchers: data.watchers_count };
  } catch {
    return null;
  }
}
