'use client';

import { useEffect, useState } from 'react';

interface RepoStats {
  stars: number;
  forks: number;
  watchers: number;
}

const REPO = 'TryingtobeingNikhil/vLLM_Inference_Engine';

export function GitHubStats({ className = '' }: { className?: string }) {
  const [stats, setStats] = useState<RepoStats>({ stars: 56, forks: 6, watchers: 56 });
  const [live, setLive] = useState(false);

  useEffect(() => {
    fetch(`https://api.github.com/repos/${REPO}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.stargazers_count !== undefined) {
          setStats({
            stars: data.stargazers_count,
            forks: data.forks_count,
            watchers: data.watchers_count,
          });
          setLive(true);
        }
      })
      .catch(() => {
        // silently fall back to default values
      });
  }, []);

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      {/* Stars */}
      <a
        href={`https://github.com/${REPO}/stargazers`}
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-1.5 border border-[#2a2a2a] px-3 py-1.5 font-mono text-xs text-[#e8e8e8] transition-colors hover:border-[#fbbf24] hover:text-[#fbbf24] group"
        title="GitHub Stars"
      >
        {/* Star icon */}
        <svg
          viewBox="0 0 16 16"
          fill="currentColor"
          className="h-3.5 w-3.5 text-[#fbbf24]"
          aria-hidden="true"
        >
          <path d="M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0 1 .416 1.279l-3.046 2.97.719 4.192a.751.751 0 0 1-1.088.791L8 12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.873 6.374a.75.75 0 0 1 .416-1.28l4.21-.611L7.327.668A.75.75 0 0 1 8 .25Z" />
        </svg>
        <span>{stats.stars}</span>
        <span className="text-[#555555]">stars</span>
      </a>

      {/* Forks */}
      <a
        href={`https://github.com/${REPO}/forks`}
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-1.5 border border-[#2a2a2a] px-3 py-1.5 font-mono text-xs text-[#888888] transition-colors hover:border-[#4ade80] hover:text-[#4ade80]"
        title="GitHub Forks"
      >
        {/* Fork icon */}
        <svg
          viewBox="0 0 16 16"
          fill="currentColor"
          className="h-3.5 w-3.5 text-[#4ade80]"
          aria-hidden="true"
        >
          <path d="M5 5.372v.878c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-.878a2.25 2.25 0 1 1 1.5 0v.878a2.25 2.25 0 0 1-2.25 2.25h-1.5v2.128a2.251 2.251 0 1 1-1.5 0V8.5h-1.5A2.25 2.25 0 0 1 3.5 6.25v-.878a2.25 2.25 0 1 1 1.5 0ZM5 3.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Zm6.75.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm-3 8.75a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Z" />
        </svg>
        <span>{stats.forks}</span>
        <span className="text-[#555555]">forks</span>
      </a>

      {/* Live indicator */}
      <span className="flex items-center gap-1.5 font-mono text-[10px] text-[#333333]">
        <span
          className={`inline-block h-1.5 w-1.5 rounded-full ${
            live ? 'bg-[#4ade80] animate-pulse' : 'bg-[#333333]'
          }`}
        />
        {live ? 'live' : 'cached'}
      </span>
    </div>
  );
}
