import { Nav }                  from '@/components/Nav';
import { RepoStatsProvider }    from '@/components/GitHubStats';
import { getRepoStats }         from '@/lib/github';
import { HeroSection }          from '@/components/HeroSection';
import { SchedulerLiveSection }  from '@/components/SchedulerLiveSection';
import { Phase1vs2Section }      from '@/components/Phase1vs2Section';
import { KVCacheSection }        from '@/components/KVCacheSection';
import { CPUSwapSection }        from '@/components/CPUSwapSection';
import { PhaseBuildLogSection }  from '@/components/PhaseBuildLogSection';
import { BenchmarksSection }     from '@/components/BenchmarksSection';
import { APICodeSection }        from '@/components/APICodeSection';
import { GitHubFooter }          from '@/components/GitHubFooter';
import { V3FeaturesSection }    from '@/components/V3FeaturesSection';
import { TestedSection }        from '@/components/TestedSection';

// Re-fetch GitHub stats at most once an hour (ISR).
export const revalidate = 3600;

export default async function HomePage() {
  const repoStats = await getRepoStats();

  return (
    <RepoStatsProvider initial={repoStats}>
      <main>
        <Nav />

        {/* § 0 — Hero + live scheduler panel */}
        <HeroSection />

        {/* § 1 — Scheduler step: one packed forward pass */}
        <SchedulerLiveSection />

        {/* § 2 — Sequential vs batched (v2 legacy replay + v3 smoke numbers) */}
        <Phase1vs2Section />

        {/* § 3 — Paged KV cache visualisation */}
        <KVCacheSection />

        {/* § 4 — Preemption under memory pressure (swap / recompute) */}
        <CPUSwapSection />

        {/* § 5 — New in v3: prefix caching, speculation, streaming, metrics */}
        <V3FeaturesSection />

        {/* § 6 — 12-phase build log timeline */}
        <PhaseBuildLogSection />

        {/* § 7 — Where it's been tested + Colab */}
        <TestedSection />

        {/* § 8 — Measured results (data/gpuBenchmarks.ts) + v2 legacy */}
        <BenchmarksSection />

        {/* § 9 — HTTP API usage examples */}
        <APICodeSection />

        {/* § 10 — Footer */}
        <GitHubFooter />
      </main>
    </RepoStatsProvider>
  );
}
