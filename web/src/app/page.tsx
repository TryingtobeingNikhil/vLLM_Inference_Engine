import { HeroSection }          from '@/components/HeroSection';
import { SchedulerLiveSection }  from '@/components/SchedulerLiveSection';
import { Phase1vs2Section }      from '@/components/Phase1vs2Section';
import { KVCacheSection }        from '@/components/KVCacheSection';
import { CPUSwapSection }        from '@/components/CPUSwapSection';
import { PhaseBuildLogSection }  from '@/components/PhaseBuildLogSection';
import { BenchmarksSection }     from '@/components/BenchmarksSection';
import { APICodeSection }        from '@/components/APICodeSection';
import { GitHubFooter }          from '@/components/GitHubFooter';

export default function HomePage() {
  return (
    <main>
      {/* § 0 — Hero + live scheduler panel */}
      <HeroSection />

      {/* § 1 — Scheduler loop architecture */}
      <SchedulerLiveSection />

      {/* § 2 — Phase 1 (sequential) vs Phase 2–9 (continuous batching) */}
      <Phase1vs2Section />

      {/* § 3 — Paged KV cache visualisation */}
      <KVCacheSection />

      {/* § 4 — CPU swap pool under memory pressure */}
      <CPUSwapSection />

      {/* § 5 — 11-phase build log timeline */}
      <PhaseBuildLogSection />

      {/* § 6 — Measured benchmark numbers */}
      <BenchmarksSection />

      {/* § 7 — HTTP API usage examples */}
      <APICodeSection />

      {/* § 8 — Footer */}
      <GitHubFooter />
    </main>
  );
}
