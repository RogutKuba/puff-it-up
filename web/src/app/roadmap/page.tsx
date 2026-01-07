import { DocsLayout } from '@/components';
import Roadmap from '@/content/roadmap.mdx';

export default function RoadmapPage() {
  return (
    <DocsLayout>
      <article>
        <Roadmap />
      </article>
    </DocsLayout>
  );
}
