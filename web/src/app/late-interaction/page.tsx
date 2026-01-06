import { DocsLayout } from "@/components";
import Article, { tocItems } from "@/content/article.mdx";

export default function Home() {
  return (
    <DocsLayout tocItems={tocItems}>
      <article>
        <Article />
      </article>
    </DocsLayout>
  );
}
