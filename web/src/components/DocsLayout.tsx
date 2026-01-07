import { Sidebar } from './Sidebar';
import { TableOfContents } from './TableOfContents';

interface DocsLayoutProps {
  children: React.ReactNode;
  tocItems?: { label: string; href: string }[];
  title?: string;
}

export function DocsLayout({
  children,
  tocItems = [],
  title,
}: DocsLayoutProps) {
  return (
    <div className=''>
      <div className='container mx-auto px-4 md:px-6 relative flex flex-col items-start md:flex-row md:gap-6 relative'>
        <Sidebar />
        <div className='relative flex w-full flex-col pb-12 py-4 md:w-[calc(100%-14rem-1.5rem)]'>
          <div id='docs-container' className='min-h-[800px] max-w-full'>
            <div
              className='relative lg:grid lg:grid-cols-[1fr_13.5rem] lg:gap-12'
              id='container'
            >
              <div className='min-w-0 max-w-4xl'>
                {title && (
                  <h1 className='group/heading relative mb-2 mt-6 flex w-full scroll-m-20 items-center text-2xl font-bold tabular-nums leading-tight '>
                    {title}
                  </h1>
                )}
                <div className='max-w-full break-words prose'>{children}</div>
              </div>
              {tocItems.length > 0 && <TableOfContents items={tocItems} />}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
