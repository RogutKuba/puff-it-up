'use client';

import { usePathname } from 'next/navigation';
import { useCallback, useRef, useState } from 'react';

type SidebarItem = {
  label: string;
  href: string;
  local?: boolean;
};

type SidebarSection = {
  title?: string;
  items: SidebarItem[];
};

const sections: SidebarSection[] = [
  {
    items: [
      { label: 'Introduction', href: '/docs' },
      { label: 'Architecture', href: '/docs/architecture' },
      { label: 'Guarantees', href: '/docs/guarantees' },
      { label: 'Tradeoffs', href: '/docs/tradeoffs' },
      { label: 'Limits', href: '/docs/limits' },
      { label: 'Regions', href: '/docs/regions' },
      { label: 'Roadmap & Changelog', href: '/docs/roadmap' },
      { label: 'Security', href: '/docs/security' },
      { label: 'Encryption', href: '/docs/cmek' },
      { label: 'Backups', href: '/docs/backups' },
      { label: 'Private Networking', href: '/docs/private-networking' },
      { label: 'Performance', href: '/docs/performance' },
    ],
  },
  {
    title: 'Guides',
    items: [
      { label: 'Quickstart', href: '/docs/quickstart' },
      { label: 'Vector Search', href: '/docs/vector' },
      { label: 'Full-Text Search', href: '/docs/fts' },
      { label: 'Hybrid Search', href: '/docs/hybrid' },
      {
        label: 'Late Interaction',
        href: '/late-interaction',
        local: true,
      },
      { label: 'Late Interaction Roadmap', href: '/roadmap', local: true },
      { label: 'Testing', href: '/docs/testing' },
    ],
  },
  {
    title: 'API',
    items: [
      { label: 'Auth & Encoding', href: '/docs/auth' },
      { label: 'Write', href: '/docs/write' },
      { label: 'Query', href: '/docs/query' },
      { label: 'Namespace metadata', href: '/docs/metadata' },
      { label: 'Export', href: '/docs/export' },
      { label: 'Warm cache', href: '/docs/warm-cache' },
      { label: 'List namespaces', href: '/docs/namespaces' },
      { label: 'Delete namespace', href: '/docs/delete-namespace' },
      { label: 'Recall', href: '/docs/recall' },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();
  const [isAtBottom, setIsAtBottom] = useState(false);
  const scrollRef = useRef<HTMLElement>(null);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 10;
    setIsAtBottom(atBottom);
  }, []);

  return (
    <aside
      ref={scrollRef}
      onScroll={handleScroll}
      className='z-30 shrink-0 overflow-y-auto md:sticky md:top-[4.5rem] md:max-h-[calc(100vh-4.5rem)] md:w-[14rem] md:pt-6 hidden md:block'
    >
      <div className='relative'>
        <div className='pointer-events-none fixed top-[4.5rem] flex h-14 w-[14rem] items-start justify-center bg-gradient-to-b from-[#0f172a] to-transparent transition-opacity duration-200 opacity-0' />

        {sections.map((section, idx) => (
          <div key={idx} className='pb-4'>
            {section.title && (
              <p className='relative z-10 pb-2 text-sm font-semibold '>
                {section.title}
              </p>
            )}
            <div className='grid grid-flow-row auto-rows-max text-xs'>
              {section.items.map((item) => (
                <div key={item.label} className='flex flex-row items-center'>
                  {pathname === item.href && (
                    <svg
                      xmlns='http://www.w3.org/2000/svg'
                      width='12'
                      height='12'
                      viewBox='0 0 24 24'
                      fill='none'
                      stroke='currentColor'
                      strokeWidth='2'
                      strokeLinecap='round'
                      strokeLinejoin='round'
                      className='shrink-0 stroke-orange-600'
                    >
                      <path d='m9 18 6-6-6-6' />
                    </svg>
                  )}
                  <a
                    target={item.local ? '_self' : '_blank'}
                    href={
                      item.local
                        ? item.href
                        : `https://turbopuffer.com${item.href}`
                    }
                    className={`flex w-full flex-row items-start rounded-md border border-transparent py-1 hover:underline ${
                      pathname === item.href
                        ? 'font-semibold text-orange-600'
                        : 'text-slate-600'
                    }`}
                  >
                    <span>{item.label}</span>
                  </a>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
