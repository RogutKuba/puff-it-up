import Link from 'next/link';

type TocItem = {
  label: string;
  href: string;
};

interface TableOfContentsProps {
  items: TocItem[];
}

export function TableOfContents({ items }: TableOfContentsProps) {
  return (
    <div className='mt-6 hidden lg:sticky lg:top-28 lg:block lg:self-start'>
      <nav className='toc text-xs'>
        <p className='mb-2 font-bold '>On this page</p>
        <ul className='space-y-2'>
          {items.map((item) => (
            <li key={item.href} className='pl-0'>
              <Link href={item.href} className='hover:underline text-slate-400'>
                {item.label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
}
