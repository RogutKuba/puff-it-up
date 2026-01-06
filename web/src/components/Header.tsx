import Image from 'next/image';
import Link from 'next/link';

const navItems = [
  { label: 'Customers', href: '/customers' },
  { label: 'Pricing', href: '/pricing' },
  { label: 'Company', href: '/about', className: 'hidden lg:inline-block' },
  { label: 'Jobs', href: '/jobs' },
  { label: 'Blog', href: '/blog' },
  { label: 'Docs', href: '/docs', active: true },
  { label: 'Contact', href: '/contact' },
  { label: 'Dashboard', href: '/dashboard' },
];

export function Header() {
  return (
    <nav className='dark sticky left-0 right-0 top-0 z-50 bg-background text-white'>
      <div className='container mx-auto flex items-center py-4 px-4 md:px-6 text-sm'>
        <div className='flex items-center gap-2'>
          <button
            className='grid relative group items-center justify-center border rounded-sm whitespace-nowrap text-sm font-medium h-6 w-6 mr-2 hover:bg-transparent md:hidden border-transparent'
            type='button'
          >
            <svg
              xmlns='http://www.w3.org/2000/svg'
              width='24'
              height='24'
              viewBox='0 0 24 24'
              fill='none'
              stroke='currentColor'
              strokeWidth='2'
              strokeLinecap='round'
              strokeLinejoin='round'
              className=''
            >
              <line x1='4' x2='20' y1='12' y2='12' />
              <line x1='4' x2='20' y1='6' y2='6' />
              <line x1='4' x2='20' y1='18' y2='18' />
            </svg>
          </button>
          <a href='https://turbopuffer.com' className='flex flex-row gap-2'>
            <Image alt='Logo' width={26} height={16} src='/turbopuffer.svg' />
            <span className='text-sm font-medium md:text-base'>
              turbopuffer
            </span>
          </a>
        </div>

        <div className='flex-1' />

        <nav className='hidden flex-row items-center md:flex md:gap-2'>
          {navItems.map((item) => (
            <a
              key={item.label}
              href={`https://turbopuffer.com${item.href}`}
              className={`flex items-center border-b-2 px-1 py-1.5  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 focus-visible:ring-offset-0 lg:px-3 ${
                item.active
                  ? 'border-white'
                  : 'border-transparent hover:border-white/50'
              } ${item.className || ''}`}
            >
              {item.label}
            </a>
          ))}
          <a
            href='https://turbopuffer.com/join'
            className='relative group items-center justify-center border rounded-sm whitespace-nowrap text-sm font-medium bg-orange-300 text-orange-950 border-orange-950 shadow-[2px_2px_0_0_#431407] hover:shadow-[3px_3px_0_0_#431407] transition-all hover:bg-orange-400 active:bg-orange-400 active:shadow-none px-3 py-1.5 h-10 ml-2 hidden md:flex'
          >
            Sign up
          </a>
        </nav>
      </div>
    </nav>
  );
}
