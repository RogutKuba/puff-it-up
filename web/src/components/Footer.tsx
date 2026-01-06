import Link from 'next/link';

const footerLinks = {
  company: [
    { label: 'About', href: '/about' },
    { label: 'Jobs', href: '/jobs' },
    { label: 'Customers', href: '/customers' },
    { label: 'Blog', href: '/blog' },
  ],
  product: [
    { label: 'Docs', href: '/docs' },
    { label: 'Pricing', href: '/pricing' },
    { label: 'Status', href: 'https://status.turbopuffer.com', external: true },
    { label: 'SOC 2', href: '/security' },
  ],
  support: [
    { label: 'Contact', href: '/contact' },
    {
      label: 'Discord',
      href: 'https://discord.gg/turbopuffer',
      external: true,
    },
    {
      label: 'Twitter',
      href: 'https://twitter.com/turbopuffer',
      external: true,
    },
    { label: 'GitHub', href: 'https://github.com/turbopuffer', external: true },
  ],
  legal: [
    { label: 'Privacy', href: '/privacy' },
    { label: 'Terms', href: '/terms' },
  ],
};

export function Footer() {
  return (
    <footer className='mt-auto border-t border-white/10 bg-[#0f172a] py-12'>
      <div className='container mx-auto px-4 md:px-6'>
        <div className='grid grid-cols-2 gap-8 md:grid-cols-4 lg:grid-cols-5'>
          {/* Brand */}
          <div className='col-span-2 md:col-span-4 lg:col-span-1'>
            <Link href='/' className='flex items-center gap-2'>
              <svg
                width='26'
                height='16'
                viewBox='0 0 26 16'
                fill='none'
                xmlns='http://www.w3.org/2000/svg'
                className='text-orange-500'
              >
                <path d='M13 0L0 8L13 16L26 8L13 0Z' fill='currentColor' />
              </svg>
              <span className='font-medium '>turbopuffer</span>
            </Link>
            <p className='mt-4 text-sm text-slate-400'>
              The search engine for the AI era.
            </p>
          </div>

          {/* Company */}
          <div>
            <h3 className='mb-3 text-sm font-semibold '>Company</h3>
            <ul className='space-y-2'>
              {footerLinks.company.map((link) => (
                <li key={link.label}>
                  <Link
                    href={link.href}
                    className='text-sm text-slate-400 hover: hover:underline'
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Product */}
          <div>
            <h3 className='mb-3 text-sm font-semibold '>Product</h3>
            <ul className='space-y-2'>
              {footerLinks.product.map((link) => (
                <li key={link.label}>
                  {link.external ? (
                    <a
                      href={link.href}
                      target='_blank'
                      rel='noopener noreferrer'
                      className='text-sm text-slate-400 hover: hover:underline'
                    >
                      {link.label}
                    </a>
                  ) : (
                    <Link
                      href={link.href}
                      className='text-sm text-slate-400 hover: hover:underline'
                    >
                      {link.label}
                    </Link>
                  )}
                </li>
              ))}
            </ul>
          </div>

          {/* Support */}
          <div>
            <h3 className='mb-3 text-sm font-semibold '>Support</h3>
            <ul className='space-y-2'>
              {footerLinks.support.map((link) => (
                <li key={link.label}>
                  {link.external ? (
                    <a
                      href={link.href}
                      target='_blank'
                      rel='noopener noreferrer'
                      className='text-sm text-slate-400 hover: hover:underline'
                    >
                      {link.label}
                    </a>
                  ) : (
                    <Link
                      href={link.href}
                      className='text-sm text-slate-400 hover: hover:underline'
                    >
                      {link.label}
                    </Link>
                  )}
                </li>
              ))}
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h3 className='mb-3 text-sm font-semibold '>Legal</h3>
            <ul className='space-y-2'>
              {footerLinks.legal.map((link) => (
                <li key={link.label}>
                  <Link
                    href={link.href}
                    className='text-sm text-slate-400 hover: hover:underline'
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className='mt-12 flex flex-col items-center justify-between gap-4 border-t border-white/10 pt-8 md:flex-row'>
          <p className='text-sm text-slate-400'>
            &copy; {new Date().getFullYear()} turbopuffer Inc. All rights
            reserved.
          </p>
          <div className='flex gap-4'>
            {/* Twitter */}
            <a
              href='https://twitter.com/turbopuffer'
              target='_blank'
              rel='noopener noreferrer'
              className='text-slate-400 hover:'
            >
              <svg
                xmlns='http://www.w3.org/2000/svg'
                width='20'
                height='20'
                viewBox='0 0 24 24'
                fill='currentColor'
              >
                <path d='M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z' />
              </svg>
            </a>
            {/* Discord */}
            <a
              href='https://discord.gg/turbopuffer'
              target='_blank'
              rel='noopener noreferrer'
              className='text-slate-400 hover:'
            >
              <svg
                xmlns='http://www.w3.org/2000/svg'
                width='20'
                height='20'
                viewBox='0 0 24 24'
                fill='currentColor'
              >
                <path d='M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z' />
              </svg>
            </a>
            {/* GitHub */}
            <a
              href='https://github.com/turbopuffer'
              target='_blank'
              rel='noopener noreferrer'
              className='text-slate-400 hover:'
            >
              <svg
                xmlns='http://www.w3.org/2000/svg'
                width='20'
                height='20'
                viewBox='0 0 24 24'
                fill='currentColor'
              >
                <path d='M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z' />
              </svg>
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
