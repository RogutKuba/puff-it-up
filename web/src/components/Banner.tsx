'use client';

export function Banner() {
  return (
    <div className='dark relative z-50 w-full bg-gradient-to-r from-red-700 to-orange-300 py-2 text-xs text-white md:text-sm'>
      <div className='container mx-auto px-4 md:px-6'>
        <a
          className='group flex w-fit items-start gap-2 md:items-center'
          target='_blank'
          href='https://tpuf.link/comms'
        >
          <style jsx>{`
            @keyframes play-coin {
              0% {
                background-position: 0px;
              }
              100% {
                background-position: -96px;
              }
            }
          `}</style>
          <div
            className='size-4 shrink-0'
            style={{
              backgroundImage:
                "url('https://turbopuffer.com/images/coin-sprite.png')",
              backgroundSize: '96px 16px',
              animation: 'play-coin 0.8s steps(6, end) infinite',
            }}
          />
          <span className='text-pretty md:hidden'>
            We&apos;ve doubled down with Lachy Groom, added Thrive
          </span>
          <span className='hidden text-pretty md:block'>
            We&apos;ve doubled down with Lachy Groom and added Thrive to the
            team
          </span>
          <svg
            width='24'
            height='24'
            fill='none'
            xmlns='http://www.w3.org/2000/svg'
            viewBox='0 0 24 24'
            className='size-4 shrink-0 group-hover:animate-bounce-horizontal'
          >
            <path
              d='M4 11v2h12v2h2v-2h2v-2h-2V9h-2v2H4zm10-4h2v2h-2V7zm0 0h-2V5h2v2zm0 10h2v-2h-2v2zm0 0h-2v2h2v-2z'
              fill='currentColor'
            />
          </svg>
        </a>
      </div>
    </div>
  );
}
