import type { Metadata } from 'next';
import { JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { Banner, Footer, Header } from '@/components';

const jetbrainsMono = JetBrains_Mono({
  variable: '--font-jetbrains-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'Late Interaction',
  description: 'Documentation for turbopuffer',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang='en'>
      <body
        className={`${jetbrainsMono.className} antialiased bg-[#0f172a] min-h-screen flex flex-col overflow-x-hidden`}
      >
        <Banner />
        <Header />
        {children}
        <Footer />
      </body>
    </html>
  );
}
