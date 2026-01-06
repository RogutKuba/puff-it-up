import type { MDXComponents } from 'mdx/types';
import {
  CodeBlock,
  ResultsComparison,
  PerformanceComparison,
} from '@/components';

const idify = (str: string) => str.toLowerCase().replace(/ /g, '-');

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    CodeBlock,
    ResultsComparison,
    PerformanceComparison,
    h1: ({ children }) => (
      <h1 className='text-3xl font-extrabold tracking-tight mb-6 font-serif'>
        {children}
      </h1>
    ),
    h2: ({ children }) => (
      <h2
        id={idify(children)}
        className='text-xl font-bold mt-10 mb-4 font-serif'
      >
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 id={idify(children)} className='text-lg font-semibold  mt-8 mb-3'>
        {children}
      </h3>
    ),
    p: ({ children }) => (
      <p className='leading-relaxed mb-4 text-[15px]'>{children}</p>
    ),
    a: ({ children, href }) => (
      <a
        href={href}
        target='_blank'
        className='underline underline-offset-2 decoration-slate-500 hover: hover:decoration-white transition-colors'
      >
        {children}
      </a>
    ),
    ul: ({ children }) => (
      <ul className='list-disc list-inside mb-4 space-y-1 text-[15px]'>
        {children}
      </ul>
    ),
    ol: ({ children }) => (
      <ol className='list-decimal list-inside mb-4 space-y-1 text-[15px]'>
        {children}
      </ol>
    ),
    li: ({ children }) => <li>{children}</li>,
    blockquote: ({ children }) => (
      <blockquote className='border-l-4 border-orange-500 pl-4 italic my-6'>
        {children}
      </blockquote>
    ),
    code: ({ children }) => (
      <code className='px-1.5 py-0.5 rounded text-sm [pre_&]:!bg-transparent [pre_&]:!text-white [pre_&]:!p-0'>
        {children}
      </code>
    ),
    pre: ({ children }) => (
      <pre className='border border-slate-800 rounded-lg p-4 overflow-x-auto mb-6'>
        {children}
      </pre>
    ),
    hr: () => <hr className='my-8' />,
    strong: ({ children }) => (
      <strong className='font-semibold '>{children}</strong>
    ),
    em: ({ children }) => <em className='italic'>{children}</em>,
    table: ({ children }) => (
      <div className='overflow-x-auto mb-6'>
        <table className='w-full border-collapse text-sm'>{children}</table>
      </div>
    ),
    thead: ({ children }) => (
      <thead className='border-b border-slate-700'>{children}</thead>
    ),
    tbody: ({ children }) => <tbody>{children}</tbody>,
    tr: ({ children }) => (
      <tr className='border-b border-slate-800'>{children}</tr>
    ),
    th: ({ children }) => (
      <th className='text-left py-3 px-4 font-semibold'>{children}</th>
    ),
    td: ({ children }) => <td className='py-3 px-4'>{children}</td>,
    ...components,
  };
}
