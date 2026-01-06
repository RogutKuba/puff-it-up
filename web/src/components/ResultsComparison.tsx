'use client';

import { useState } from 'react';

interface SearchResult {
  title: string;
  snippet: string;
}

interface QueryResult {
  query: string;
  results: SearchResult[];
}

interface ResultsComparisonProps {
  dense: QueryResult;
  late: QueryResult;
}

function Tooltip({
  content,
  children,
  position = 'below',
}: {
  content: string;
  children: React.ReactNode;
  position?: 'above' | 'below';
}) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div
      className='relative'
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      {isVisible && (
        <div
          className={`absolute z-50 left-0 right-0 p-3 bg-gray-900 text-white text-xs rounded-lg shadow-lg max-h-48 overflow-y-auto ${
            position === 'above' ? 'bottom-full mb-1' : 'top-full mt-0.5'
          }`}
        >
          {content}
        </div>
      )}
    </div>
  );
}

function ResultCell({
  result,
  isLast,
  index,
}: {
  result: SearchResult;
  isLast: boolean;
  index: number;
}) {
  return (
    <div className='px-3 py-3 h-full flex flex-col justify-between'>
      <div className='text-sm text-gray-900 font-medium'>
        <span className='text-gray-400 font-normal'>#{index}</span> &ldquo;
        {result.title}&rdquo;
      </div>
      <Tooltip content={result.snippet} position={isLast ? 'above' : 'below'}>
        <div className='text-[13px] text-gray-600 line-clamp-2 cursor-help border-b border-dashed border-gray-300 pb-0.5 hover:text-gray-900 hover:border-gray-400 transition-colors mt-2'>
          {result.snippet}
        </div>
      </Tooltip>
    </div>
  );
}

export function ResultsComparison({ dense, late }: ResultsComparisonProps) {
  const rowCount = Math.max(dense.results.length, late.results.length);

  return (
    <div className='bg-white rounded-lg border border-gray-200 my-6'>
      <div className='px-4 py-3 border-b border-gray-200 bg-gray-50 rounded-t-lg'>
        <div className='text-xs text-gray-500 mb-1'>Query</div>
        <div className='text-sm text-gray-900 font-medium'>
          &ldquo;{dense.query}&rdquo;
        </div>
      </div>
      <table className='w-full table-fixed rounded-b-lg'>
        <thead>
          <tr className='border-b border-gray-200'>
            <th
              className='w-1/2 text-left text-xs font-semibold px-3 py-2'
              style={{ color: '#ea580c' }}
            >
              Dense Embeddings
            </th>
            <th
              className='w-1/2 text-left text-xs font-semibold px-3 py-2 border-l border-gray-200'
              style={{ color: '#0891b2' }}
            >
              Late Interaction
            </th>
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rowCount }).map((_, i) => (
            <tr
              key={i}
              className={i < rowCount - 1 ? 'border-b border-gray-200' : ''}
            >
              <td className='align-top'>
                {dense.results[i] && (
                  <ResultCell
                    result={dense.results[i]}
                    isLast={i === rowCount - 1}
                    index={i + 1}
                  />
                )}
              </td>
              <td className='align-top border-l border-gray-200'>
                {late.results[i] && (
                  <ResultCell
                    result={late.results[i]}
                    isLast={i === rowCount - 1}
                    index={i + 1}
                  />
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
