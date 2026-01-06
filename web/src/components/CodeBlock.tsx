'use client';

import { useState, useEffect } from 'react';
import hljs from 'highlight.js/lib/core';
import typescript from 'highlight.js/lib/languages/typescript';
import python from 'highlight.js/lib/languages/python';
import go from 'highlight.js/lib/languages/go';
import java from 'highlight.js/lib/languages/java';
import ruby from 'highlight.js/lib/languages/ruby';
import bash from 'highlight.js/lib/languages/bash';
import 'highlight.js/styles/atom-one-dark.css';

// Register languages
hljs.registerLanguage('typescript', typescript);
hljs.registerLanguage('python', python);
hljs.registerLanguage('go', go);
hljs.registerLanguage('java', java);
hljs.registerLanguage('ruby', ruby);
hljs.registerLanguage('bash', bash);
hljs.registerLanguage('curl', bash);

type Language = 'curl' | 'python' | 'typescript' | 'go' | 'java' | 'ruby';

interface CodeBlockProps {
  code: Record<string, string>;
  defaultLanguage?: Language;
}

export function CodeBlock({
  code,
  defaultLanguage = 'typescript',
}: CodeBlockProps) {
  const [activeLanguage, setActiveLanguage] =
    useState<Language>(defaultLanguage);
  const [highlightedCode, setHighlightedCode] = useState('');

  const currentCode = code[activeLanguage] || '';

  useEffect(() => {
    const result = hljs.highlight(currentCode, {
      language: activeLanguage,
    });
    setHighlightedCode(result.value);
  }, [currentCode, activeLanguage]);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code[activeLanguage]);
  };

  const availableLanguages = Object.keys(code);

  return (
    <div className='bg-[#002635] rounded-lg border border-[#3e4451] overflow-hidden my-6'>
      <div className='flex items-center justify-between bg-[#0d1f2d]'>
        <div className='flex'>
          {availableLanguages.map((lang) => (
            <button
              key={lang}
              onClick={() => setActiveLanguage(lang as Language)}
              className={`px-5 py-3 text-sm transition-colors border-b-2 ${
                activeLanguage === lang
                  ? 'text-white border-slate-400'
                  : 'text-slate-400 border-transparent hover:text-slate-300'
              }`}
            >
              {lang}
            </button>
          ))}
        </div>
        <button
          onClick={copyToClipboard}
          className='p-2 mr-2 text-[#6b7280] hover:text-white transition-colors'
          title='Copy code'
        >
          <svg
            width='16'
            height='16'
            viewBox='0 0 24 24'
            fill='none'
            stroke='currentColor'
            strokeWidth='2'
          >
            <rect x='9' y='9' width='13' height='13' rx='2' ry='2' />
            <path d='M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1' />
          </svg>
        </button>
      </div>
      <pre className='p-4 overflow-x-auto text-sm font-mono leading-relaxed m-0 bg-[#002635]!'>
        <code
          className='hljs'
          dangerouslySetInnerHTML={{ __html: highlightedCode }}
          style={{
            background: 'none',
          }}
        />
      </pre>
    </div>
  );
}
