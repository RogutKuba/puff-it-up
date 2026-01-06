'use client';

interface MetricRowProps {
  label: string;
  dense: string;
  late: string;
  highlight?: 'dense' | 'late';
}

function MetricRow({ label, dense, late, highlight }: MetricRowProps) {
  return (
    <tr className='border-b border-gray-100 last:border-b-0'>
      <td className='py-3 pr-4 text-sm text-gray-600'>{label}</td>
      <td className={`py-3 px-4 text-sm font-medium text-right`}>{dense}</td>
      <td className={`py-3 px-4 text-sm font-medium text-right`}>{late}</td>
    </tr>
  );
}

function BarComparison({
  label,
  denseValue,
  lateValue,
  denseLabel,
  lateLabel,
}: {
  label: string;
  denseValue: number;
  lateValue: number;
  denseLabel: string;
  lateLabel: string;
}) {
  const maxValue = Math.max(denseValue, lateValue);
  const denseWidth = (denseValue / maxValue) * 100;
  const lateWidth = (lateValue / maxValue) * 100;

  return (
    <div className='mb-6 last:mb-0'>
      <div className='text-sm font-medium text-gray-700 mb-2'>{label}</div>
      <div className='space-y-2'>
        <div className='flex items-center gap-3'>
          <div className='w-24 text-xs text-gray-500'>Dense</div>
          <div className='flex-1 h-6 bg-gray-100 rounded overflow-hidden'>
            <div
              className='h-full bg-[#0f172a] rounded flex items-center justify-end pr-2'
              style={{ width: `${denseWidth}%` }}
            >
              <span className='text-xs text-white font-medium'>
                {denseLabel}
              </span>
            </div>
          </div>
        </div>
        <div className='flex items-center gap-3'>
          <div className='w-24 text-xs text-gray-500'>Late Interaction</div>
          <div className='flex-1 h-6 bg-gray-100 rounded overflow-hidden'>
            <div
              className='h-full bg-orange-500 rounded flex items-center justify-end pr-2'
              style={{ width: `${lateWidth}%` }}
            >
              <span className='text-xs text-white font-medium'>
                {lateLabel}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function PerformanceComparison() {
  return (
    <div className='bg-white rounded-lg border border-gray-200 overflow-hidden my-6'>
      <div className='px-4 py-3 border-b border-gray-200 bg-gray-50'>
        <div className='text-sm text-gray-900 font-medium'>
          Performance &amp; Cost Comparison
        </div>
        <div className='text-xs text-gray-500'>
          15,000 documents, 100 queries. Dense uses text-embedding-3-small, late
          interaction uses Jina ColBERT v2. Latency includes embedding time.
        </div>
      </div>

      <div className='p-4'>
        <table className='w-full'>
          <thead>
            <tr className='border-b border-gray-200'>
              <th className='pb-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide'>
                Metric
              </th>
              <th
                className='pb-2 text-right text-xs font-semibold uppercase tracking-wide px-4'
                style={{ color: '#ea580c' }}
              >
                Dense
              </th>
              <th
                className='pb-2 text-right text-xs font-semibold uppercase tracking-wide px-4'
                style={{ color: '#0891b2' }}
              >
                Late
              </th>
            </tr>
          </thead>
          <tbody>
            <MetricRow label='Total vectors' dense='15,000' late='1,105,809' />
            <MetricRow
              label='Total storage'
              dense='92.36 MB'
              late='597.73 MB'
              highlight='dense'
            />
            <MetricRow label='Avg. queries per search' dense='1' late='64' />
            <MetricRow
              label='Total data queried'
              dense='25.6 GB'
              late='3.83 TB'
              highlight='dense'
            />
            <MetricRow
              label='P50 latency'
              dense='70 ms'
              late='971 ms'
              highlight='dense'
            />
            <MetricRow
              label='P90 latency'
              dense='181 ms'
              late='1531 ms'
              highlight='dense'
            />
            <MetricRow
              label='P99 latency'
              dense='310 ms'
              late='2330 ms'
              highlight='dense'
            />
          </tbody>
        </table>
      </div>

      {/* Cost estimate */}
      <div className='p-4 bg-gray-50 border-t border-gray-200'>
        <div className='text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1'>
          Extrapolated Monthly Cost
        </div>
        <div className='text-xs text-gray-500 mb-3'>
          Scaled from benchmark: 100k documents, 1M queries/month
        </div>
        <div className='grid grid-cols-2 gap-4'>
          <div className='bg-white rounded border border-gray-200 p-3'>
            <div
              className='text-xs font-semibold mb-1'
              style={{ color: '#ea580c' }}
            >
              Dense Embeddings
            </div>
            <div className='text-2xl font-bold text-gray-900'>
              ~$3.10
              <span className='text-xs text-gray-500'>/month</span>
            </div>
          </div>
          <div className='bg-white rounded border border-gray-200 p-3'>
            <div
              className='text-xs font-semibold mb-1'
              style={{ color: '#0891b2' }}
            >
              Late Interaction
            </div>
            <div className='text-2xl font-bold text-gray-900'>
              ~$1,280
              <span className='text-xs text-gray-500'>/month</span>
            </div>
          </div>
        </div>
        <div className='mt-4 text-xs text-gray-600 space-y-2'>
          <p>
            <strong>How we calculated this:</strong> Scaling from 15k to 100k
            documents, dense storage grows to ~620MB and late interaction to
            ~4GB. Query cost is storage × number of queries at{' '}
            <a
              href='https://turbopuffer.com/pricing'
              target='_blank'
              className='underline'
            >
              $5/PB
            </a>
            . Dense: 620MB x 1M queries = 620TB = ~$3. Late interaction requires
            ~64 sub-queries per search (2 stages x ~32 query tokens), so 1M user
            searches = 64M vector queries. That&apos;s ~4GB x 64M = ~256PB =
            ~$1,280.
          </p>
          <p>
            Late interaction is worth it when precision matters. For example,
            legal, medical, or compliance documents. For general-purpose search,
            dense is more cost-effective. To reduce late interaction costs, use
            shorter queries, lower FIRST_STAGE_TOP_K, or use it selectively as a{' '}
            <a
              href='https://www.pinecone.io/learn/series/rag/rerankers/'
              target='_blank'
              className='underline'
            >
              reranking step
            </a>
            .
          </p>
        </div>
      </div>
    </div>
  );
}

/*
Cost calculation:

Dense:
actual storage:
15k documents = 92.36MB
100k documents = (100 / 15) * 92.36MB = ~620MB
storage: 620MB * $0.33/GB = ~$0.20/month
writes: 620MB * $2.00/GB * 50% discount = ~$0.62/month
queries:
15k documents, 100 queries = 25.6 GB
100k documents, 1M queries = 620MB * 1,000,000 = ~620TB
queries: 620TB * $5.00/PB = ~$3.10/month

*technically this was too small number of documents and turbopuffer just used billable minimums of 256MB per query, so not perfect test at all*

Late:
*each query is actually 64 sub-queries*

actual storage:
15k documents = 597.73MB
100k documents = (100 / 15) * 597.73MB = ~3.98GB
storage: 3.98GB * $0.33/GB = ~$1.31/month
writes: 3.98GB * $2.00/GB * 50% discount = ~$3.98/month
queries:
15k documents, 100 queries = 3.83TB
100k documents, 1M queries = 3.98GB * (1,000,000 * 64) = ~254.72PB
queries: 254.72PB * $5.00/PB = ~$1273.60/month

The real cost difference emerges at scale. Late interaction requires
~64 sub-queries per search (2 stages × ~32 query tokens), meaning 1M
user searches become ~64M vector queries. Combined with 6.5× more
storage, late interaction costs grow significantly faster as you
scale documents or query volume.
*/
