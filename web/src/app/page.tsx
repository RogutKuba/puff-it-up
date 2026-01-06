// redirect to /late-interaction

import { redirect } from 'next/navigation';

export default function Home() {
  redirect('/late-interaction');
}
