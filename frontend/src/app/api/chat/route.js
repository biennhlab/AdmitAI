import { proxyBackend, structuredError } from '@/lib/backendProxy.mjs';

export async function POST(request) {
  let body;
  try {
    body = await request.json();
  } catch {
    return structuredError('INTERNAL_ERROR', 400);
  }

  return proxyBackend('/api/chat', { method: 'POST', body });
}
