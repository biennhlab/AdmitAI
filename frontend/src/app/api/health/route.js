import { proxyBackend } from '@/lib/backendProxy.mjs';

export async function GET() {
  return proxyBackend('/api/health', { timeoutMs: 5000 });
}
