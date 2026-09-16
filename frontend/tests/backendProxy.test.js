import { describe, expect, it, vi } from 'vitest';

import { proxyBackend } from '@/lib/backendProxy.mjs';

describe('controlled backend proxy', () => {
  it('forwards a successful JSON payload unchanged', async () => {
    const payload = { answer: 'Xin chào', citations: [], session_id: 'session-1' };
    const fetchImpl = vi.fn().mockResolvedValue(Response.json(payload));

    const response = await proxyBackend('/api/chat', {
      method: 'POST',
      body: { message: 'Xin chào' },
      fetchImpl,
      baseUrl: 'http://backend:8000',
      timeoutMs: 100
    });

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      'http://backend:8000/api/chat',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ message: 'Xin chào' })
      })
    );
  });

  it('maps a backend 503 without leaking its raw detail', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(Response.json(
      { detail: { code: 'RAG_UNAVAILABLE', message: 'secret internal path' } },
      { status: 503 }
    ));

    const response = await proxyBackend('/api/chat', { fetchImpl, timeoutMs: 100 });
    const payload = await response.json();

    expect(response.status).toBe(503);
    expect(payload.error.code).toBe('RAG_UNAVAILABLE');
    expect(JSON.stringify(payload)).not.toContain('secret internal path');
  });

  it.each(['ECONNREFUSED', 'ECONNRESET'])('maps %s to BACKEND_UNAVAILABLE', async (code) => {
    const networkError = Object.assign(new Error(code), { code });
    const response = await proxyBackend('/api/chat', {
      fetchImpl: vi.fn().mockRejectedValue(networkError),
      timeoutMs: 100
    });

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      error: {
        code: 'BACKEND_UNAVAILABLE',
        message: 'Dịch vụ tư vấn đang tạm thời chưa sẵn sàng.'
      }
    });
  });

  it('returns a structured timeout error', async () => {
    const fetchImpl = vi.fn((url, { signal }) => new Promise((resolve, reject) => {
      void url;
      void resolve;
      signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    }));

    const response = await proxyBackend('/api/chat', { fetchImpl, timeoutMs: 5 });

    expect(response.status).toBe(504);
    expect((await response.json()).error.code).toBe('BACKEND_TIMEOUT');
  });

  it('rejects a non-JSON backend response', async () => {
    const response = await proxyBackend('/api/chat', {
      fetchImpl: vi.fn().mockResolvedValue(new Response('<html>bad gateway</html>')),
      timeoutMs: 100
    });

    expect(response.status).toBe(502);
    expect((await response.json()).error.code).toBe('INVALID_BACKEND_RESPONSE');
  });
});
