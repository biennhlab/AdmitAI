import { afterEach, describe, expect, it, vi } from 'vitest';

import { POST as proxyChat } from '@/app/api/chat/route.js';
import { proxyBackend } from '@/lib/backendProxy.mjs';

describe('controlled backend proxy', () => {
  it.each(['json', 'sse', 'error'])('preserves the backend request ID for %s responses', async (kind) => {
    const id = '1234567890abcdef1234567890abcdef';
    const upstream = kind === 'sse'
      ? new Response('data: {"type":"chunk","text":"Verified"}\n\n', {
        headers: { 'content-type': 'text/event-stream', 'x-request-id': id }
      })
      : Response.json(kind === 'error' ? { detail: { code: 'GENERATION_UNGROUNDED' } } : { answer: 'Verified' }, {
        status: kind === 'error' ? 502 : 200, headers: { 'x-request-id': id }
      });
    const response = await proxyBackend('/api/chat', { fetchImpl: vi.fn().mockResolvedValue(upstream), timeoutMs: 100 });
    expect(response.headers.get('x-request-id')).toBe(id);
    await response.text();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

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
        body: JSON.stringify({ message: 'Xin chào' }),
        headers: expect.objectContaining({ accept: 'application/json' })
      })
    );
  });

  it('requests and forwards a chat event stream without buffering it as JSON', async () => {
    const sse = 'data: {"type":"chunk","text":"Xin chào"}\n\n';
    const fetchImpl = vi.fn().mockResolvedValue(new Response(sse, {
      headers: { 'content-type': 'text/event-stream; charset=utf-8' }
    }));
    vi.stubGlobal('fetch', fetchImpl);

    const response = await proxyChat(new Request('http://frontend.test/api/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ message: 'Xin chào' })
    }));

    expect(fetchImpl).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/chat$/),
      expect.objectContaining({
        headers: expect.objectContaining({ accept: 'text/event-stream' })
      })
    );
    expect(response.headers.get('content-type')).toContain('text/event-stream');
    expect(await response.text()).toBe(sse);
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

  it('preserves generation grounding failure separately from retrieval errors', async () => {
    const response = await proxyBackend('/api/chat', {
      fetchImpl: vi.fn().mockResolvedValue(Response.json(
        { detail: { code: 'GENERATION_UNGROUNDED', message: 'internal unsupported output' } },
        { status: 502 }
      )), timeoutMs: 100
    });
    expect(response.status).toBe(503);
    const payload = await response.json();
    expect(payload.error.code).toBe('GENERATION_UNGROUNDED');
    expect(JSON.stringify(payload)).not.toContain('internal unsupported output');
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

    expect(response.status).toBe(503);
    expect((await response.json()).error.code).toBe('INVALID_BACKEND_RESPONSE');
  });
});
