import { describe, expect, it, vi } from 'vitest';

import { parseChatStream } from '@/lib/chatErrors.mjs';

function streamResponse(parts, cancel = vi.fn()) {
  const encoder = new TextEncoder();
  const body = new ReadableStream({
    start(controller) {
      for (const part of parts) {
        controller.enqueue(typeof part === 'string' ? encoder.encode(part) : part);
      }
      controller.close();
    },
    cancel
  });
  return new Response(body, { headers: { 'content-type': 'text/event-stream' } });
}

describe('chat SSE parsing', () => {
  it('preserves split UTF-8 chunks and metadata on the final line', async () => {
    const encodedChunk = new TextEncoder().encode('data: {"type":"chunk","text":"Học phí"}\r\n\r\n');
    const metadata = { type: 'metadata', session_id: 'session-1', citations: [] };
    const onChunk = vi.fn();
    const onMetadata = vi.fn();
    const response = streamResponse([
      encodedChunk.slice(0, 32),
      encodedChunk.slice(32),
      `data: ${JSON.stringify(metadata)}`
    ]);

    await parseChatStream(response, onChunk, onMetadata);

    expect(onChunk).toHaveBeenCalledExactlyOnceWith('Học phí');
    expect(onMetadata).toHaveBeenCalledExactlyOnceWith(metadata);
    expect(response.body.locked).toBe(false);
  });

  it('throws a structured error when the final error data has no newline', async () => {
    const response = streamResponse([
      'event: error\n',
      'data: {"code":"LLM_TIMEOUT","message":"Timeout"}'
    ]);

    await expect(parseChatStream(response)).rejects.toMatchObject({ code: 'LLM_TIMEOUT' });
    expect(response.body.locked).toBe(false);
  });

  it.each([
    '',
    'data: not-json\n\n',
    'data: null\n\n',
    'data: {"type":"chunk","text":42}\n\n',
    'data: {"type":"metadata","citations":[]}\n\n',
    '{"answer":"This is not a stream"}'
  ])('rejects empty or invalid streamed responses: %s', async (body) => {
    await expect(parseChatStream(streamResponse([body]))).rejects.toMatchObject({
      code: 'INVALID_RESPONSE'
    });
  });

  it('rejects a response without a body', async () => {
    await expect(parseChatStream(new Response(null))).rejects.toMatchObject({
      code: 'INVALID_RESPONSE'
    });
  });

  it('cancels the reader after a streamed error', async () => {
    const cancel = vi.fn();
    const body = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(
          'event: error\ndata: {"code":"RAG_UNAVAILABLE"}\n\n'
        ));
      },
      cancel
    });

    await expect(parseChatStream(new Response(body))).rejects.toMatchObject({
      code: 'RAG_UNAVAILABLE'
    });
    expect(cancel).toHaveBeenCalledOnce();
    expect(body.locked).toBe(false);
  });
});
