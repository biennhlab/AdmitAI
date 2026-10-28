import { describe, expect, it, vi } from 'vitest';

import { friendlyChatError, marksServiceUnavailable, parseChatResponse, parseChatStream } from '@/lib/chatErrors.mjs';

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
  it.each([parseChatResponse, parseChatStream])('keeps a safe request ID with the structured API rejection', async (parse) => {
    const id = '1234567890abcdef1234567890abcdef';
    const response = Response.json({ detail: { code: 'GENERATION_UNGROUNDED' } }, {
      status: 502, headers: { 'x-request-id': id }
    });
    await expect(parse(response)).rejects.toMatchObject({ code: 'GENERATION_UNGROUNDED', status: 502, requestId: id });
  });
  it('explains unsupported generation without marking retrieval unavailable', () => {
    const error = { code: 'GENERATION_UNGROUNDED' };
    expect(friendlyChatError(error)).toContain('chưa thể xác minh');
    expect(marksServiceUnavailable(error)).toBe(false);
  });

  it('handles a grounding SSE rejection without displaying answer chunks', async () => {
    const onChunk = vi.fn();
    const onMetadata = vi.fn();
    const response = streamResponse([
      'event: error\ndata: {"code":"GENERATION_UNGROUNDED","message":"Cannot verify"}\n\n'
    ]);

    await expect(parseChatStream(response, onChunk, onMetadata)).rejects.toMatchObject({
      code: 'GENERATION_UNGROUNDED'
    });
    expect(onChunk).not.toHaveBeenCalled();
    expect(onMetadata).not.toHaveBeenCalled();
    expect(response.body.locked).toBe(false);
  });
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
