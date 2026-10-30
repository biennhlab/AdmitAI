const UNAVAILABLE_CODES = new Set([
  'BACKEND_UNAVAILABLE',
  'RAG_UNAVAILABLE',
  'LLM_AUTHENTICATION_FAILED',
  'LLM_RATE_LIMITED',
  'LLM_UNAVAILABLE',
  'LLM_PROVIDER_ERROR'
]);

const TIMEOUT_CODES = new Set(['BACKEND_TIMEOUT', 'LLM_TIMEOUT']);

export class ChatApiError extends Error {
  constructor(code, status, technicalMessage = '', requestId = null) {
    super(technicalMessage || code);
    this.name = 'ChatApiError';
    this.code = code;
    this.status = status;
    if (/^[a-f0-9]{32}$/i.test(requestId || '')) this.requestId = requestId;
  }
}

export async function parseChatResponse(response) {
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new ChatApiError('INVALID_RESPONSE', response.status, 'Chat API returned non-JSON data');
  }

  if (!response.ok) {
    const code = payload?.error?.code || payload?.detail?.code || 'SERVER_ERROR';
    throw new ChatApiError(code, response.status, `Chat API returned ${response.status} (${code})`, response.headers?.get?.('x-request-id'));
  }

  if (!payload || typeof payload.answer !== 'string') {
    throw new ChatApiError('INVALID_RESPONSE', response.status, 'Chat API response has no answer');
  }

  return payload;
}

export function friendlyChatError(error, isOnline = true) {
  if (!isOnline) return 'Có vẻ kết nối mạng đang gặp vấn đề.';
  if (error?.code === 'INVALID_GENERATION') {
    return 'Phản hồi từ dịch vụ xử lý không hợp lệ. Bạn thử gửi lại câu hỏi nhé.';
  }
  if (TIMEOUT_CODES.has(error?.code)) {
    return 'Mình đang mất nhiều thời gian hơn bình thường để xử lý câu hỏi này. Bạn thử lại giúp mình nhé.';
  }
  if (UNAVAILABLE_CODES.has(error?.code)) {
    return 'Dịch vụ tư vấn đang tạm thời chưa sẵn sàng. Bạn thử lại sau một chút nhé.';
  }
  return 'Xin lỗi, kết nối đến máy chủ bị lỗi. Vui lòng thử lại sau.';
}

export function marksServiceUnavailable(error) {
  return ['BACKEND_UNAVAILABLE', 'RAG_UNAVAILABLE', 'INVALID_BACKEND_RESPONSE',
    'LLM_AUTHENTICATION_FAILED'].includes(error?.code);
}

export function canRetryChatError(error) {
  return ['LLM_PROVIDER_ERROR', 'LLM_UNAVAILABLE', 'LLM_RATE_LIMITED',
    'LLM_TIMEOUT', 'BACKEND_TIMEOUT', 'INVALID_GENERATION'].includes(error?.code);
}

export async function parseChatStream(response, onChunk, onMetadata) {
  if (!response.ok) {
    let payload;
    try {
      payload = await response.json();
    } catch {
      throw new ChatApiError('INVALID_RESPONSE', response.status, 'Chat API returned non-JSON data');
    }
    const code = payload?.error?.code || payload?.detail?.code || 'SERVER_ERROR';
    throw new ChatApiError(code, response.status, `Chat API returned ${response.status} (${code})`, response.headers?.get?.('x-request-id'));
  }

  if (!response.body) {
    throw new ChatApiError('INVALID_RESPONSE', response.status, 'Chat API response has no body');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let isErrorEvent = false;
  let receivedText = false;
  let completed = false;

  const consumeLine = (rawLine) => {
    const line = rawLine.replace(/\r$/, '');
    if (line === '') {
      isErrorEvent = false;
      return;
    }
    if (line.startsWith('event:')) {
      isErrorEvent = line.slice(6).trim() === 'error';
      return;
    }
    if (!line.startsWith('data:')) return;

    let data;
    try {
      data = JSON.parse(line.slice(5).trimStart());
    } catch {
      throw new ChatApiError('INVALID_RESPONSE', response.status, 'Chat stream returned non-JSON data');
    }
    if (!data || typeof data !== 'object') {
      throw new ChatApiError('INVALID_RESPONSE', response.status, 'Chat stream returned invalid data');
    }
    if (isErrorEvent) {
      throw new ChatApiError(data.code || 'SERVER_ERROR', response.status || 500, data.message || 'Stream error', response.headers?.get?.('x-request-id'));
    }
    if (data.type === 'chunk') {
      if (typeof data.text !== 'string') {
        throw new ChatApiError('INVALID_RESPONSE', response.status, 'Chat stream chunk has no text');
      }
      receivedText ||= data.text.length > 0;
      if (onChunk) onChunk(data.text);
    } else if (data.type === 'metadata' && onMetadata) {
      onMetadata(data);
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        if (buffer) consumeLine(buffer);
        completed = true;
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) consumeLine(line);
    }
    if (!receivedText) {
      throw new ChatApiError('INVALID_RESPONSE', response.status, 'Chat stream response has no answer');
    }
  } finally {
    try {
      if (!completed) {
        try {
          await reader.cancel();
        } catch {
          // Preserve the original stream or API error if cancellation also fails.
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}
