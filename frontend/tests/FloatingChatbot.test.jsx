import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import FloatingChatbot from '@/components/chatbot/FloatingChatbot';

const healthyResponse = {
  ok: true,
  status: 200,
  json: async () => ({
    status: 'healthy',
    components: { rag: { ready: true } }
  })
};

function errorResponse(status, code) {
  return {
    ok: false,
    status,
    json: async () => ({ error: { code } })
  };
}

describe('FloatingChatbot service errors', () => {
  beforeEach(() => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('shows a friendly unavailable message and recovers when retry succeeds', async () => {
    global.fetch = vi.fn()
      .mockResolvedValueOnce(healthyResponse)
      .mockResolvedValueOnce(errorResponse(503, 'BACKEND_UNAVAILABLE'))
      .mockResolvedValueOnce(healthyResponse);

    render(<FloatingChatbot />);
    fireEvent.click(screen.getByLabelText('Mở chat'));

    const input = screen.getByPlaceholderText('Nhập câu hỏi của bạn...');
    await waitFor(() => expect(input).toBeEnabled());
    fireEvent.change(input, { target: { value: 'Cho mình hỏi học phí?' } });
    fireEvent.submit(input.closest('form'));

    expect(await screen.findByText(
      'Dịch vụ tư vấn đang tạm thời chưa sẵn sàng. Bạn thử lại sau một chút nhé.'
    )).toBeInTheDocument();
    expect(screen.queryByText(/Qdrant|RAG|embedding|ECONNRESET/i)).not.toBeInTheDocument();
    expect(screen.queryByTitle('Hữu ích')).not.toBeInTheDocument();
    expect(input).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: 'Thử lại' }));
    await waitFor(() => expect(input).toBeEnabled());
  });

  it('resets loading after a provider timeout', async () => {
    global.fetch = vi.fn()
      .mockResolvedValueOnce(healthyResponse)
      .mockResolvedValueOnce(errorResponse(504, 'LLM_TIMEOUT'));

    render(<FloatingChatbot />);
    fireEvent.click(screen.getByLabelText('Mở chat'));

    const input = screen.getByPlaceholderText('Nhập câu hỏi của bạn...');
    await waitFor(() => expect(input).toBeEnabled());
    fireEvent.change(input, { target: { value: 'Điểm chuẩn là bao nhiêu?' } });
    fireEvent.submit(input.closest('form'));

    expect(await screen.findByText(
      'Mình đang mất nhiều thời gian hơn bình thường để xử lý câu hỏi này. Bạn thử lại giúp mình nhé.'
    )).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByLabelText('Đang trả lời')).not.toBeInTheDocument());
    expect(input).toBeEnabled();
  });

  it('keeps partial text when a stream fails without duplicate message keys', async () => {
    vi.spyOn(Date, 'now').mockReturnValue(1000);
    global.fetch = vi.fn()
      .mockResolvedValueOnce(healthyResponse)
      .mockResolvedValueOnce(new Response(
        'data: {"type":"chunk","text":"Nội dung đã nhận"}\n\n'
        + 'event: error\ndata: {"code":"LLM_TIMEOUT"}\n\n',
        { headers: { 'content-type': 'text/event-stream' } }
      ));

    render(<FloatingChatbot />);
    fireEvent.click(screen.getByLabelText('Mở chat'));
    const input = screen.getByPlaceholderText('Nhập câu hỏi của bạn...');
    await waitFor(() => expect(input).toBeEnabled());
    fireEvent.change(input, { target: { value: 'Học phí?' } });
    fireEvent.submit(input.closest('form'));

    expect(await screen.findByText(
      'Mình đang mất nhiều thời gian hơn bình thường để xử lý câu hỏi này. Bạn thử lại giúp mình nhé.'
    )).toBeInTheDocument();
    expect(screen.getByText('Nội dung đã nhận')).toBeInTheDocument();
    expect(console.error.mock.calls.flat().join(' ')).not.toMatch(/same key/i);
  });

  it.each([
    [{ marker: 2, source_url: 'https://example.test/admissions' }, '[2] Đề án tuyển sinh'],
    [{ marker: 2 }, '[2] Đề án tuyển sinh'],
    [{}, '[1] Đề án tuyển sinh']
  ])('uses citation markers and preserves older citation numbering', async (citation, label) => {
    const metadata = {
      type: 'metadata',
      session_id: 'session-1',
      citations: [{ title: 'Đề án tuyển sinh', ...citation }]
    };
    global.fetch = vi.fn()
      .mockResolvedValueOnce(healthyResponse)
      .mockResolvedValueOnce(new Response(
        'data: {"type":"chunk","text":"Thông tin tuyển sinh [2]"}\n\n'
        + `data: ${JSON.stringify(metadata)}`,
        { headers: { 'content-type': 'text/event-stream' } }
      ));

    render(<FloatingChatbot />);
    fireEvent.click(screen.getByLabelText('Mở chat'));
    const input = screen.getByPlaceholderText('Nhập câu hỏi của bạn...');
    await waitFor(() => expect(input).toBeEnabled());
    fireEvent.change(input, { target: { value: 'Phương thức tuyển sinh?' } });
    fireEvent.submit(input.closest('form'));

    expect(await screen.findByText(label)).toBeInTheDocument();
    expect(localStorage.getItem('admitai_session_id')).toBe('session-1');
    await waitFor(() => expect(input).toBeEnabled());
  });
});
