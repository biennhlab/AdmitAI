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
});
