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
  it('retries the failed question once on user click and labels verified evidence recovery', async () => {
    const question = 'PTIT có những phương thức tuyển sinh nào trong năm 2026?';
    const events = 'data: {"type":"chunk","text":"- Xét tuyển tài năng."}\n\n'
      + 'data: {"type":"metadata","route_type":"general","answer_mode":"evidence","session_id":"recovered","citations":[{"marker":1,"title":"Tuyển sinh 2026","source_url":"https://example.test/admissions"}]}\n\n';
    global.fetch = vi.fn().mockResolvedValueOnce(healthyResponse)
      .mockResolvedValueOnce(errorResponse(503, 'LLM_PROVIDER_ERROR'))
      .mockResolvedValueOnce(new Response(events, { headers: { 'content-type': 'text/event-stream' } }));
    render(<FloatingChatbot />);
    fireEvent.click(screen.getByLabelText('Mở chat'));
    const input = screen.getByPlaceholderText('Nhập câu hỏi của bạn...');
    await waitFor(() => expect(input).toBeEnabled());
    fireEvent.change(input, { target: { value: question } });
    fireEvent.submit(input.closest('form'));
    const retry = await screen.findByRole('button', { name: 'Gửi lại câu hỏi' });
    expect(input).toBeEnabled();
    expect(global.fetch).toHaveBeenCalledTimes(2);
    fireEvent.click(retry);
    expect(await screen.findByText('Xét tuyển tài năng.')).toBeInTheDocument();
    expect(await screen.findByText(/Nội dung dưới đây được trích xuất và đối chiếu/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Nguồn tham khảo (1)' })).toBeInTheDocument();
    expect(screen.getAllByText(question)).toHaveLength(1);
    expect(global.fetch).toHaveBeenCalledTimes(3);
    expect(JSON.parse(global.fetch.mock.calls[2][1].body).message).toBe(question);
    expect(screen.queryByRole('button', { name: 'Gửi lại câu hỏi' })).toBeNull();
    expect(console.error).not.toHaveBeenCalled();
  });

  it('keeps manual retry available after another provider failure without duplicating the question', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce(healthyResponse)
      .mockResolvedValueOnce(errorResponse(503, 'LLM_PROVIDER_ERROR'))
      .mockResolvedValueOnce(errorResponse(503, 'LLM_PROVIDER_ERROR'));
    render(<FloatingChatbot />);
    fireEvent.click(screen.getByLabelText('Mở chat'));
    const input = screen.getByPlaceholderText('Nhập câu hỏi của bạn...');
    await waitFor(() => expect(input).toBeEnabled());
    fireEvent.change(input, { target: { value: 'Học phí?' } });
    fireEvent.submit(input.closest('form'));
    fireEvent.click(await screen.findByRole('button', { name: 'Gửi lại câu hỏi' }));
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(3));
    expect(await screen.findByRole('button', { name: 'Gửi lại câu hỏi' })).toBeInTheDocument();
    expect(screen.getAllByText('Học phí?')).toHaveLength(1);
    expect(input).toBeEnabled();
    expect(console.error).not.toHaveBeenCalled();
  });
  beforeEach(() => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
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

  it.each([
    ['điểm chuẩn cn1', 'INVALID_GENERATION', false],
    ['Điểm chuẩn ngành Công nghệ thông tin mã 7480201 năm 2026 là bao nhiêu?', 'INVALID_GENERATION', false],
    ['Học phí PTIT năm học 2026–2027 là bao nhiêu?', 'LLM_PROVIDER_ERROR', false]
  ])('keeps the exact failed question safe: %s', async (question, code, unavailable) => {
    global.fetch = vi.fn()
      .mockResolvedValueOnce(healthyResponse)
      .mockResolvedValueOnce(errorResponse(502, code));

    const { container } = render(<FloatingChatbot />);
    fireEvent.click(screen.getByLabelText('Mở chat'));
    const input = screen.getByPlaceholderText('Nhập câu hỏi của bạn...');
    await waitFor(() => expect(input).toBeEnabled());
    fireEvent.change(input, { target: { value: question } });
    fireEvent.submit(input.closest('form'));
    await waitFor(() => expect(console.warn).toHaveBeenCalledWith(
      'Chat request rejected:', { code, status: 502 }
    ));
    expect(console.error).not.toHaveBeenCalled();
    await waitFor(() => expect(screen.queryByLabelText('Đang trả lời')).not.toBeInTheDocument());
    expect(JSON.parse(global.fetch.mock.calls[1][1].body).message).toBe(question);
    expect(container.querySelector('[class*="citationsContainer"]')).toBeNull();
    expect(container.textContent).not.toContain(code);
    expect(container.textContent).toContain(unavailable
      ? 'Dịch vụ tư vấn đang tạm thời chưa sẵn sàng.'
      : code === 'LLM_PROVIDER_ERROR' ? 'Dịch vụ tư vấn đang tạm thời chưa sẵn sàng.'
        : 'Phản hồi từ dịch vụ xử lý không hợp lệ.');
    if (unavailable) expect(input).toBeDisabled();
    else expect(input).toBeEnabled();
  });

  it('renders the reported cutoff question and validated answer without a development error', async () => {
    const question = 'Điểm chuẩn ngành Công nghệ thông tin mã 7480201 năm 2026 là bao nhiêu?';
    const answer = 'Điểm chuẩn ngành Công nghệ thông tin mã 7480201 năm 2026 là 23 điểm.';
    const metadata = { type: 'metadata', route_type: 'general', session_id: 'cutoff-session',
      citations: [{ marker: 1, title: 'Điểm chuẩn 2026', snippet: 'Công nghệ thông tin | 7480201 | 23.00' }] };
    global.fetch = vi.fn()
      .mockResolvedValueOnce(healthyResponse)
      .mockResolvedValueOnce(new Response(
        `data: ${JSON.stringify({ type: 'chunk', text: answer })}\n\n`
        + `data: ${JSON.stringify(metadata)}\n\n`,
        { headers: { 'content-type': 'text/event-stream' } }
      ));
    render(<FloatingChatbot />);
    fireEvent.click(screen.getByLabelText('Mở chat'));
    const input = screen.getByPlaceholderText('Nhập câu hỏi của bạn...');
    await waitFor(() => expect(input).toBeEnabled());
    fireEvent.change(input, { target: { value: question } });
    fireEvent.submit(input.closest('form'));
    expect(await screen.findByText(answer)).toBeInTheDocument();
    expect(await screen.findByText('[1] Điểm chuẩn 2026')).toBeInTheDocument();
    await waitFor(() => expect(input).toBeEnabled());
    expect(console.error).not.toHaveBeenCalled();
    expect(console.warn).not.toHaveBeenCalled();
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });

  it('renders escaped citation Markdown without loading images or interpreting HTML', async () => {
    const metadata = {
      type: 'metadata', session_id: 'snippet-test', citations: [{
        marker: 1, title: 'Nguồn học phí',
        snippet: '**Học phí** năm 2026\\-2027: 35,9 triệu đồng/năm. <script>unsafeSnippet()</script> ![tracking](https://example.test/pixel.png)'
      }]
    };
    global.fetch = vi.fn()
      .mockResolvedValueOnce(healthyResponse)
      .mockResolvedValueOnce(new Response(
        'data: {"type":"chunk","text":"Học phí [1]"}\n\n'
        + `data: ${JSON.stringify(metadata)}\n\n`,
        { headers: { 'content-type': 'text/event-stream' } }
      ));
    const { container } = render(<FloatingChatbot />);
    fireEvent.click(screen.getByLabelText('Mở chat'));
    const input = screen.getByPlaceholderText('Nhập câu hỏi của bạn...');
    await waitFor(() => expect(input).toBeEnabled());
    fireEvent.change(input, { target: { value: 'Học phí PTIT năm học 2026–2027 là bao nhiêu?' } });
    fireEvent.submit(input.closest('form'));
    await screen.findByText('[1] Nguồn học phí');
    const snippet = container.querySelector('[class*="citationSnippet"]');
    expect(snippet.textContent).toContain('Học phí năm 2026-2027: 35,9 triệu đồng/năm.');
    expect(snippet.textContent).not.toContain('\\-');
    expect(snippet.querySelector('strong')).toHaveTextContent('Học phí');
    expect(snippet.querySelector('script, img')).toBeNull();
  });

  it('renders verified paragraphs, lists and tables with one source section at the end', async () => {
    const answer = 'Điểm SAT tối thiểu **1130/1600**.\n\n- Điểm ACT tối thiểu 25/36.\n\n'
      + '| Kỳ thi | Ngưỡng điểm |\n| --- | --- |\n| SAT | 1130/1600 |';
    const metadata = { type: 'metadata', session_id: 'end-sources', route_type: 'general', citations: [
      { marker: 1, chunk_id: 'sat', title: 'Quy định SAT', source_url: 'https://example.test/sat' },
      { marker: 2, chunk_id: 'act', title: 'Quy định ACT', source_url: 'https://example.test/act' }
    ] };
    global.fetch = vi.fn().mockResolvedValueOnce(healthyResponse).mockResolvedValueOnce(new Response(
      `data: ${JSON.stringify({ type: 'chunk', text: answer })}\n\n`
      + `data: ${JSON.stringify(metadata)}\n\n`,
      { headers: { 'content-type': 'text/event-stream' } }
    ));
    const { container } = render(<FloatingChatbot />);
    fireEvent.click(screen.getByLabelText('Mở chat'));
    const input = screen.getByPlaceholderText('Nhập câu hỏi của bạn...');
    await waitFor(() => expect(input).toBeEnabled());
    fireEvent.change(input, { target: { value: 'Ngưỡng SAT và ACT?' } });
    fireEvent.submit(input.closest('form'));
    const sourceButton = await screen.findByRole('button', { name: /Nguồn tham khảo \(2\)/ });
    fireEvent.click(sourceButton);
    const sources = container.querySelector('[class*="citationsContainer"]');
    const markdown = sources.previousElementSibling;
    expect(markdown.textContent).not.toMatch(/\[\d+\]/);
    expect(markdown.querySelector('strong')).toHaveTextContent('1130/1600');
    expect(markdown.querySelectorAll('li')).toHaveLength(1);
    expect(markdown.querySelector('table')).toBeInTheDocument();
    expect(container.querySelectorAll('[class*="citationsContainer"]')).toHaveLength(1);
    expect(screen.getByRole('link', { name: '[1] Quy định SAT' })).toHaveAttribute('href', 'https://example.test/sat');
    expect(screen.getByRole('link', { name: '[2] Quy định ACT' })).toHaveAttribute('rel', 'noopener noreferrer');
    expect(localStorage.getItem('admitai_session_id')).toBe('end-sources');
  });

  it('shows a legitimate abstention without a fabricated source section', async () => {
    const answer = 'Mình chưa tìm thấy thông tin này trong dữ liệu tuyển sinh hiện có.';
    global.fetch = vi.fn().mockResolvedValueOnce(healthyResponse).mockResolvedValueOnce(new Response(
      `data: ${JSON.stringify({ type: 'metadata', route_type: 'out_of_scope', session_id: 'abstention', citations: [] })}\n\n`
      + `data: ${JSON.stringify({ type: 'chunk', text: answer })}\n\n`,
      { headers: { 'content-type': 'text/event-stream' } }
    ));
    render(<FloatingChatbot />);
    fireEvent.click(screen.getByLabelText('Mở chat'));
    const input = screen.getByPlaceholderText('Nhập câu hỏi của bạn...');
    await waitFor(() => expect(input).toBeEnabled());
    fireEvent.change(input, { target: { value: 'PTIT có tuyển ngành Thú y năm 2026 không?' } });
    fireEvent.submit(input.closest('form'));
    expect(await screen.findByText(answer)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Nguồn tham khảo/ })).toBeNull();
    expect(console.warn).not.toHaveBeenCalled();
  });

  it.each([['same_document', 1], ['different_versions', 2]])('deduplicates %s while retaining evidence locations', async (kind, count) => {
    const citations = [
      { marker: 1, chunk_id: 'sat', doc_id: 'admissions', title: 'Thông báo tuyển sinh',
        source_url: 'https://example.test/admissions', page: 1, snippet: 'SAT tối thiểu 1130/1600.', published_at: '2026-01-01' },
      { marker: 2, chunk_id: 'act', doc_id: 'admissions', title: 'Thông báo tuyển sinh',
        source_url: 'https://example.test/admissions', page: 2, snippet: 'ACT tối thiểu 25/36.',
        published_at: kind === 'same_document' ? '2026-01-01' : '2026-02-01' }
    ];
    global.fetch = vi.fn().mockResolvedValueOnce(healthyResponse).mockResolvedValueOnce(new Response(
      'data: {"type":"chunk","text":"Điểm SAT tối thiểu 1130/1600 và ACT tối thiểu 25/36."}\n\n'
      + `data: ${JSON.stringify({ type: 'metadata', citations, session_id: 'dedup' })}\n\n`,
      { headers: { 'content-type': 'text/event-stream' } }
    ));
    render(<FloatingChatbot />);
    fireEvent.click(screen.getByLabelText('Mở chat'));
    const input = screen.getByPlaceholderText('Nhập câu hỏi của bạn...');
    await waitFor(() => expect(input).toBeEnabled());
    fireEvent.change(input, { target: { value: 'Điểm SAT ACT tối thiểu?' } });
    fireEvent.submit(input.closest('form'));
    const toggle = await screen.findByRole('button', { name: `Nguồn tham khảo (${count})` });
    fireEvent.click(toggle);
    expect(screen.getAllByRole('link', { name: /Thông báo tuyển sinh/ })).toHaveLength(count);
    expect(screen.getByText('Trang 1')).toBeInTheDocument();
    expect(screen.getByText('Trang 2')).toBeInTheDocument();
    expect(screen.getByText('SAT tối thiểu 1130/1600.')).toBeInTheDocument();
    expect(screen.getByText('ACT tối thiểu 25/36.')).toBeInTheDocument();
  });
});
