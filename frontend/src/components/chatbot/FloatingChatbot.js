'use client';
import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  ChatApiError,
  canRetryChatError,
  friendlyChatError,
  marksServiceUnavailable,
  parseChatStream
} from '@/lib/chatErrors.mjs';
import styles from './FloatingChatbot.module.css';

const API_URL = '/api/chat';

function citationGroups(citations = []) {
  const groups = [];
  const known = new Map();
  citations.forEach((citation, index) => {
    // Group only an established document identity. Keep every distinct evidence
    // location/snippet and keep different document versions separate.
    const identity = citation.doc_id || citation.source_url
      ? JSON.stringify([citation.doc_id || '', citation.source_url || '',
        citation.source_type || '', citation.published_at || '']) : null;
    let group = identity ? known.get(identity) : null;
    if (!group) {
      group = { source: citation, markers: [], members: [] };
      groups.push(group);
      if (identity) known.set(identity, group);
    }
    const marker = citation.marker ?? index + 1;
    if (!group.markers.includes(marker)) group.markers.push(marker);
    const evidenceKey = JSON.stringify([citation.chunk_id, citation.page, citation.section, citation.snippet]);
    if (!group.members.some(member => member.key === evidenceKey)) {
      group.members.push({ ...citation, key: evidenceKey });
    }
  });
  return groups;
}

const markdownComponents = {
  a({ node, ...props }) {
    void node;
    return <a {...props} target="_blank" rel="noopener noreferrer" />;
  },
  table({ node, ...props }) {
    void node;
    return (
      <div className={styles.tableWrapper} role="region" aria-label="Bảng thông tin" tabIndex={0}>
        <table {...props} />
      </div>
    );
  }
};

export default function FloatingChatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'bot',
      content: 'Xin chào! Mình là AdmitAI, trợ lý ảo tư vấn tuyển sinh của Học viện Công nghệ Bưu chính Viễn thông (PTIT). Bạn cần hỗ trợ thông tin gì?',
      citations: []
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [serviceStatus, setServiceStatus] = useState('unknown');
  const sessionIdRef = useRef(null);
  const chatAreaRef = useRef(null);

  useEffect(() => {
    // Load session ID from localStorage on mount
    const savedSessionId = localStorage.getItem('admitai_session_id');
    if (savedSessionId) {
      sessionIdRef.current = savedSessionId;
    }
  }, []);

  useEffect(() => {
    // Scroll to bottom when messages change
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const checkService = async () => {
    setServiceStatus('checking');
    try {
      const response = await fetch('/api/health', { cache: 'no-store' });
      const data = await response.json();
      const ragReady = data?.components?.rag?.ready ?? data?.rag?.ready;
      setServiceStatus(response.ok && ragReady ? 'available' : 'unavailable');
    } catch (error) {
      console.error('Health check error:', error);
      setServiceStatus('unavailable');
    }
  };

  const toggleChat = () => {
    const willOpen = !isOpen;
    setIsOpen(willOpen);
    if (willOpen) {
      void checkService();
    }
  };

  const sendMessage = async (text, retryMessageId = null) => {
    if (
      !text
      || !text.trim()
      || isLoading
      || serviceStatus === 'checking'
      || serviceStatus === 'unavailable'
    ) return;
    
    setShowSuggestions(false);
    
    // Add user message
    const newUserMsg = { id: retryMessageId || Date.now(), role: 'user', content: text.trim() };
    setMessages((prev) => retryMessageId
      ? prev.filter(msg => msg.id !== `error-${retryMessageId}` && msg.id !== `bot-${retryMessageId}`)
      : [...prev, newUserMsg]);
    setInput('');
    setIsLoading(true);
    const newBotMsgId = `bot-${newUserMsg.id}`;

    try {
      const payload = { message: text.trim() };
      if (sessionIdRef.current) {
        payload.session_id = sessionIdRef.current;
      }
      
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      setServiceStatus('available');
      
      setMessages((prev) => [...prev, {
        id: newBotMsgId,
        role: 'bot',
        content: '',
        citations: []
      }]);
      
      await parseChatStream(response, (textChunk) => {
        setMessages(prev => prev.map(msg => 
          msg.id === newBotMsgId ? { ...msg, content: msg.content + textChunk } : msg
        ));
      }, (metadata) => {
        if (metadata.session_id && metadata.session_id !== sessionIdRef.current) {
          sessionIdRef.current = metadata.session_id;
          localStorage.setItem('admitai_session_id', metadata.session_id);
        }
        setMessages(prev => prev.map(msg => 
          msg.id === newBotMsgId ? { ...msg, citations: metadata.citations || [], answerMode: metadata.answer_mode } : msg
        ));
      });
    } catch (error) {
      if (error instanceof ChatApiError) {
        // Expected API rejections are handled below. Logging their Error object
        // at error level also opens Next.js's development error overlay.
        console.warn('Chat request rejected:', {
          code: error.code, status: error.status,
          ...(error.requestId ? { requestId: error.requestId } : {})
        });
      } else {
        console.error('Chat error:', error);
      }
      const isOnline = typeof navigator === 'undefined' || navigator.onLine !== false;
      if (!isOnline || marksServiceUnavailable(error)) {
        setServiceStatus('unavailable');
      }
      const errorMsg = {
        id: `error-${newUserMsg.id}`,
        role: 'bot',
        content: friendlyChatError(error, isOnline),
        isError: true,
        citations: []
      };
      if (canRetryChatError(error)) {
        errorMsg.retryQuestion = text.trim();
        errorMsg.userMessageId = newUserMsg.id;
      }
      setMessages((prev) => [
        ...prev.filter(msg => msg.id !== newBotMsgId || msg.content.trim()),
        errorMsg
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage(input);
  };

  const [openCitations, setOpenCitations] = useState({});
  const toggleCitation = (msgId) => {
    setOpenCitations(prev => ({
      ...prev,
      [msgId]: !prev[msgId]
    }));
  };

  return (
    <>
      {/* Floating Button */}
      <button 
        className={`${styles.floatingBtn} ${isOpen ? styles.hidden : ''}`}
        onClick={toggleChat}
        aria-label="Mở chat"
      >
        <i className="fa-solid fa-message"></i>
      </button>

      {/* Chat Panel */}
      <div className={`${styles.chatPanel} ${isOpen ? styles.open : ''}`}>
        <div className={styles.chatHeader}>
          <div className={styles.logo}>
            <i className="fa-solid fa-graduation-cap"></i>
          </div>
          <div className={styles.headerInfo}>
            <h4>AdmitAI</h4>
            <p>Trợ lý Tuyển sinh PTIT</p>
          </div>
          <button className={styles.closeBtn} onClick={toggleChat} aria-label="Đóng chat">
            <i className="fa-solid fa-times"></i>
          </button>
        </div>

        <div className={styles.chatArea} ref={chatAreaRef}>
          {messages.map((msg) => {
            const sources = citationGroups(msg.citations);
            return (
            <div key={msg.id} className={`${styles.message} ${msg.role === 'bot' ? styles.botMessage : styles.userMessage} ${msg.isError ? styles.error : ''}`}>
              {msg.role === 'bot' && (
                <div className={styles.messageAvatar}>
                  <i className="fa-solid fa-robot"></i>
                </div>
              )}
              <div className={styles.messageContent}>
                {msg.answerMode === 'evidence' && (
                  <p role="status">Nội dung dưới đây được trích xuất và đối chiếu trực tiếp từ tài liệu tuyển sinh.</p>
                )}
                {msg.role === 'bot' ? (
                  <div className={styles.markdown}>
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={markdownComponents}
                      skipHtml
                    >
                      {msg.content}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <div className={styles.plainText}>{msg.content}</div>
                )}
                
                {sources.length > 0 && (
                  <div className={`${styles.citationsContainer} ${openCitations[msg.id] ? styles.openCitation : ''}`}>
                    <button className={styles.citationsToggle} onClick={() => toggleCitation(msg.id)}>
                      <i className="fa-solid fa-book-open"></i> Nguồn tham khảo ({sources.length})
                      <i className={`fa-solid fa-chevron-down ${styles.toggleIcon}`}></i>
                    </button>
                    
                    <div className={styles.citationsList}>
                      {sources.map(({ source: c, markers, members }, i) => {
                        const label = c.title || c.source || 'Tài liệu tuyển sinh PTIT';
                        
                        return (
                          <div key={i} className={styles.citationItem}>
                            {c.source_url ? (
                              <a href={c.source_url} target="_blank" rel="noopener noreferrer">
                                [{markers.join(', ')}] {label}
                              </a>
                            ) : (
                              <span>[{markers.join(', ')}] {label}</span>
                            )}
                            {members.map((evidence) => {
                              const location = [
                                evidence.page ? `Trang ${evidence.page}` : '',
                                evidence.section && evidence.section !== `Trang ${evidence.page}` ? evidence.section : ''
                              ].filter(Boolean).join(' · ');
                              return (
                                <div key={evidence.key}>
                                  {location && <div className={styles.citationLocation}>{location}</div>}
                                  {evidence.snippet && (
                                    <div className={styles.citationSnippet}>
                                      <ReactMarkdown
                                        remarkPlugins={[remarkGfm]}
                                        components={markdownComponents}
                                        disallowedElements={['img']}
                                        skipHtml
                                      >
                                        {evidence.snippet}
                                      </ReactMarkdown>
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
                {msg.isError && msg.retryQuestion && (
                  <button type="button" disabled={isLoading} onClick={() => sendMessage(msg.retryQuestion, msg.userMessageId)}>
                    Gửi lại câu hỏi
                  </button>
                )}
                
                {msg.role === 'bot' && !msg.isError && msg.id !== 'welcome' && (
                  <div className={styles.feedbackActions}>
                    <button className={styles.feedbackBtn} title="Hữu ích"><i className="fa-regular fa-thumbs-up"></i></button>
                    <button className={styles.feedbackBtn} title="Không hữu ích"><i className="fa-regular fa-thumbs-down"></i></button>
                  </div>
                )}
              </div>
            </div>
            );
          })}
          
          {showSuggestions && (
            <div className={styles.suggestedQuestions}>
              <button disabled={serviceStatus !== 'available'} className={styles.suggestionBtn} onClick={() => sendMessage('PTIT có những phương thức tuyển sinh nào trong năm 2026?')}>Phương thức tuyển sinh 2026?</button>
              <button disabled={serviceStatus !== 'available'} className={styles.suggestionBtn} onClick={() => sendMessage('Học phí PTIT năm 2026 là bao nhiêu?')}>Học phí năm 2026?</button>
              <button disabled={serviceStatus !== 'available'} className={styles.suggestionBtn} onClick={() => sendMessage('Điểm chuẩn ngành Công nghệ thông tin năm 2025 là bao nhiêu?')}>Điểm chuẩn CNTT 2025?</button>
            </div>
          )}

          {(serviceStatus === 'checking' || serviceStatus === 'unavailable') && (
            <div className={styles.serviceStatus} role="status">
              <span>
                {serviceStatus === 'checking'
                  ? 'Đang kiểm tra dịch vụ tư vấn...'
                  : 'Dịch vụ tư vấn đang tạm thời chưa sẵn sàng.'}
              </span>
              {serviceStatus === 'unavailable' && (
                <button type="button" onClick={checkService} className={styles.retryBtn}>
                  Thử lại
                </button>
              )}
            </div>
          )}

          {isLoading && (
            <div aria-label="Đang trả lời" className={`${styles.message} ${styles.botMessage} ${styles.typingIndicator}`}>
              <div className={styles.messageAvatar}>
                <i className="fa-solid fa-robot"></i>
              </div>
              <div className={styles.messageContent}>
                <div className={styles.typingDots}>
                  <span></span><span></span><span></span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className={styles.chatInputArea}>
          <form className={styles.chatForm} onSubmit={handleSubmit}>
            <input 
              type="text" 
              className={styles.messageInput}
              placeholder="Nhập câu hỏi của bạn..." 
              autoComplete="off"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading || serviceStatus === 'checking' || serviceStatus === 'unavailable'}
            />
            <button
              type="submit"
              className={styles.sendBtn}
              aria-label="Gửi câu hỏi"
              disabled={
                isLoading
                || !input.trim()
                || serviceStatus === 'checking'
                || serviceStatus === 'unavailable'
              }
            >
              <i className="fa-solid fa-paper-plane"></i>
            </button>
          </form>
        </div>
      </div>
    </>
  );
}
