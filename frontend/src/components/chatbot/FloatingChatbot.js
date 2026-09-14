'use client';
import { useState, useRef, useEffect } from 'react';
import styles from './FloatingChatbot.module.css';

const API_URL = '/api/chat';

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
  const [sessionId, setSessionId] = useState(null);
  const chatAreaRef = useRef(null);

  useEffect(() => {
    // Load session ID from localStorage on mount
    const savedSessionId = localStorage.getItem('admitai_session_id');
    if (savedSessionId) {
      setSessionId(savedSessionId);
    }
  }, []);

  useEffect(() => {
    // Scroll to bottom when messages change
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const toggleChat = () => setIsOpen(!isOpen);

  const sendMessage = async (text) => {
    if (!text || !text.trim() || isLoading) return;
    
    setShowSuggestions(false);
    
    // Add user message
    const newUserMsg = { id: Date.now(), role: 'user', content: text.trim() };
    setMessages((prev) => [...prev, newUserMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const payload = { message: text.trim() };
      if (sessionId) {
        payload.session_id = sessionId;
      }
      
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id);
        localStorage.setItem('admitai_session_id', data.session_id);
      }
      
      const newBotMsg = {
        id: Date.now() + 1,
        role: 'bot',
        content: data.answer,
        citations: data.citations || []
      };
      
      setMessages((prev) => [...prev, newBotMsg]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMsg = {
        id: Date.now() + 1,
        role: 'bot',
        content: 'Xin lỗi, kết nối đến máy chủ bị lỗi. Vui lòng thử lại sau.',
        isError: true,
        citations: []
      };
      setMessages((prev) => [...prev, errorMsg]);
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
          {messages.map((msg) => (
            <div key={msg.id} className={`${styles.message} ${msg.role === 'bot' ? styles.botMessage : styles.userMessage} ${msg.isError ? styles.error : ''}`}>
              {msg.role === 'bot' && (
                <div className={styles.messageAvatar}>
                  <i className="fa-solid fa-robot"></i>
                </div>
              )}
              <div className={styles.messageContent}>
                <div className={styles.text}>{msg.content}</div>
                
                {msg.citations && msg.citations.length > 0 && (
                  <div className={`${styles.citationsContainer} ${openCitations[msg.id] ? styles.openCitation : ''}`}>
                    <button className={styles.citationsToggle} onClick={() => toggleCitation(msg.id)}>
                      <i className="fa-solid fa-book-open"></i> {msg.citations.length} Nguồn tham khảo
                      <i className={`fa-solid fa-chevron-down ${styles.toggleIcon}`}></i>
                    </button>
                    
                    <div className={styles.citationsList}>
                      {msg.citations.map((c, i) => {
                        const label = c.title || c.source || 'Tài liệu tuyển sinh PTIT';
                        const location = [
                          c.page ? `Trang ${c.page}` : '',
                          c.section && c.section !== `Trang ${c.page}` ? c.section : ''
                        ].filter(Boolean).join(' · ');
                        
                        return (
                          <div key={i} className={styles.citationItem}>
                            {c.source_url ? (
                              <a href={c.source_url} target="_blank" rel="noopener noreferrer">
                                [{i + 1}] {label}
                              </a>
                            ) : (
                              <span>[{i + 1}] {label}</span>
                            )}
                            {location && <div className={styles.citationLocation}>{location}</div>}
                            {c.snippet && <div className={styles.citationSnippet}>{c.snippet}</div>}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
                
                {msg.role === 'bot' && !msg.isError && msg.id !== 'welcome' && (
                  <div className={styles.feedbackActions}>
                    <button className={styles.feedbackBtn} title="Hữu ích"><i className="fa-regular fa-thumbs-up"></i></button>
                    <button className={styles.feedbackBtn} title="Không hữu ích"><i className="fa-regular fa-thumbs-down"></i></button>
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {showSuggestions && (
            <div className={styles.suggestedQuestions}>
              <button className={styles.suggestionBtn} onClick={() => sendMessage('PTIT có những phương thức tuyển sinh nào trong năm 2026?')}>Phương thức tuyển sinh 2026?</button>
              <button className={styles.suggestionBtn} onClick={() => sendMessage('Học phí PTIT năm 2026 là bao nhiêu?')}>Học phí năm 2026?</button>
              <button className={styles.suggestionBtn} onClick={() => sendMessage('Điểm chuẩn ngành Công nghệ thông tin năm 2025 là bao nhiêu?')}>Điểm chuẩn CNTT 2025?</button>
            </div>
          )}

          {isLoading && (
            <div className={`${styles.message} ${styles.botMessage} ${styles.typingIndicator}`}>
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
              disabled={isLoading}
            />
            <button type="submit" className={styles.sendBtn} disabled={isLoading || !input.trim()}>
              <i className="fa-solid fa-paper-plane"></i>
            </button>
          </form>
        </div>
      </div>
    </>
  );
}
