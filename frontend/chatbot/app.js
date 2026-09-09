const API_URL = 'http://localhost:8000/api/chat';

class ChatApp {
    constructor() {
        this.sessionId = localStorage.getItem('admitai_session_id');
        this.isWaiting = false;
        
        // DOM Elements
        this.chatArea = document.getElementById('chatArea');
        this.chatForm = document.getElementById('chatForm');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.suggestedQuestions = document.getElementById('suggestedQuestions');
        
        // Templates
        this.userMsgTpl = document.getElementById('userMessageTemplate');
        this.botMsgTpl = document.getElementById('botMessageTemplate');
        this.typingTpl = document.getElementById('typingTemplate');
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Focus input on load
        this.messageInput.focus();
    }
    
    scrollToBottom() {
        this.chatArea.scrollTop = this.chatArea.scrollHeight;
    }
    
    setLoading(isLoading) {
        this.isWaiting = isLoading;
        this.messageInput.disabled = isLoading;
        this.sendBtn.disabled = isLoading;
        
        if (isLoading) {
            // Add typing indicator
            const typingNode = this.typingTpl.content.cloneNode(true);
            this.chatArea.appendChild(typingNode);
            this.scrollToBottom();
        } else {
            // Remove typing indicator
            const indicator = document.getElementById('typingIndicator');
            if (indicator) {
                indicator.remove();
            }
            this.messageInput.focus();
        }
    }
    
    hideSuggestedQuestions() {
        if (this.suggestedQuestions && this.suggestedQuestions.style.display !== 'none') {
            this.suggestedQuestions.style.display = 'none';
        }
    }
    
    appendUserMessage(text) {
        const node = this.userMsgTpl.content.cloneNode(true);
        node.querySelector('.text').textContent = text;
        this.chatArea.appendChild(node);
        this.scrollToBottom();
    }
    
    appendBotMessage(text, citations = [], isError = false) {
        const node = this.botMsgTpl.content.cloneNode(true);
        const msgDiv = node.querySelector('.bot-message');
        
        if (isError) {
            msgDiv.classList.add('error');
        }
        
        // Use innerHTML for simple formatting (replace newlines with <br>)
        // In a real app, use a markdown parser
        node.querySelector('.text').innerHTML = text.replace(/\n/g, '<br>');
        
        // Handle citations
        if (citations && citations.length > 0) {
            const citContainer = node.querySelector('.citations-container');
            const citCount = node.querySelector('.citation-count');
            const citList = node.querySelector('.citations-list');
            const toggleBtn = node.querySelector('.citations-toggle');
            
            citContainer.style.display = 'block';
            citCount.textContent = citations.length;
            
            citations.forEach(c => {
                const p = document.createElement('p');
                p.className = 'citation-item';
                p.textContent = `• ${c.source || 'Tài liệu tuyển sinh'}`;
                citList.appendChild(p);
            });
            
            toggleBtn.addEventListener('click', (e) => {
                const container = e.currentTarget.closest('.citations-container');
                container.classList.toggle('open');
            });
        }
        
        // Handle feedback
        const positiveBtn = node.querySelector('.positive');
        const negativeBtn = node.querySelector('.negative');
        
        if (!isError) {
            positiveBtn.addEventListener('click', function() {
                this.classList.add('fa-solid');
                this.style.color = 'var(--primary-color)';
                negativeBtn.classList.remove('fa-solid');
                negativeBtn.style.color = '';
                // In a real app: send feedback API
            });
            
            negativeBtn.addEventListener('click', function() {
                this.classList.add('fa-solid');
                this.style.color = 'var(--primary-color)';
                positiveBtn.classList.remove('fa-solid');
                positiveBtn.style.color = '';
                // In a real app: send feedback API
            });
        } else {
            node.querySelector('.feedback-actions').style.display = 'none';
        }
        
        this.chatArea.appendChild(node);
        this.scrollToBottom();
    }
    
    async sendMessage(text) {
        if (!text || !text.trim() || this.isWaiting) return;
        
        text = text.trim();
        this.hideSuggestedQuestions();
        this.appendUserMessage(text);
        this.messageInput.value = '';
        this.setLoading(true);
        
        try {
            const payload = {
                message: text
            };
            if (this.sessionId) {
                payload.session_id = this.sessionId;
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
            
            // Save session_id if new
            if (data.session_id && data.session_id !== this.sessionId) {
                this.sessionId = data.session_id;
                localStorage.setItem('admitai_session_id', this.sessionId);
            }
            
            this.setLoading(false);
            this.appendBotMessage(data.answer, data.citations);
            
        } catch (error) {
            console.error('Chat error:', error);
            this.setLoading(false);
            this.appendBotMessage(
                'Xin lỗi, kết nối đến máy chủ bị lỗi. Vui lòng thử lại sau.', 
                [], 
                true
            );
        }
    }
    
    handleSubmit(event) {
        event.preventDefault();
        const text = this.messageInput.value;
        this.sendMessage(text);
    }
    
    sendSuggestion(text) {
        this.sendMessage(text);
    }
}

// Initialize App
const app = new ChatApp();
