document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatContainer = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const btnSend = document.getElementById('btn-send');
    const welcomeCard = document.getElementById('welcome-card');
    const btnClear = document.getElementById('btn-clear');
    const btnSettings = document.getElementById('btn-settings');
    const modalSettings = document.getElementById('settings-modal');
    const btnCloseModal = document.getElementById('btn-close-modal');
    const btnSaveSettings = document.getElementById('btn-save-settings');
    const apiUrlInput = document.getElementById('api-url-input');
    const statusLabel = document.getElementById('status-label');

    // Storage for API URL (defaults to current origin or localhost:8000)
    let apiUrl = localStorage.getItem('custom_ai_api_url') || (window.location.origin.includes('http') ? window.location.origin : 'http://localhost:8000');
    apiUrlInput.value = apiUrl;

    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
    });

    // Send on Enter (unless Shift is held)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    btnSend.addEventListener('click', sendMessage);

    // Quick prompts
    document.querySelectorAll('.prompt-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            userInput.value = chip.getAttribute('data-text');
            sendMessage();
        });
    });

    // Clear chat
    btnClear.addEventListener('click', () => {
        chatContainer.innerHTML = '';
        if (welcomeCard) {
            chatContainer.appendChild(welcomeCard);
        }
    });

    // Settings Modal
    btnSettings.addEventListener('click', () => {
        modalSettings.style.display = 'flex';
    });

    btnCloseModal.addEventListener('click', () => {
        modalSettings.style.display = 'none';
    });

    btnSaveSettings.addEventListener('click', () => {
        const val = apiUrlInput.value.trim().replace(/\/$/, "");
        if (val) {
            apiUrl = val;
            localStorage.setItem('custom_ai_api_url', apiUrl);
            modalSettings.style.display = 'none';
            checkServerStatus();
        }
    });

    // Check Server Status
    async function checkServerStatus() {
        try {
            statusLabel.textContent = "جاري التحقق من السيرفر...";
            const res = await fetch(`${apiUrl}/health`);
            if (res.ok) {
                const data = await res.json();
                statusLabel.textContent = `متصل • ${data.device.toUpperCase()}`;
            } else {
                statusLabel.textContent = "السيرفر لا يستجيب";
            }
        } catch {
            statusLabel.textContent = "غير متصل بالسيرفر";
        }
    }

    checkServerStatus();

    // Send message logic
    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        // Hide welcome card once chat starts
        if (welcomeCard && welcomeCard.parentElement) {
            welcomeCard.remove();
        }

        // 1. Add User Message
        appendMessage('user', text);
        userInput.value = '';
        userInput.style.height = 'auto';

        // 2. Add Typing Indicator
        const typingIndicator = appendTypingIndicator();
        chatContainer.scrollTop = chatContainer.scrollHeight;

        try {
            const endpoint = `${apiUrl.replace(/\/$/, "")}/api/chat`;
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prompt: text,
                    max_tokens: 300,
                    temperature: 0.7
                })
            });

            typingIndicator.remove();

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error: ${response.status}`);
            }

            const data = await response.json();
            appendMessage('ai', data.response || "لم يتم استلام أي إجابة.");
        } catch (error) {
            typingIndicator.remove();
            appendMessage('ai', `⚠️ تعذر الاتصال بالنموذج: ${error.message}\nتأكد أن أمر 'python api.py' شغال وأن رابط الـ API صحيح في الإعدادات.`);
        }

        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function appendMessage(sender, content) {
        const row = document.createElement('div');
        row.className = `message-row ${sender}`;

        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar';
        avatar.textContent = sender === 'user' ? '👤' : '⚡';

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        
        // Format basic markdown / code snippets
        bubble.innerHTML = formatContent(content);

        row.appendChild(avatar);
        row.appendChild(bubble);

        chatContainer.appendChild(row);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return row;
    }

    function appendTypingIndicator() {
        const row = document.createElement('div');
        row.className = 'message-row ai';

        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar';
        avatar.textContent = '⚡';

        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        bubble.innerHTML = `
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
        `;

        row.appendChild(avatar);
        row.appendChild(bubble);
        chatContainer.appendChild(row);
        return row;
    }

    function formatContent(text) {
        // Escape HTML
        let escaped = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // Code blocks: ```python ... ```
        escaped = escaped.replace(/```([a-zA-Z]*)\n?([\s\S]*?)```/g, (match, lang, code) => {
            return `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`;
        });

        // Inline code: `code`
        escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');

        // New lines to <br>
        escaped = escaped.replace(/\n/g, '<br>');

        return escaped;
    }
});
