// ── State ────────────────────────────────────────────────────
const state = {
    conversations: [],
    currentConversation: null,
    conversationHistory: [],
    isLoading: false
};

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    newConversation();
});

// ── Conversations ─────────────────────────────────────────────
function newConversation() {
    const id = Date.now();
    const conv = { id, title: 'New Conversation', messages: [] };
    state.conversations.unshift(conv);
    state.currentConversation = conv;
    state.conversationHistory = [];
    renderConversations();
    clearChat();
}

function renderConversations() {
    const list = document.getElementById('conversations-list');
    list.innerHTML = '';
    state.conversations.forEach(conv => {
        const item = document.createElement('div');
        item.className = `conversation-item ${conv.id === state.currentConversation?.id ? 'active' : ''}`;
        item.innerHTML = `
            <div class="conversation-title" id="title-${conv.id}">${conv.title}</div>
            <button class="edit-title-btn" onclick="editTitle(event, ${conv.id})" title="Rename">✏️</button>
        `;
        item.addEventListener('click', (e) => {
            if (!e.target.classList.contains('edit-title-btn')) {
                switchConversation(conv.id);
            }
        });
        list.appendChild(item);
    });
}

function switchConversation(id) {
    const conv = state.conversations.find(c => c.id === id);
    if (!conv) return;
    state.currentConversation = conv;
    state.conversationHistory = conv.messages
        .filter(m => m.role === 'user')
        .map((m, i) => ({
            query: m.content,
            answer: conv.messages[i * 2 + 1]?.content || ''
        }));
    renderConversations();
    replayMessages(conv.messages);
}

function replayMessages(messages) {
    clearChat();
    if (messages.length === 0) return;
    showChat();
    messages.forEach(msg => {
        if (msg.role === 'user') {
            appendUserMessage(msg.content, msg.rewrittenQuery);
        } else {
            appendAIMessage(msg.content, msg.followUps || []);
        }
    });
    scrollToBottom();
}

function editTitle(event, id) {
    event.stopPropagation();
    const titleEl = document.getElementById(`title-${id}`);
    const conv = state.conversations.find(c => c.id === id);
    if (!conv) return;
    const input = document.createElement('input');
    input.value = conv.title;
    input.onclick = e => e.stopPropagation();
    input.onblur = () => {
        conv.title = input.value || 'New Conversation';
        renderConversations();
    };
    input.onkeydown = (e) => {
        if (e.key === 'Enter') input.blur();
        if (e.key === 'Escape') { input.value = conv.title; input.blur(); }
    };
    titleEl.innerHTML = '';
    titleEl.appendChild(input);
    input.focus();
    input.select();
}

// ── Chat ──────────────────────────────────────────────────────
function clearChat() {
    document.getElementById('messages-container').innerHTML = '';
    document.getElementById('welcome-screen').style.display = 'flex';
    document.getElementById('messages-container').style.display = 'none';
}

function showChat() {
    document.getElementById('welcome-screen').style.display = 'none';
    document.getElementById('messages-container').style.display = 'flex';
}

// ── Send ──────────────────────────────────────────────────────
function sendSample(text) {
    document.getElementById('query-input').value = text;
    sendQuery();
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
    }
}

function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

async function sendQuery() {
    const input = document.getElementById('query-input');
    const query = input.value.trim();
    if (!query || state.isLoading) return;

    input.value = '';
    input.style.height = 'auto';
    state.isLoading = true;
    document.getElementById('send-btn').disabled = true;

    showChat();
    appendUserMessage(query, null);

    if (state.currentConversation) {
        if (state.currentConversation.title === 'New Conversation') {
            state.currentConversation.title = query.slice(0, 40) + (query.length > 40 ? '...' : '');
            renderConversations();
        }
        state.currentConversation.messages.push({ role: 'user', content: query });
    }

    showTyping('Analysing your question...');

    try {
        const typingStages = [
            'Analysing your question...',
            'Searching documents...',
            'Reranking results...',
            'Generating answer...'
        ];
        let stageIdx = 0;
        const stageInterval = setInterval(() => {
            stageIdx = Math.min(stageIdx + 1, typingStages.length - 1);
            updateTypingText(typingStages[stageIdx]);
        }, 3000);

        const response = await fetch('/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                conversation_history: state.conversationHistory
            })
        });

        clearInterval(stageInterval);

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Request failed');
        }

        const data = await response.json();
        hideTyping();

        const answer = data.answer || 'No answer returned.';
        const followUps = data.follow_ups || [];

        appendAIMessage(answer, followUps);

        if (state.currentConversation) {
            state.currentConversation.messages.push({
                role: 'ai',
                content: answer,
                followUps: followUps
            });
        }

        // Update conversation history for multi-turn
        state.conversationHistory.push({
            query,
            answer: answer.slice(0, 300)
        });
        if (state.conversationHistory.length > 5) {
            state.conversationHistory.shift();
        }

        // Update rewrite badge
        if (data.rewritten_query && data.rewritten_query !== query) {
            updateLastUserRewrite(data.rewritten_query);
        }

    } catch (err) {
        hideTyping();
        appendAIMessage(`⚠️ ${err.message || 'Something went wrong. Please try again.'}`, []);
    } finally {
        state.isLoading = false;
        document.getElementById('send-btn').disabled = false;
        scrollToBottom();
    }
}

// ── Messages ──────────────────────────────────────────────────
function appendUserMessage(text, rewrittenQuery) {
    const container = document.getElementById('messages-container');
    const div = document.createElement('div');
    div.className = 'message-user';
    div.innerHTML = `
        <div>
            <div class="bubble">${escapeHtml(text)}</div>
            <div class="rewrite-badge ${rewrittenQuery ? '' : 'hidden'}">
                🔄 <span title="${escapeHtml(rewrittenQuery || '')}">${escapeHtml((rewrittenQuery || '').slice(0, 60))}${(rewrittenQuery || '').length > 60 ? '...' : ''}</span>
            </div>
        </div>
    `;
    container.appendChild(div);
    scrollToBottom();
}

function updateLastUserRewrite(rewrittenQuery) {
    const messages = document.querySelectorAll('.message-user');
    const last = messages[messages.length - 1];
    if (!last) return;
    const badge = last.querySelector('.rewrite-badge');
    if (badge) {
        badge.classList.remove('hidden');
        badge.innerHTML = `🔄 <span title="${escapeHtml(rewrittenQuery)}">${escapeHtml(rewrittenQuery.slice(0, 60))}${rewrittenQuery.length > 60 ? '...' : ''}</span>`;
    }
}

function appendAIMessage(text, followUps) {
    const container = document.getElementById('messages-container');
    const div = document.createElement('div');
    div.className = 'message-ai';

    // Build follow-ups HTML
    let followUpsHTML = '';
    if (followUps && followUps.length > 0) {
        followUpsHTML = `
            <div class="followup-pills">
                <div class="followup-label">Suggested follow-ups</div>
                ${followUps.map(q => `
                    <button class="followup-pill" onclick="sendSample('${escapeHtml(q)}')">${escapeHtml(q)}</button>
                `).join('')}
            </div>
        `;
    }

    div.innerHTML = `
        <div class="ai-avatar">AI</div>
        <div class="bubble">
            ${formatAnswer(text)}
            ${followUpsHTML}
        </div>
    `;
    container.appendChild(div);
    scrollToBottom();
}

function formatAnswer(text) {
    // Remove sources line — not shown in simplified UI
    let cleaned = text.replace(/\n*Sources:.*$/s, '').trim();
    cleaned = cleaned.replace(/\[Source \d+\]:?\s*/g, '');
    cleaned = cleaned.replace(/According to \[Source \d+\]:?\s*/gi, '');

    return cleaned
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>')
        .replace(/\* (.*?)(?=<br>|<\/p>)/g, '• $1');
}

// ── Typing ────────────────────────────────────────────────────
function showTyping(text) {
    const el = document.getElementById('typing-indicator');
    el.classList.remove('hidden');
    updateTypingText(text);
    scrollToBottom();
}

function updateTypingText(text) {
    document.getElementById('typing-text').textContent = text;
}

function hideTyping() {
    document.getElementById('typing-indicator').classList.add('hidden');
}

// ── Helpers ───────────────────────────────────────────────────
function scrollToBottom() {
    const container = document.getElementById('messages-container');
    setTimeout(() => { container.scrollTop = container.scrollHeight; }, 50);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(text || ''));
    return div.innerHTML;
}