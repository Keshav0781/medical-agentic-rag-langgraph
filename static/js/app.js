// ── State ────────────────────────────────────────────────────
const state = {
    mode: 'search',
    conversations: [],
    currentConversation: null,
    conversationHistory: [],
    isLoading: false
};

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    newConversation();
});

// ── Mode ─────────────────────────────────────────────────────
function setMode(mode) {
    state.mode = mode;
    document.getElementById('btn-search').classList.toggle('active', mode === 'search');
    document.getElementById('btn-summarise').classList.toggle('active', mode === 'summarise');

    const placeholder = mode === 'search'
        ? 'Ask about Siemens Healthineers R&D documents...'
        : 'e.g. Summarise the 2023 Annual Report...';
    document.getElementById('query-input').placeholder = placeholder;
}

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
    
    // Show chat area — hide welcome screen
    showChat();
    
    messages.forEach(msg => {
        if (msg.role === 'user') {
            appendUserMessage(msg.content, msg.rewrittenQuery);
        } else {
            appendAIMessage(msg.content);
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
        if (e.key === 'Escape') {
            input.value = conv.title;
            input.blur();
        }
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
    hideDetails();
}

function hideDetails() {
    document.getElementById('pipeline-stats').style.display = 'none';
    document.getElementById('rewrite-section').style.display = 'none';
    document.getElementById('sources-section').style.display = 'none';
    document.getElementById('followups-section').style.display = 'none';
    document.getElementById('details-empty').style.display = 'flex';
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

    // Save to conversation
    if (state.currentConversation) {
        if (state.currentConversation.title === 'New Conversation') {
            state.currentConversation.title = query.slice(0, 40) + (query.length > 40 ? '...' : '');
            renderConversations();
        }
        state.currentConversation.messages.push({ role: 'user', content: query });
    }

    showTyping('Analysing your question...');

    try {
        const endpoint = state.mode === 'summarise' ? '/summarise' : '/search';
        const body = state.mode === 'summarise'
            ? { query }
            : { query, conversation_history: state.conversationHistory };

        // Update typing stages
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

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        clearInterval(stageInterval);

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Request failed');
        }

        const data = await response.json();
        hideTyping();

        const answer = data.answer || 'No answer returned.';
        appendAIMessage(answer);

        // Save AI message
        if (state.currentConversation) {
            state.currentConversation.messages.push({ role: 'ai', content: answer });
        }

        // Update conversation history for multi-turn
        if (state.mode === 'search') {
            state.conversationHistory.push({ query, answer: answer.slice(0, 300) });
            if (state.conversationHistory.length > 5) {
                state.conversationHistory.shift();
            }
        }

        // Update details panel
        updateDetails(data);

        // Update user message with rewrite badge
        if (data.rewritten_query && data.rewritten_query !== query) {
            updateLastUserRewrite(data.rewritten_query);
        }

    } catch (err) {
        hideTyping();
        appendAIMessage(`⚠️ ${err.message || 'Something went wrong. Please try again.'}`);
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
    div.dataset.hasRewrite = 'false';
    div.innerHTML = `
        <div>
            <div class="bubble">${escapeHtml(text)}</div>
            ${rewrittenQuery ? `
            <div class="rewrite-badge">
                🔄 <span title="${escapeHtml(rewrittenQuery)}">${escapeHtml(rewrittenQuery.slice(0, 60))}${rewrittenQuery.length > 60 ? '...' : ''}</span>
            </div>` : '<div class="rewrite-badge hidden"></div>'}
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

function appendAIMessage(text) {
    const container = document.getElementById('messages-container');
    const div = document.createElement('div');
    div.className = 'message-ai';
    div.innerHTML = `
        <div class="ai-avatar">🏥</div>
        <div class="bubble">${formatAnswer(text)}</div>
    `;
    container.appendChild(div);
    scrollToBottom();
}

function formatAnswer(text) {
    // Remove sources line — shown in right panel only
    let cleaned = text.replace(/\n*Sources:.*$/s, '').trim();
    // Remove [Source X] inline references
    cleaned = cleaned.replace(/\[Source \d+\]:?\s*/g, '');
    // Remove "According to [Source X]" patterns
    cleaned = cleaned.replace(/According to \[Source \d+\]:?\s*/gi, '');

    return cleaned
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>')
        .replace(/\* (.*?)(?=\n|$)/g, '• $1');
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

// ── Details Panel ─────────────────────────────────────────────
function updateDetails(data) {
    document.getElementById('details-empty').style.display = 'none';

    // Always clear follow-ups first — repopulate only if present
    document.getElementById('followups-section').style.display = 'none';
    document.getElementById('followups-list').innerHTML = '';

    // Pipeline stats
    if (data.route !== undefined) {
        document.getElementById('pipeline-stats').style.display = 'block';
        document.getElementById('stat-route').textContent = data.route?.toUpperCase() || '—';
        document.getElementById('stat-retrieved').textContent = data.chunks_retrieved ?? '—';
        document.getElementById('stat-reranked').textContent = data.chunks_reranked ?? '—';
    }

    // Query rewrite
    if (data.rewritten_query && data.rewritten_query !== data.query) {
        document.getElementById('rewrite-section').style.display = 'block';
        document.getElementById('rewrite-text').textContent = data.rewritten_query;
    }

    // Sources — parse from answer text
    const sourcesMatch = data.answer?.match(/Sources: (.+?)(?:\n|$)/);
    if (sourcesMatch) {
        const sources = sourcesMatch[1].split(', ').map(s => s.trim());
        if (sources.length > 0) {
            document.getElementById('sources-section').style.display = 'block';
            const list = document.getElementById('sources-list');
            list.innerHTML = '';
            sources.forEach(source => {
                const pageMatch = source.match(/\(Page (\d+)\)/);
                const name = source.replace(/\(Page \d+\)/, '').trim();
                const card = document.createElement('div');
                card.className = 'source-card';
                card.innerHTML = `
                    <div class="source-name" title="${escapeHtml(name)}">📄 ${escapeHtml(name.replace('siemens-healthineers-ir-', '').replace('.pdf', ''))}</div>
                    ${pageMatch ? `<div class="source-meta">Page ${pageMatch[1]}</div>` : ''}
                `;
                list.appendChild(card);
            });
        }
    }

    // Follow-ups
    if (data.follow_ups && data.follow_ups.length > 0) {
        document.getElementById('followups-section').style.display = 'block';
        const list = document.getElementById('followups-list');
        list.innerHTML = '';
        data.follow_ups.forEach(q => {
            const btn = document.createElement('button');
            btn.className = 'followup-btn';
            btn.textContent = q;
            btn.onclick = () => {
                document.getElementById('query-input').value = q;
                sendQuery();
            };
            list.appendChild(btn);
        });
    }
}

// ── Helpers ───────────────────────────────────────────────────
function scrollToBottom() {
    const container = document.getElementById('messages-container');
    setTimeout(() => {
        container.scrollTop = container.scrollHeight;
    }, 50);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(text || ''));
    return div.innerHTML;
}