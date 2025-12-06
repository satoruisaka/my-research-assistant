// app.js - MRA v3 Frontend Logic

// State Management
const state = {
    currentSessionId: null,
    messages: [],
    context: [],
    settings: {
        model: 'mistral:latest',
        temperature: 0.7,
        maxTokens: 4000,
        topK: 20,
        searchScope: {
            referencePapers: true,
            myPapers: true,
            sessions: false,
            webCache: false
        },
        useWebSearch: false,
        useDistortion: false,
        useEnsemble: false,
        includeConversationContext: true,
        distortion: {
            mode: 'cucumb_er',
            tone: 'neutral',
            gain: 5
        }
    }
};

// Current exchange container (for collapsible chat)
let currentExchange = null;

// API Base URL
const API_BASE = window.location.origin;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkServiceStatus();
    loadSessions();
    
    // Setup welcome exchange toggle
    const welcomeHeader = document.querySelector('#welcome-exchange .exchange-header');
    if (welcomeHeader) {
        welcomeHeader.addEventListener('click', () => toggleExchange('welcome-exchange'));
    }
    
    // Auto-resize textarea
    const textarea = document.getElementById('user-input');
    textarea.addEventListener('input', () => {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    });
});

// Event Listeners
function initializeEventListeners() {
    // Quick search
    const quickSearchBtn = document.getElementById('quick-search-btn');
    const quickSearchInput = document.getElementById('quick-search');
    
    quickSearchBtn.addEventListener('click', () => {
        const query = quickSearchInput.value.trim();
        if (query) {
            document.getElementById('user-input').value = query;
            sendMessage();
            quickSearchInput.value = '';
        }
    });
    
    quickSearchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            quickSearchBtn.click();
        }
    });
    
    // Send message
    document.getElementById('send-btn').addEventListener('click', sendMessage);
    document.getElementById('user-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Settings - Model
    document.getElementById('model-select').addEventListener('change', (e) => {
        state.settings.model = e.target.value;
    });
    
    // Settings - Sliders
    document.getElementById('temperature').addEventListener('input', (e) => {
        state.settings.temperature = parseFloat(e.target.value);
        document.getElementById('temp-value').textContent = e.target.value;
    });
    
    document.getElementById('max-tokens').addEventListener('input', (e) => {
        state.settings.maxTokens = parseInt(e.target.value);
        document.getElementById('max-tokens-value').textContent = e.target.value;
    });
    
    document.getElementById('top-k').addEventListener('input', (e) => {
        state.settings.topK = parseInt(e.target.value);
        document.getElementById('top-k-value').textContent = e.target.value;
    });
    
    document.getElementById('distortion-gain').addEventListener('input', (e) => {
        state.settings.distortion.gain = parseInt(e.target.value);
        document.getElementById('gain-value').textContent = e.target.value;
    });
    
    // Settings - Checkboxes
    document.getElementById('scope-reference').addEventListener('change', (e) => {
        state.settings.searchScope.referencePapers = e.target.checked;
    });
    
    document.getElementById('scope-authored').addEventListener('change', (e) => {
        state.settings.searchScope.myPapers = e.target.checked;
    });
    
    document.getElementById('scope-sessions').addEventListener('change', (e) => {
        state.settings.searchScope.sessions = e.target.checked;
    });
    
    document.getElementById('scope-web-cache').addEventListener('change', (e) => {
        state.settings.searchScope.webCache = e.target.checked;
    });
    
    document.getElementById('enable-web-search').addEventListener('change', (e) => {
        state.settings.useWebSearch = e.target.checked;
    });
    
    document.getElementById('enable-distortion').addEventListener('change', (e) => {
        state.settings.useDistortion = e.target.checked;
        const controls = document.getElementById('distortion-controls');
        controls.classList.toggle('active', e.target.checked);
    });
    
    document.getElementById('ensemble-mode').addEventListener('change', (e) => {
        state.settings.useEnsemble = e.target.checked;
    });
    
    document.getElementById('conversation-context').addEventListener('change', (e) => {
        state.settings.includeConversationContext = e.target.checked;
    });
    
    // Distortion settings
    document.getElementById('distortion-mode').addEventListener('change', (e) => {
        state.settings.distortion.mode = e.target.value;
    });
    
    document.getElementById('distortion-tone').addEventListener('change', (e) => {
        state.settings.distortion.tone = e.target.value;
    });
    
    // Panel tabs (simplified - only Sessions now)
    document.querySelectorAll('.panel-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            // No-op for now since we only have one tab
        });
    });
    
    // New session
    document.getElementById('new-session-btn').addEventListener('click', createNewSession);
}

// Switch panel tabs
function switchTab(tabName) {
    document.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel-content').forEach(c => c.classList.remove('active'));
    
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`${tabName}-panel`).classList.add('active');
}

// Check service status
async function checkServiceStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        
        console.log('Health check response:', data);  // DEBUG
        
        // Update Ollama status and populate models
        const ollamaDot = document.getElementById('ollama-status');
        const modelSelect = document.getElementById('model-select');
        
        console.log('Services data:', data.services);  // DEBUG
        
        if (data.services && data.services.ollama && data.services.ollama.status === 'online') {
            ollamaDot.classList.remove('offline');
            ollamaDot.classList.add('online');
            
            // Populate model dropdown dynamically
            if (data.services.ollama.models && data.services.ollama.models.length > 0) {
                modelSelect.disabled = false;
                
                // Save current selection
                const currentSelection = modelSelect.value || state.settings.model;
                
                // Rebuild options
                modelSelect.innerHTML = data.services.ollama.models.map(model => 
                    `<option value="${model}">${model}</option>`
                ).join('');
                
                // Restore previous selection if it still exists, otherwise use first model
                if (currentSelection && data.services.ollama.models.includes(currentSelection)) {
                    modelSelect.value = currentSelection;
                    state.settings.model = currentSelection;
                } else {
                    state.settings.model = data.services.ollama.models[0];
                    modelSelect.value = data.services.ollama.models[0];
                }
            } else {
                modelSelect.innerHTML = '<option value="">No models available</option>';
                modelSelect.disabled = true;
            }
        } else {
            ollamaDot.classList.add('offline');
            ollamaDot.classList.remove('online');
            modelSelect.disabled = true;
        }
        
        // Update TwistedPair status
        const tpDot = document.getElementById('twisted-status');
        
        if (data.services && data.services.twistedpair && data.services.twistedpair.status === 'online') {
            tpDot.classList.remove('offline');
            tpDot.classList.add('online');
        } else {
            tpDot.classList.add('offline');
            tpDot.classList.remove('online');
        }
        
        // Update index counts
        if (data.indices) {
            const countElements = {
                reference_papers: document.getElementById('count-reference'),
                my_papers: document.getElementById('count-authored'),
                sessions: document.getElementById('count-sessions'),
                web_cache: document.getElementById('count-web')
            };
            
            Object.keys(countElements).forEach(key => {
                if (countElements[key] && data.indices[key]) {
                    // Extract docs count from index stats object
                    const count = data.indices[key].docs || 0;
                    countElements[key].textContent = count.toLocaleString();
                }
            });
        }
    } catch (error) {
        console.error('Health check failed:', error);
        // Mark all as offline on error
        document.querySelectorAll('.status-dot').forEach(dot => dot.classList.add('offline'));
        document.getElementById('model-select').disabled = true;
    }
    
    // Check again in 60 seconds
    setTimeout(checkServiceStatus, 60000);
}

// Send message
async function sendMessage() {
    const textarea = document.getElementById('user-input');
    const message = textarea.value.trim();
    
    if (!message) return;
    
    // Clear input
    textarea.value = '';
    textarea.style.height = 'auto';
    
    // Disable send button
    const sendBtn = document.getElementById('send-btn');
    sendBtn.disabled = true;
    
    // Collapse all previous exchanges before adding new one
    collapseAllExchanges();
    
    // Start new exchange container
    startNewExchange();
    
    // Add user message to UI
    addMessage('user', message);
    
    // Show loading
    const loadingId = addMessage('assistant', '<div class="loading"><span></span><span></span><span></span></div>', true);
    
    try {
        // Prepare request
        const requestData = {
            session_id: state.currentSessionId || 'new',
            message: message,
            // Flatten settings to match server schema
            use_rag: true,
            use_web_search: state.settings.useWebSearch,
            use_distortion: state.settings.useDistortion,
            use_ensemble_distortion: state.settings.useEnsemble,
            include_conversation_context: state.settings.includeConversationContext,
            search_scope: {
                reference_papers: state.settings.searchScope.referencePapers,
                my_papers: state.settings.searchScope.myPapers,
                sessions: state.settings.searchScope.sessions,
                web_cache: state.settings.searchScope.webCache
            },
            model: state.settings.model,
            temperature: state.settings.temperature,
            max_tokens: state.settings.maxTokens,
            top_k_retrieval: state.settings.topK,
            distortion_mode: state.settings.distortion.mode,
            distortion_tone: state.settings.distortion.tone,
            distortion_gain: state.settings.distortion.gain
        };
        
        // Send request
        const response = await fetch(`${API_BASE}/api/chat/message`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update session ID
        if (data.session_id) {
            state.currentSessionId = data.session_id;
        }
        
        // Remove loading message
        removeMessage(loadingId);
        
        // Add assistant response
        addMessage('assistant', data.assistant_response, false, data.context_used);
        
        // If ensemble outputs available, display them
        if (data.ensemble_outputs && data.ensemble_outputs.length > 0) {
            displayEnsembleOutputs(data.ensemble_outputs);
        }
        
        // Update context panel
        if (data.context_used && data.context_used.length > 0) {
            updateContextPanel(data.context_used);
        }
        
    } catch (error) {
        console.error('Send message failed:', error);
        removeMessage(loadingId);
        addMessage('assistant', `❌ Error: ${error.message}`, true);
    } finally {
        sendBtn.disabled = false;
    }
}

// Add message to UI
function addMessage(role, content, isHtml = false, sources = null) {
    const messagesContainer = document.getElementById('chat-messages');
    const targetContainer = currentExchange ? currentExchange.querySelector('.exchange-content') : messagesContainer;
    const messageId = `msg-${Date.now()}`;
    
    // Update exchange header with user message preview
    if (role === 'user' && !isHtml && currentExchange) {
        updateExchangeHeader(content);
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.id = messageId;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (isHtml) {
        contentDiv.innerHTML = content;
    } else {
        contentDiv.textContent = content;
    }
    
    // Add sources if provided
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        sourcesDiv.innerHTML = '<strong>Sources:</strong><br>';

        sources.forEach((source, i) => {
            let scoreText = (typeof source.score === 'number') ? source.score.toFixed(2) : 'N/A';
            let label = source.title || source.source || `Source ${i+1}`;
            let tag;
            if (source.url) {
                tag = document.createElement('a');
                tag.href = source.url;
                tag.target = '_blank';
                tag.className = 'source-tag web-source-link';
                tag.textContent = `${label} (${scoreText})`;
            } else {
                tag = document.createElement('span');
                tag.className = 'source-tag';
                tag.textContent = `${label} (${scoreText})`;
            }
            sourcesDiv.appendChild(tag);
            // Show snippet if present
            if (source.snippet) {
                const snippetDiv = document.createElement('div');
                snippetDiv.className = 'source-snippet';
                snippetDiv.textContent = source.snippet;
                sourcesDiv.appendChild(snippetDiv);
            }
        });

        contentDiv.appendChild(sourcesDiv);
    }
    
    // Add timestamp
    const timestamp = document.createElement('div');
    timestamp.className = 'message-timestamp';
    timestamp.textContent = new Date().toLocaleTimeString();
    contentDiv.appendChild(timestamp);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    targetContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageId;
}

// Exchange Management

function startNewExchange() {
    const chatContainer = document.getElementById('chat-messages');
    
    // Create exchange container
    const exchangeDiv = document.createElement('div');
    exchangeDiv.className = 'exchange-container';
    exchangeDiv.id = `exchange-${Date.now()}`;
    
    // Create header (will be populated after messages added)
    const headerDiv = document.createElement('div');
    headerDiv.className = 'exchange-header';
    headerDiv.innerHTML = '<span class="exchange-summary">Loading...</span><span class="exchange-toggle">▼</span>';
    headerDiv.addEventListener('click', () => toggleExchange(exchangeDiv.id));
    
    // Create content container
    const contentDiv = document.createElement('div');
    contentDiv.className = 'exchange-content';
    
    exchangeDiv.appendChild(headerDiv);
    exchangeDiv.appendChild(contentDiv);
    chatContainer.appendChild(exchangeDiv);
    
    currentExchange = exchangeDiv;
    
    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function collapseAllExchanges() {
    const exchanges = document.querySelectorAll('.exchange-container');
    exchanges.forEach(exchange => {
        if (!exchange.classList.contains('collapsed')) {
            exchange.classList.add('collapsed');
        }
    });
}

function toggleExchange(exchangeId) {
    const exchange = document.getElementById(exchangeId);
    if (exchange) {
        exchange.classList.toggle('collapsed');
    }
}

function updateExchangeHeader(userMessage) {
    if (currentExchange) {
        const header = currentExchange.querySelector('.exchange-summary');
        const preview = userMessage.length > 60 ? userMessage.substring(0, 60) + '...' : userMessage;
        header.textContent = `💬 ${preview}`;
    }
}

function displayEnsembleOutputs(outputs) {
    const chatContainer = document.getElementById('chat-messages');
    const targetContainer = currentExchange ? currentExchange.querySelector('.exchange-content') : chatContainer;
    
    // Create ensemble container
    const ensembleDiv = document.createElement('div');
    ensembleDiv.className = 'ensemble-outputs';
    ensembleDiv.innerHTML = '<h3 style="margin: 0 0 1rem 0; font-size: 1rem;">🎸 Ensemble Distortion - All 6 Perspectives</h3>';
    
    // Mode labels for display
    const modeLabels = {
        'invert_er': '🔄 Inverter',
        'so_what_er': '❓ So-What-er',
        'echo_er': '📣 Echo-er',
        'what_if_er': '💡 What-If-er',
        'cucumb_er': '🥒 Cucumber',
        'archiv_er': '📚 Archiver'
    };
    
    outputs.forEach((output, i) => {
        const perspectiveDiv = document.createElement('details');
        perspectiveDiv.className = 'ensemble-perspective';
        perspectiveDiv.open = i === 0;  // Open first one by default
        
        const summary = document.createElement('summary');
        summary.textContent = modeLabels[output.mode] || output.mode;
        summary.style.cursor = 'pointer';
        summary.style.fontWeight = '600';
        summary.style.padding = '0.5rem';
        summary.style.background = 'var(--bg-dark)';
        summary.style.borderRadius = '4px';
        summary.style.marginBottom = '0.5rem';
        
        const content = document.createElement('div');
        content.className = 'ensemble-content';
        content.textContent = output.response;
        content.style.padding = '0.5rem';
        content.style.whiteSpace = 'pre-wrap';
        
        perspectiveDiv.appendChild(summary);
        perspectiveDiv.appendChild(content);
        ensembleDiv.appendChild(perspectiveDiv);
    });
    
    targetContainer.appendChild(ensembleDiv);
    
    // Scroll to bottom (find the messages container)
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Remove message from UI
function removeMessage(messageId) {
    const message = document.getElementById(messageId);
    if (message) {
        message.remove();
    }
}

// Update context panel
function updateContextPanel(contextItems) {
    // Context panel removed - this function is now a no-op
    // Context is already displayed inline with messages
    return;
}

// Load sessions
async function loadSessions() {
    try {
        const response = await fetch(`${API_BASE}/api/sessions`);
        const data = await response.json();
        const sessions = data.sessions || [];  // Extract sessions array from response
        
        const sessionList = document.getElementById('session-list');
        sessionList.innerHTML = '';
        
        // Add current session
        const currentDiv = document.createElement('div');
        currentDiv.className = 'session-item active';
        currentDiv.innerHTML = `
            <h4>Current Session</h4>
            <div class="meta">Active • ${state.messages.length} messages</div>
        `;
        sessionList.appendChild(currentDiv);
        
        // Add past sessions
        sessions.forEach(session => {
            const sessionDiv = document.createElement('div');
            sessionDiv.className = 'session-item';
            sessionDiv.innerHTML = `
                <h4>${session.title || 'Untitled Session'}</h4>
                <div class="meta">${new Date(session.created_at).toLocaleDateString()} • ${session.message_count} messages</div>
            `;
            sessionDiv.addEventListener('click', () => loadSession(session.session_id));
            sessionList.appendChild(sessionDiv);
        });
    } catch (error) {
        console.error('Load sessions failed:', error);
    }
}

// Load specific session
async function loadSession(sessionId) {
    try {
        const response = await fetch(`${API_BASE}/api/sessions/${sessionId}`);
        const sessionData = await response.json();
        
        console.log('Loading session:', sessionData);
        
        state.currentSessionId = sessionId;
        state.messages = sessionData.messages || [];
        currentExchange = null;
        
        // Clear and repopulate messages
        const chatContainer = document.getElementById('chat-messages');
        chatContainer.innerHTML = '';
        
        // Check if we have messages
        if (!sessionData.messages || sessionData.messages.length === 0) {
            console.warn('No messages in session');
            // Add empty state message
            const emptyDiv = document.createElement('div');
            emptyDiv.className = 'message assistant';
            emptyDiv.innerHTML = '<div class="message-content"><p>This session has no messages yet.</p></div>';
            chatContainer.appendChild(emptyDiv);
            return;
        }
        
        // Group messages into user-assistant pairs and create exchanges
        for (let i = 0; i < sessionData.messages.length; i++) {
            const msg = sessionData.messages[i];
            
            if (msg.role === 'user') {
                // Start new exchange
                startNewExchange();
                addMessage('user', msg.content, false);
                
                // Add assistant response if it follows
                if (i + 1 < sessionData.messages.length && sessionData.messages[i + 1].role === 'assistant') {
                    i++; // Skip to assistant message
                    const assistantMsg = sessionData.messages[i];
                    addMessage('assistant', assistantMsg.content, false, assistantMsg.metadata?.sources);
                }
                
                // Collapse this exchange (except the last one)
                if (currentExchange && i < sessionData.messages.length - 2) {
                    currentExchange.classList.add('collapsed');
                }
            }
        }
        
        console.log('Session loaded successfully');
        
        // Update session list active state
        document.querySelectorAll('.session-item').forEach(item => {
            item.classList.remove('active');
        });
        
    } catch (error) {
        console.error('Load session failed:', error);
        alert('Failed to load session: ' + error.message);
    }
}

// Create new session
function createNewSession() {
    state.currentSessionId = null;
    state.messages = [];
    currentExchange = null;
    
    const chatContainer = document.getElementById('chat-messages');
    chatContainer.innerHTML = '';
    
    // Create welcome exchange
    const welcomeExchange = document.createElement('div');
    welcomeExchange.className = 'exchange-container';
    welcomeExchange.id = 'welcome-exchange';
    welcomeExchange.innerHTML = `
        <div class="exchange-header">
            <span class="exchange-summary">💬 Welcome Message</span>
            <span class="exchange-toggle">▼</span>
        </div>
        <div class="exchange-content">
            <div class="message assistant">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <p>Hello! I'm MRA v3, your research assistant. What would you like to explore?</p>
                </div>
            </div>
        </div>
    `;
    
    // Add click handler for welcome exchange
    welcomeExchange.querySelector('.exchange-header').addEventListener('click', () => {
        toggleExchange('welcome-exchange');
    });
    
    chatContainer.appendChild(welcomeExchange);
    
    // Update session list
    document.querySelectorAll('.session-item').forEach(item => {
        item.classList.remove('active');
    });
    
    const firstItem = document.querySelector('.session-item');
    if (firstItem) {
        firstItem.classList.add('active');
    }
}

// Utility: Format time ago
function timeAgo(date) {
    const seconds = Math.floor((new Date() - date) / 1000);
    
    let interval = seconds / 31536000;
    if (interval > 1) return Math.floor(interval) + ' years ago';
    
    interval = seconds / 2592000;
    if (interval > 1) return Math.floor(interval) + ' months ago';
    
    interval = seconds / 86400;
    if (interval > 1) return Math.floor(interval) + ' days ago';
    
    interval = seconds / 3600;
    if (interval > 1) return Math.floor(interval) + ' hours ago';
    
    interval = seconds / 60;
    if (interval > 1) return Math.floor(interval) + ' minutes ago';
    
    return 'Just now';
}
