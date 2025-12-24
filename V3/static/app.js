// app.js - MRA v3 Frontend Logic

// State Management
const state = {
    currentSessionId: null,
    messages: [],
    context: [],
    uploadedDocuments: [],  // Store uploaded document content
    settings: {
        model: 'ministral-3:latest',
        temperature: 0.7,
        topP: 0.9,
        topKGen: 40,
        maxTokens: 32000,
        contextWindow: 128000,
        topK: 20,
        searchScope: {
            referencePapers: false,
            myPapers: false,
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

// Configure marked.js for markdown parsing
if (typeof marked !== 'undefined') {
    marked.setOptions({
        breaks: true,  // Convert \n to <br>
        gfm: true,     // GitHub Flavored Markdown
        headerIds: false,
        mangle: false
    });
}

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
    
    document.getElementById('top-p').addEventListener('input', (e) => {
        state.settings.topP = parseFloat(e.target.value);
        document.getElementById('top-p-value').textContent = e.target.value;
    });
    
    document.getElementById('top-k-gen').addEventListener('input', (e) => {
        state.settings.topKGen = parseInt(e.target.value);
        document.getElementById('top-k-gen-value').textContent = e.target.value;
    });
    
    document.getElementById('max-tokens').addEventListener('input', (e) => {
        state.settings.maxTokens = parseInt(e.target.value);
        document.getElementById('max-tokens-value').textContent = e.target.value;
    });
    
    document.getElementById('context-window').addEventListener('input', (e) => {
        state.settings.contextWindow = parseInt(e.target.value);
        document.getElementById('context-window-value').textContent = e.target.value;
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
    
    // Sessions panel toggle
    document.getElementById('sessions-toggle').addEventListener('click', (e) => {
        e.stopPropagation();
        const panel = document.getElementById('sessions-panel');
        const toggle = document.getElementById('sessions-toggle');
        panel.classList.toggle('collapsed');
        toggle.textContent = panel.classList.contains('collapsed') ? '▶' : '▼';
    });
    
    // File upload
    document.getElementById('upload-btn').addEventListener('click', () => {
        document.getElementById('file-upload-input').click();
    });
    
    document.getElementById('file-upload-input').addEventListener('change', handleFileUpload);
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

// Handle file upload
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const statusDiv = document.getElementById('upload-status');
    const fileInput = document.getElementById('file-upload-input');
    
    // Validate file type
    const validExtensions = ['.pdf', '.txt', '.csv', '.md', '.markdown'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validExtensions.includes(fileExtension)) {
        statusDiv.innerHTML = `<span style="color: var(--danger);">❌ Unsupported file type</span>`;
        fileInput.value = '';
        return;
    }
    
    // Show uploading status
    statusDiv.innerHTML = `<span style="color: var(--info);">⏳ Uploading ${file.name}...</span>`;
    
    try {
        // Create FormData
        const formData = new FormData();
        formData.append('file', file);
        
        // Upload file to server
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const data = await response.json();
        
        // Show success
        statusDiv.innerHTML = `<span style="color: var(--success);">✅ ${data.filename}</span>`;
        
        // Store uploaded document in state
        state.uploadedDocuments.push({
            filename: data.filename,
            content: data.markdown_content,
            token_count: data.token_count,
            uploaded_at: new Date().toISOString()
        });
        
        console.log('Uploaded document added to state:', data.filename);
        console.log('Total uploaded documents:', state.uploadedDocuments.length);
        
        // Add uploaded content to chat (visible to user)
        addUploadedDocToChat(data);
        
        // Clear file input after 2 seconds
        setTimeout(() => {
            statusDiv.innerHTML = '';
            fileInput.value = '';
        }, 3000);
        
    } catch (error) {
        console.error('Upload error:', error);
        statusDiv.innerHTML = `<span style="color: var(--danger);">❌ ${error.message}</span>`;
        fileInput.value = '';
    }
}

// Add uploaded document to chat as system message
function addUploadedDocToChat(uploadData) {
    const chatMessages = document.getElementById('chat-messages');
    
    // Create new exchange container for upload notification
    const exchangeId = `exchange-upload-${Date.now()}`;
    const exchangeDiv = document.createElement('div');
    exchangeDiv.className = 'exchange-container';
    exchangeDiv.id = exchangeId;
    
    exchangeDiv.innerHTML = `
        <div class="exchange-header">
            <span class="exchange-summary">📄 Document Uploaded: ${uploadData.filename}</span>
            <span class="exchange-toggle">▼</span>
        </div>
        <div class="exchange-content">
            <div class="message system">
                <div class="message-avatar">📤</div>
                <div class="message-content">
                    <p><strong>Document uploaded and ready for chat:</strong></p>
                    <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                        <li><strong>File:</strong> ${uploadData.filename}</li>
                        <li><strong>Type:</strong> ${uploadData.file_type}</li>
                        <li><strong>Tokens:</strong> ${uploadData.token_count.toLocaleString()}</li>
                        ${uploadData.saved_path ? `<li><strong>Saved:</strong> ${uploadData.saved_path}</li>` : '<li><em>In-memory only</em></li>'}
                    </ul>
                    <p style="margin-top: 0.75rem; font-style: italic;">This document is now available in the chat context. You can ask questions about it!</p>
                </div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(exchangeDiv);
    
    // Add toggle functionality
    const header = exchangeDiv.querySelector('.exchange-header');
    header.addEventListener('click', () => toggleExchange(exchangeId));
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
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
    
    // Create streaming message placeholder
    const streamingId = addStreamingMessage();
    console.log('Created streaming message with ID:', streamingId);
    console.log('Streaming message element:', document.getElementById(streamingId));
    
    try {
        // Prepare request parameters
        const requestData = {
            session_id: state.currentSessionId || 'new',
            message: message,
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
            top_p: state.settings.topP,
            top_k: state.settings.topKGen,
            max_tokens: state.settings.maxTokens,
            num_ctx: state.settings.contextWindow,
            top_k_retrieval: state.settings.topK,
            distortion_mode: state.settings.distortion.mode,
            distortion_tone: state.settings.distortion.tone,
            distortion_gain: state.settings.distortion.gain
        };
        
        // Add uploaded documents to context if any exist
        if (state.uploadedDocuments.length > 0) {
            requestData.uploaded_context = state.uploadedDocuments.map(doc => ({
                filename: doc.filename,
                content: doc.content
            }));
            console.log('Including', state.uploadedDocuments.length, 'uploaded documents in request');
        }
        
        // Use Server-Sent Events for streaming
        const response = await fetch(`${API_BASE}/api/chat/message/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        // Read streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let contextData = null;
        let tokenCount = 0;
        
        console.log('Starting to read streaming response...');
        
        while (true) {
            const {done, value} = await reader.read();
            
            if (done) {
                console.log('Stream completed. Total tokens received:', tokenCount);
                break;
            }
            
            buffer += decoder.decode(value, {stream: true});
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // Keep incomplete line in buffer
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        console.log('Received SSE event:', data.type, data);
                        
                        if (data.type === 'session_id') {
                            state.currentSessionId = data.session_id;
                        } else if (data.type === 'context') {
                            contextData = data.context;
                            console.log('Context received:', contextData.length, 'items');
                        } else if (data.type === 'token') {
                            tokenCount++;
                            appendToStreamingMessage(streamingId, data.content);
                        } else if (data.type === 'ensemble') {
                            // Ensemble distortion outputs received
                            displayEnsembleOutputs(data.outputs);
                        } else if (data.type === 'distortion') {
                            // Single distortion output - replace main response
                            replaceStreamingMessageContent(streamingId, data.content);
                        } else if (data.type === 'done') {
                            console.log('Finalizing message with', tokenCount, 'tokens');
                            finalizeStreamingMessage(streamingId, contextData);
                        } else if (data.type === 'error') {
                            throw new Error(data.message);
                        }
                    } catch (e) {
                        console.error('Failed to parse SSE:', e, 'Line:', line);
                    }
                }
            }
        }
        
    } catch (error) {
        console.error('Send message failed:', error);
        removeMessage(streamingId);
        addMessage('assistant', `❌ Error: ${error.message}`, true);
    } finally {
        sendBtn.disabled = false;
    }
}

// Add streaming message placeholder
function addStreamingMessage() {
    const messagesContainer = document.getElementById('chat-messages');
    const targetContainer = currentExchange ? currentExchange.querySelector('.exchange-content') : messagesContainer;
    const messageId = `msg-assistant-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    console.log('Creating streaming message:', messageId);
    console.log('Target container:', targetContainer);
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = messageId;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content streaming';
    contentDiv.innerHTML = '<span class="streaming-cursor">▋</span>';
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    targetContainer.appendChild(messageDiv);
    
    console.log('Streaming message created and appended');
    
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return messageId;
}

// Append token to streaming message
function appendToStreamingMessage(messageId, token) {
    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) {
        console.error('Message div not found:', messageId);
        return;
    }
    
    console.log('Appending to element:', messageDiv.className, 'ID:', messageId);
    
    const contentDiv = messageDiv.querySelector('.message-content');
    if (!contentDiv) {
        console.error('Content div not found in message:', messageId);
        return;
    }
    
    console.log('Content div classes:', contentDiv.className);
    
    let cursor = contentDiv.querySelector('.streaming-cursor');
    
    // Insert token before cursor
    if (cursor) {
        const textNode = document.createTextNode(token);
        cursor.parentNode.insertBefore(textNode, cursor);
    } else {
        // Cursor missing - recreate it and append token
        console.warn('Streaming cursor not found, recreating it');
        contentDiv.appendChild(document.createTextNode(token));
        
        // Recreate cursor if this is the first token
        if (contentDiv.textContent.length <= token.length) {
            const newCursor = document.createElement('span');
            newCursor.className = 'streaming-cursor';
            newCursor.textContent = '▋';
            contentDiv.appendChild(newCursor);
        }
    }
    
    // Auto-scroll
    const messagesContainer = document.getElementById('chat-messages');
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Replace streaming message content with distorted version
function replaceStreamingMessageContent(messageId, newContent) {
    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) return;
    
    const contentDiv = messageDiv.querySelector('.message-content');
    const cursor = contentDiv.querySelector('.streaming-cursor');
    
    // Clear current content but keep cursor
    contentDiv.innerHTML = '';
    contentDiv.appendChild(document.createTextNode(newContent));
    if (cursor) {
        contentDiv.appendChild(cursor);
    }
}

// Finalize streaming message
function finalizeStreamingMessage(messageId, contextData = null) {
    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) return;
    
    const contentDiv = messageDiv.querySelector('.message-content');
    
    // Remove cursor
    const cursor = contentDiv.querySelector('.streaming-cursor');
    if (cursor) cursor.remove();
    
    // Remove streaming class
    contentDiv.classList.remove('streaming');
    
    // Get accumulated text
    const text = contentDiv.textContent.trim();
    
    // Check if we actually got content
    if (!text || text.length === 0) {
        console.warn('No content received from LLM');
        contentDiv.textContent = '⚠️ No response received from LLM. The model may have timed out or returned an empty response.';
        return;
    }
    
    // Store original markdown as data attribute
    messageDiv.setAttribute('data-markdown', text);
    
    // Parse as markdown
    if (typeof marked !== 'undefined') {
        contentDiv.innerHTML = marked.parse(text);
    }
    
    // Add copy markdown button
    addCopyButton(messageDiv, contentDiv, text);
    
    // Add sources if available
    if (contextData && contextData.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        sourcesDiv.innerHTML = '<strong>Sources:</strong><br>';
        
        contextData.forEach((source, i) => {
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
        // Store original markdown for assistant messages
        if (role === 'assistant') {
            messageDiv.setAttribute('data-markdown', content);
        }
        
        // Parse markdown for assistant messages, plain text for user messages
        if (role === 'assistant' && typeof marked !== 'undefined') {
            contentDiv.innerHTML = marked.parse(content);
            // Add copy markdown button
            addCopyButton(messageDiv, contentDiv, content);
        } else {
            contentDiv.textContent = content;
        }
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

// Add copy markdown button to message
function addCopyButton(messageDiv, contentDiv, markdownText) {
    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-markdown-btn';
    copyBtn.innerHTML = '📋 Copy Markdown';
    copyBtn.title = 'Copy original markdown to clipboard';
    
    copyBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        
        // Get markdown text from data attribute if not provided
        const textToCopy = markdownText || messageDiv.getAttribute('data-markdown') || '';
        
        if (!textToCopy) {
            console.error('No markdown text to copy');
            copyBtn.innerHTML = '❌ No text';
            setTimeout(() => {
                copyBtn.innerHTML = '📋 Copy Markdown';
            }, 2000);
            return;
        }
        
        try {
            // Try modern Clipboard API first
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(textToCopy);
            } else {
                // Fallback to older method
                const textarea = document.createElement('textarea');
                textarea.value = textToCopy;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                document.body.appendChild(textarea);
                textarea.select();
                const success = document.execCommand('copy');
                document.body.removeChild(textarea);
                
                if (!success) {
                    throw new Error('execCommand failed');
                }
            }
            
            // Show success feedback
            const originalText = copyBtn.innerHTML;
            copyBtn.innerHTML = '✅ Copied!';
            copyBtn.classList.add('copied');
            setTimeout(() => {
                copyBtn.innerHTML = originalText;
                copyBtn.classList.remove('copied');
            }, 2000);
            
        } catch (err) {
            console.error('Failed to copy:', err, 'Text length:', textToCopy.length);
            copyBtn.innerHTML = '❌ Failed';
            setTimeout(() => {
                copyBtn.innerHTML = '📋 Copy Markdown';
            }, 2000);
        }
    });
    
    // Insert button at the top of content (before markdown rendering)
    contentDiv.insertBefore(copyBtn, contentDiv.firstChild);
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
