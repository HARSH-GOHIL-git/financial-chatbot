    // Application State
let activeThreadId = null;
let threadsList = [];
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let selectedFile = null;
let activeToolElement = null;
let selectedLanguage = 'English';
let isGenerating = false;
let currentAbortController = null;

function updateSendBtnState() {
    if (isGenerating) {
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<i class="fa-solid fa-stop"></i>';
        sendBtn.title = "Stop Generating";
        sendBtn.classList.add('stop-active');
    } else {
        sendBtn.disabled = chatInput.value.trim() === '';
        sendBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i>';
        sendBtn.title = "Send Message";
        sendBtn.classList.remove('stop-active');
    }
}

async function stopGeneration() {
    if (currentAbortController) {
        currentAbortController.abort();
    }
    try {
        await fetch('/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ thread_id: activeThreadId })
        });
    } catch (err) {
        console.warn("Error calling stop endpoint:", err);
    }
    isGenerating = false;
    updateSendBtnState();
    chatInput.disabled = false;
    chatInput.focus();
}


// DOM Elements
const sidebar = document.getElementById('sidebar');
const sidebarBackdrop = document.getElementById('sidebar-backdrop');
const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
const sidebarCloseBtn = document.getElementById('sidebar-close-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const sessionsList = document.getElementById('sessions-list');
const pdfInput = document.getElementById('pdf-input');
const uploadZone = document.getElementById('upload-zone');
const uploadBtn = document.getElementById('upload-btn');
const uploadFilename = document.getElementById('upload-filename');
const uploadProgressContainer = document.getElementById('upload-progress-container');
const uploadProgressBar = document.getElementById('upload-progress-bar');
const uploadStatus = document.getElementById('upload-status');
const activeSessionName = document.getElementById('active-session-name');
const activeSessionIdCaption = document.getElementById('active-session-id-caption');
const editNameBtn = document.getElementById('edit-name-btn');
const deleteSessionBtn = document.getElementById('delete-session-btn');
const exportThreadBtn = document.getElementById('export-thread-btn');
const renameModal = document.getElementById('rename-modal');
const renameInput = document.getElementById('rename-input');
const renameCancelBtn = document.getElementById('rename-cancel-btn');
const renameSaveBtn = document.getElementById('rename-save-btn');
const messagesContainer = document.getElementById('messages-container');
const welcomeScreen = document.getElementById('welcome-screen');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const voiceNotification = document.getElementById('voice-notification');

// Initialize Marked.js Options
marked.setOptions({
    breaks: true,
    highlight: function(code, lang) {
        if (Prism.languages[lang]) {
            return Prism.highlight(code, Prism.languages[lang], lang);
        }
        return code;
    }
});

// App Startup
document.addEventListener('DOMContentLoaded', () => {
    initApp();
    setupEventListeners();
});

// Helper: Generate UUID v4
function generateUUID() {
    return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
        (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
    );
}

// Helper: Auto-resize chat textarea input height
function autoResizeInput() {
    chatInput.style.height = 'auto';
    chatInput.style.height = chatInput.scrollHeight + 'px';
}

// Helper: Scroll messages container to bottom
function scrollToBottom() {
    messagesContainer.scrollTo({
        top: messagesContainer.scrollHeight,
        behavior: 'smooth'
    });
}

// Initial Loading Logic
async function initApp() {
    // Check if session ID was stored in local storage
    const storedThreadId = localStorage.getItem('active_thread_id');
    
    // Load languages list from backend first
    await loadLanguages();
    
    try {
        await refreshThreads();
        
        if (threadsList.length > 0) {
            // Check if stored ID still exists in the refreshed list
            const exists = threadsList.some(t => t.thread_id === storedThreadId);
            const idToLoad = exists ? storedThreadId : threadsList[0].thread_id;
            switchThread(idToLoad);
        } else {
            // Brand new application - create a new thread
            createNewSession();
        }
    } catch (err) {
        console.error("Failed to connect to backend:", err);
        // Fallback to offline UUID creation
        createNewSession();
    }
}

// Fetch and populate supported languages list
async function loadLanguages() {
    const langSelect = document.getElementById('language-select');
    if (!langSelect) return;
    
    // Default fallback languages list
    const fallbackLanguages = [
        {"code": "en", "name": "English"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"},
        {"code": "de", "name": "German"},
        {"code": "zh", "name": "Chinese"},
        {"code": "ja", "name": "Japanese"},
        {"code": "hi", "name": "Hindi"},
        {"code": "pt", "name": "Portuguese"},
        {"code": "ru", "name": "Russian"},
        {"code": "it", "name": "Italian"},
        {"code": "ar", "name": "Arabic"},
        {"code": "ko", "name": "Korean"},
        {"code": "tr", "name": "Turkish"},
        {"code": "vi", "name": "Vietnamese"},
        {"code": "nl", "name": "Dutch"}
    ];
    
    let languages = [];
    try {
        const response = await fetch('/languages');
        if (!response.ok) throw new Error('API error');
        const data = await response.json();
        languages = data.languages || [];
    } catch (err) {
        console.warn("Could not retrieve supported languages from backend, using client fallback:", err);
        languages = fallbackLanguages;
    }
    
    if (languages.length > 0) {
        langSelect.innerHTML = '';
        languages.forEach(lang => {
            const opt = document.createElement('option');
            opt.value = lang.name;
            opt.textContent = lang.name;
            langSelect.appendChild(opt);
        });
        
        // Restore selection from localStorage if valid
        const storedLang = localStorage.getItem('selected_language');
        if (storedLang && languages.some(l => l.name === storedLang)) {
            selectedLanguage = storedLang;
            langSelect.value = storedLang;
        } else {
            selectedLanguage = 'English';
            langSelect.value = 'English';
        }
    }
}




// Create New Session Local Logic
function createNewSession() {
    const newId = generateUUID();
    activeThreadId = newId;
    localStorage.setItem('active_thread_id', newId);
    
    // Clear log and show welcome screen
    messagesContainer.innerHTML = '';
    welcomeScreen.style.display = 'flex';
    
    // Update labels
    activeSessionName.textContent = `${newId.substring(0, 8)}...`;
    activeSessionIdCaption.textContent = `ID: ${newId}`;
    
    // Render list
    renderThreadsSidebar();
    loadThreadDocuments(newId);
}

// Fetch Thread List from backend
async function refreshThreads() {
    try {
        const response = await fetch('/threads');
        if (!response.ok) throw new Error('API error');
        const data = await response.json();
        threadsList = data.threads || [];
    } catch (err) {
        console.warn("Could not retrieve threads from database:", err);
        threadsList = [];
    }
}

// Render Thread List in Sidebar
function renderThreadsSidebar() {
    sessionsList.innerHTML = '';
    
    // If the active thread is temporary (not saved yet in backend), prepend it
    const activeExistsInBackend = threadsList.some(t => t.thread_id === activeThreadId);
    let displayList = [...threadsList];
    
    if (!activeExistsInBackend && activeThreadId) {
        displayList.unshift({
            thread_id: activeThreadId,
            thread_name: `${activeThreadId.substring(0, 8)}...`
        });
    }

    displayList.forEach(t => {
        const li = document.createElement('li');
        li.className = `session-item ${t.thread_id === activeThreadId ? 'active' : ''}`;
        
        const nameSpan = document.createElement('span');
        nameSpan.className = 'session-item-name';
        nameSpan.textContent = t.thread_name;
        
        li.appendChild(nameSpan);
        li.addEventListener('click', () => switchThread(t.thread_id));
        sessionsList.appendChild(li);
    });
}

// Switch Thread Logic
async function switchThread(threadId) {
    if (activeThreadId === threadId && messagesContainer.children.length > 1) return;
    
    // Clear upload status when switching threads
    if (uploadStatus) {
        uploadStatus.textContent = '';
        uploadStatus.className = 'upload-status';
    }
    
    activeThreadId = threadId;
    localStorage.setItem('active_thread_id', threadId);
    renderThreadsSidebar();
    
    // Collapse sidebar on mobile screens
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('open');
        if (sidebarBackdrop) sidebarBackdrop.classList.remove('active');
    }
    
    // Find active metadata
    const activeSession = threadsList.find(t => t.thread_id === threadId);
    activeSessionName.textContent = activeSession ? activeSession.thread_name : `${threadId.substring(0, 8)}...`;
    activeSessionIdCaption.textContent = `ID: ${threadId}`;
    
    // Load history
    welcomeScreen.style.display = 'none';
    messagesContainer.innerHTML = '<div style="text-align: center; color: var(--text-secondary); margin: 20px;"><i class="fa-solid fa-spinner fa-spin"></i> Loading history...</div>';
    
    try {
        const response = await fetch(`/history/${threadId}`);
        if (!response.ok) throw new Error('History retrieval error');
        const messages = await response.json();
        
        messagesContainer.innerHTML = '';
        
        if (messages.length === 0) {
            welcomeScreen.style.display = 'flex';
        } else {
            messages.forEach(msg => {
                appendMessageBubble(msg.role, msg.content, false, msg.timestamp, msg.tool_calls);
            });
        }
    } catch (err) {
        console.error("Error fetching session history:", err);
        messagesContainer.innerHTML = '<div style="text-align: center; color: var(--accent-red); margin: 20px;"><i class="fa-solid fa-circle-exclamation"></i> Failed to load message history.</div>';
    }
    
    scrollToBottom();
    loadThreadDocuments(threadId);
}

// Load active thread's indexed documents
async function loadThreadDocuments(threadId) {
    const kbFilesSection = document.getElementById('kb-files-section');
    const kbFilesList = document.getElementById('kb-files-list');
    if (!kbFilesSection || !kbFilesList) return;

    try {
        const response = await fetch(`/threads/${threadId}/documents`);
        if (!response.ok) throw new Error('Failed to fetch documents');
        const data = await response.json();
        
        kbFilesList.innerHTML = '';
        if (data.documents && data.documents.length > 0) {
            data.documents.forEach(docObj => {
                const docName = docObj.filename;
                const chunkCount = docObj.chunks;
                const docItem = document.createElement('div');
                docItem.className = 'kb-file-item';
                const lowerName = docName.toLowerCase();
                let iconClass = 'fa-solid fa-file-pdf';
                if (lowerName.endsWith('.xlsx') || lowerName.endsWith('.xls')) {
                    iconClass = 'fa-solid fa-file-excel';
                } else if (lowerName.endsWith('.csv')) {
                    iconClass = 'fa-solid fa-file-csv';
                } else if (lowerName.endsWith('.docx')) {
                    iconClass = 'fa-solid fa-file-word';
                }
                docItem.innerHTML = `
                    <div style="display: flex; align-items: center; gap: 8px; flex-grow: 1; overflow: hidden;">
                        <i class="${iconClass}"></i>
                        <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: flex; align-items: center; gap: 6px;">
                            ${docName}
                            ${chunkCount > 0 ? `<span class="kb-file-chunks" style="color: var(--text-muted); font-size: 10px; font-weight: 500;">(${chunkCount} chunks)</span>` : ''}
                        </span>
                    </div>
                    <button class="kb-file-remove-btn" onclick="removeSource(event, '${docName.replace(/'/g, "\\'")}')" title="Remove Source"><i class="fa-solid fa-xmark"></i></button>
                `;
                kbFilesList.appendChild(docItem);
            });
            kbFilesSection.style.display = 'block';
        } else {
            kbFilesSection.style.display = 'none';
        }
    } catch (err) {
        console.error("Error fetching thread documents:", err);
        kbFilesSection.style.display = 'none';
    }
}

// Click suggested queries
window.setQuery = function(text) {
    if (isGenerating) return;
    chatInput.value = text;
    sendBtn.disabled = false;
    autoResizeInput();
    chatInput.focus();
}

// Format timestamp helper
function formatTimestamp(timestampStr) {
    if (!timestampStr) return "";
    const date = new Date(timestampStr);
    if (isNaN(date.getTime())) return "";
    
    let hours = date.getHours();
    let minutes = date.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12;
    hours = hours ? hours : 12; // the hour '0' should be '12'
    minutes = minutes < 10 ? '0' + minutes : minutes;
    const timeStr = `${hours}:${minutes} ${ampm}`;

    const today = new Date();
    const yesterday = new Date();
    yesterday.setDate(today.getDate() - 1);

    let datePrefix = "";
    if (date.toDateString() === today.toDateString()) {
        datePrefix = "Today, ";
    } else if (date.toDateString() === yesterday.toDateString()) {
        datePrefix = "Yesterday, ";
    } else {
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        datePrefix = `${months[date.getMonth()]} ${date.getDate()}, `;
    }

    return `${datePrefix}${timeStr}`;
}

// Format absolute timestamp helper (DD-MM-YYYY, HH:MM AM/PM) for exports
function formatAbsoluteTimestamp(timestampStr) {
    if (!timestampStr) return "";
    const date = new Date(timestampStr);
    if (isNaN(date.getTime())) return "";
    
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = date.getFullYear();

    let hours = date.getHours();
    let minutes = date.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12;
    hours = hours ? hours : 12;
    minutes = minutes < 10 ? '0' + minutes : minutes;
    
    return `${day}-${month}-${year}, ${hours}:${minutes} ${ampm}`;
}

// Append Chat Message bubble
function appendMessageBubble(role, content = '', isStreaming = false, timestamp = null, toolCalls = null) {
    const row = document.createElement('div');
    row.className = `chat-msg-row ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.innerHTML = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
    
    const wrapper = document.createElement('div');
    wrapper.className = 'msg-content-wrapper';
    
    const bubble = document.createElement('div');
    bubble.className = `msg-bubble ${isStreaming ? 'streaming-active' : ''}`;
    
    if (role === 'assistant') {
        const toolsDiv = document.createElement('div');
        toolsDiv.className = 'msg-tools-container';
        bubble.appendChild(toolsDiv);

        // Pre-populate tool calls if they exist in history
        if (toolCalls && toolCalls.length > 0) {
            toolCalls.forEach(tc => {
                const toolEl = appendToolStatus(tc.name, tc.args, bubble);
                completeToolStatus(toolEl, tc.name, tc.output);
            });
        }

        const textDiv = document.createElement('div');
        textDiv.className = 'msg-text';
        if (content) {
            textDiv.innerHTML = marked.parse(content);
        } else {
            textDiv.innerHTML = '<p>...</p>';
        }
        bubble.appendChild(textDiv);
    } else {
        if (content) {
            bubble.innerHTML = marked.parse(content);
        } else {
            bubble.innerHTML = '<p>...</p>';
        }
    }
    
    const timestampDiv = document.createElement('div');
    timestampDiv.className = 'msg-timestamp';
    timestampDiv.textContent = formatTimestamp(timestamp);
    bubble.appendChild(timestampDiv);
    
    wrapper.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(wrapper);
    
    messagesContainer.appendChild(row);
    
    // Highlight syntax
    Prism.highlightAllUnder(bubble);
    
    scrollToBottom();
    return bubble;
}

// Append Tool Exec Status log
function appendToolStatus(name, args, parentBubble = null) {
    const details = document.createElement('details');
    details.className = 'tool-status-details';
    
    const summary = document.createElement('summary');
    summary.className = 'tool-status-summary';
    summary.innerHTML = `
        <i class="fa-solid fa-gear tool-spin"></i>
        <span>Running tool <code>${name}</code>...</span>
    `;
    
    const content = document.createElement('div');
    content.className = 'tool-details';
    content.innerHTML = `
        <strong>Arguments:</strong>
        <pre><code class="language-json">${JSON.stringify(args, null, 2)}</code></pre>
    `;
    
    details.appendChild(summary);
    details.appendChild(content);
    
    if (parentBubble) {
        const toolsContainer = parentBubble.querySelector('.msg-tools-container');
        if (toolsContainer) {
            toolsContainer.appendChild(details);
        } else {
            parentBubble.appendChild(details);
        }
    } else {
        messagesContainer.appendChild(details);
    }
    
    Prism.highlightAllUnder(content);
    scrollToBottom();
    
    return details;
}

// Complete Tool Exec Status log
function completeToolStatus(element, name, output) {
    if (!element) return;
    const summary = element.querySelector('.tool-status-summary');
    if (summary) {
        summary.className = 'tool-status-summary complete';
        summary.innerHTML = `
            <i class="fa-solid fa-circle-check"></i>
            <span>Finished running <code>${name}</code></span>
        `;
    }
    
    const detailsContent = element.querySelector('.tool-details');
    if (detailsContent) {
        // Parse JSON outputs if possible for cleaner render
        let outputText = output;
        try {
            const parsed = JSON.parse(output);
            outputText = JSON.stringify(parsed, null, 2);
        } catch(e) {}
        
        const resultsContainer = document.createElement('div');
        resultsContainer.style.marginTop = '10px';
        resultsContainer.innerHTML = `
            <strong>Output:</strong>
            <pre><code class="language-json">${outputText}</code></pre>
        `;
        
        detailsContent.appendChild(resultsContainer);
        Prism.highlightAllUnder(resultsContainer);
    }
}

// Sending Messages Core Logic
async function sendMessage() {
    if (isGenerating) {
        stopGeneration();
        return;
    }

    const text = chatInput.value.trim();
    if (!text) return;
    
    // UI states
    welcomeScreen.style.display = 'none';
    chatInput.value = '';
    autoResizeInput();
    
    isGenerating = true;
    updateSendBtnState();
    chatInput.disabled = true;
    
    // Append user message
    appendMessageBubble('user', text, false, new Date().toISOString());
    
    // Setup assistant bubble
    const assistantBubble = appendMessageBubble('assistant', '', true, new Date().toISOString());
    
    currentAbortController = new AbortController();
    const { signal } = currentAbortController;

    // Fetch and Stream response
    try {
        const response = await fetch('/chat_stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: text,
                thread_id: activeThreadId,
                language: selectedLanguage
            }),
            signal: signal
        });
        
        if (!response.ok) throw new Error('Network response not ok');
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let assistantReply = "";
        activeToolElement = null;
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep partial line
            
            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const event = JSON.parse(line);
                    
                    if (event.type === 'text') {
                        assistantReply += event.content;
                        // Render updated markdown inside the text sub-container
                        const textDiv = assistantBubble.querySelector('.msg-text');
                        if (textDiv) {
                            textDiv.innerHTML = marked.parse(assistantReply);
                            Prism.highlightAllUnder(textDiv);
                        } else {
                            assistantBubble.innerHTML = marked.parse(assistantReply);
                            Prism.highlightAllUnder(assistantBubble);
                        }
                        scrollToBottom();
                    } else if (event.type === 'tool_start') {
                        activeToolElement = appendToolStatus(event.name, event.args, assistantBubble);
                    } else if (event.type === 'tool_end') {
                        completeToolStatus(activeToolElement, event.name, event.output);
                        activeToolElement = null;
                    } else if (event.type === 'meta') {
                        const timestampDiv = assistantBubble.querySelector('.msg-timestamp');
                        if (timestampDiv && event.timestamp) {
                            timestampDiv.textContent = formatTimestamp(event.timestamp);
                        }
                    }
                } catch (e) {
                    console.warn("JSON parse error on stream line:", e, line);
                }
            }
        }
        
        // Finalize stream bubble
        assistantBubble.classList.remove('streaming-active');
        if (!assistantReply) {
            const toolsContainer = assistantBubble.querySelector('.msg-tools-container');
            const textDiv = assistantBubble.querySelector('.msg-text');
            if (textDiv && textDiv.innerHTML === '<p>...</p>' && (!toolsContainer || !toolsContainer.children.length)) {
                textDiv.textContent = "No reply received.";
            }
        }
        
        // Refresh sessions to capture thread in side panel if it was first message
        await refreshThreads();
        renderThreadsSidebar();
        
    } catch (err) {
        if (err.name === 'AbortError') {
            console.log("Chat stream aborted by user.");
            assistantBubble.classList.remove('streaming-active');
            const textDiv = assistantBubble.querySelector('.msg-text');
            if (textDiv) {
                if (textDiv.innerHTML === '<p>...</p>') {
                    textDiv.innerHTML = "<p><em>Generation stopped by user.</em></p>";
                } else {
                    textDiv.innerHTML += "<p><em>[Generation stopped by user]</em></p>";
                }
            }
        } else {
            console.error("Error streaming chat:", err);
            assistantBubble.classList.remove('streaming-active');
            assistantBubble.innerHTML = `<span style="color: var(--accent-red)"><i class="fa-solid fa-circle-exclamation"></i> Error communicating with server: ${err.message}</span>`;
        }
    } finally {
        isGenerating = false;
        currentAbortController = null;
        updateSendBtnState();
        chatInput.disabled = false;
        chatInput.focus();
    }
}

// Voice recording logic
async function toggleVoiceRecording() {
    if (isRecording) {
        // Stop recording
        mediaRecorder.stop();
        isRecording = false;
        micBtn.classList.remove('recording');
        voiceNotification.style.display = 'none';
        return;
    }
    
    // Start recording
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Audio recording not supported in this browser.");
        return;
    }
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunks = [];
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            
            // Send file to /transcribe
            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.wav');
            
            voiceNotification.style.display = 'flex';
            voiceNotification.querySelector('.voice-text').textContent = "Transcribing voice...";
            voiceNotification.querySelector('.mic-wave').style.opacity = '0.3';
            
            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });
                if (!response.ok) throw new Error('Transcription error');
                const result = await response.json();
                
                if (result.text && result.text.trim()) {
                    chatInput.value = result.text;
                    if (!isGenerating) {
                        sendBtn.disabled = false;
                    }
                    autoResizeInput();
                    chatInput.focus();
                } else {
                    console.log("Empty transcription received.");
                }
            } catch (err) {
                console.error("Transcription failed:", err);
                alert("Could not transcribe voice audio: " + err.message);
            } finally {
                voiceNotification.style.display = 'none';
                voiceNotification.querySelector('.mic-wave').style.opacity = '1';
                // Stop audio tracks
                stream.getTracks().forEach(track => track.stop());
            }
        };
        
        mediaRecorder.start();
        isRecording = true;
        micBtn.classList.add('recording');
        voiceNotification.style.display = 'flex';
        voiceNotification.querySelector('.voice-text').textContent = "Listening... Click microphone again to submit.";
        
    } catch (err) {
        console.error("Could not capture audio:", err);
        alert("Microphone access denied or unavailable.");
    }
}

// PDF Upload Execution
async function uploadPdfFile() {
    if (!selectedFile) return;
    
    uploadProgressContainer.style.display = 'block';
    uploadProgressBar.style.width = '20%';
    uploadStatus.className = 'upload-status';
    uploadStatus.textContent = "Uploading file...";
    uploadBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    const isExcel = selectedFile.name.toLowerCase().endsWith('.xlsx') || 
                    selectedFile.name.toLowerCase().endsWith('.xls') || 
                    selectedFile.name.toLowerCase().endsWith('.csv');
    const endpoint = isExcel ? `/upload-excel?thread_id=${activeThreadId}` : `/upload-pdf?thread_id=${activeThreadId}`;
    
    try {
        uploadProgressBar.style.width = '60%';
        
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Indexing request failed');
        const data = await response.json();
        
        uploadProgressBar.style.width = '100%';
        uploadStatus.className = 'upload-status success';
        uploadStatus.textContent = data.message || "File uploaded and indexed successfully!";
        
        // Reset file select
        selectedFile = null;
        uploadFilename.textContent = '';
        
        // Refresh knowledge base documents list
        loadThreadDocuments(activeThreadId);
        
        setTimeout(() => {
            uploadProgressContainer.style.display = 'none';
            uploadProgressBar.style.width = '0%';
        }, 3000);
        
        setTimeout(() => {
            uploadStatus.textContent = '';
            uploadStatus.className = 'upload-status';
        }, 4000);
        
    } catch(err) {
        console.error("Upload error:", err);
        uploadStatus.className = 'upload-status error';
        uploadStatus.textContent = "Upload failed: " + err.message;
        uploadProgressBar.style.width = '0%';
        uploadProgressContainer.style.display = 'none';
        uploadBtn.disabled = false;
    }
}

// Rename thread execution
async function executeRename() {
    const newNameText = renameInput.value.trim();
    if (!newNameText) return;
    
    try {
        const response = await fetch(`/threads/${activeThreadId}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ thread_name: newNameText })
        });
        
        if (!response.ok) throw new Error('Rename request failed');
        
        // Refresh and render
        await refreshThreads();
        renderThreadsSidebar();
        
        activeSessionName.textContent = newNameText;
        closeRenameModal();
        
    } catch (err) {
        console.error("Rename failed:", err);
        alert("Failed to rename session: " + err.message);
    }
}

// Delete thread execution
async function executeDelete() {
    if (!confirm("Are you sure you want to delete this session? All conversation state will be deleted permanently.")) return;
    
    try {
        const response = await fetch(`/threads/${activeThreadId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Delete request failed');
        
        // Refresh local memory
        await refreshThreads();
        
        if (threadsList.length > 0) {
            switchThread(threadsList[0].thread_id);
        } else {
            createNewSession();
        }
        
    } catch (err) {
        console.error("Delete failed:", err);
        alert("Failed to delete session: " + err.message);
    }
}

// Export current thread to Markdown file
async function exportCurrentThread() {
    if (!activeThreadId) {
        alert("No active session to export.");
        return;
    }
    try {
        const response = await fetch(`/history/${activeThreadId}`);
        if (!response.ok) throw new Error("Failed to fetch session history.");
        const messages = await response.json();
        
        if (messages.length === 0) {
            alert("This session has no messages to export.");
            return;
        }

        const threadName = activeSessionName.textContent.trim() || "Chat Session";

        let mdContent = `# Chat Session: ${threadName}\n`;
        mdContent += `**Thread ID:** \`${activeThreadId}\`  \n`;
        mdContent += `**Exported On:** ${new Date().toLocaleString()}  \n\n`;
        mdContent += `---\n\n`;

        messages.forEach((msg, idx) => {
            const roleLabel = msg.role === 'user' ? '👤 User' : '🤖 Assistant';
            const formattedTime = formatAbsoluteTimestamp(msg.timestamp);

            mdContent += `### ${roleLabel} (${formattedTime})\n\n`;

            if (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0) {
                mdContent += `#### 🛠️ Tool Execution Logs\n\n`;
                msg.tool_calls.forEach(tc => {
                    mdContent += `* **Tool:** \`${tc.name}\`\n`;
                    mdContent += `  * **Arguments:**\n\`\`\`json\n${JSON.stringify(tc.args, null, 2)}\n\`\`\`\n`;
                    
                    let cleanOutput = tc.output || "";
                    try {
                        const parsed = JSON.parse(cleanOutput);
                        cleanOutput = JSON.stringify(parsed, null, 2);
                    } catch(e) {}
                    
                    mdContent += `  * **Output:**\n\`\`\`json\n${cleanOutput}\n\`\`\`\n\n`;
                });
                mdContent += `#### 💬 Response\n\n`;
            }

            mdContent += `${msg.content || "_No response content_"}\n\n`;
            mdContent += `---\n\n`;
        });

        // Trigger file download
        const blob = new Blob([mdContent], { type: 'text/markdown;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        
        const safeName = threadName.replace(/[^a-z0-9]/gi, '_').toLowerCase();
        link.setAttribute("href", url);
        link.setAttribute("download", `chat_export_${safeName}_${activeThreadId.slice(0, 8)}.md`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

    } catch (err) {
        console.error("Error exporting session:", err);
        alert("Error exporting thread: " + err.message);
    }
}

// Modal open/close helpers
function openRenameModal() {
    renameInput.value = activeSessionName.textContent;
    renameModal.classList.add('open');
    renameInput.focus();
}

function closeRenameModal() {
    renameModal.classList.remove('open');
}

// Event Listeners setup
function setupEventListeners() {
    // Language Selection Event
    const langSelect = document.getElementById('language-select');
    if (langSelect) {
        langSelect.addEventListener('change', (e) => {
            selectedLanguage = e.target.value;
            localStorage.setItem('selected_language', selectedLanguage);
        });
    }


    // Sidebar responsive toggle
    sidebarToggleBtn.addEventListener('click', () => {
        sidebar.classList.add('open');
        if (sidebarBackdrop) sidebarBackdrop.classList.add('active');
    });
    
    sidebarCloseBtn.addEventListener('click', () => {
        sidebar.classList.remove('open');
        if (sidebarBackdrop) sidebarBackdrop.classList.remove('active');
    });
    
    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener('click', () => {
            sidebar.classList.remove('open');
            sidebarBackdrop.classList.remove('active');
        });
    }
    
    // Create new chat
    newChatBtn.addEventListener('click', () => {
        createNewSession();
        // Collapse sidebar on mobile screens
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('open');
            if (sidebarBackdrop) sidebarBackdrop.classList.remove('active');
        }
    });
    
    // Rename & delete actions
    editNameBtn.addEventListener('click', openRenameModal);
    renameCancelBtn.addEventListener('click', closeRenameModal);
    renameSaveBtn.addEventListener('click', executeRename);
    renameInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') executeRename();
    });
    
    deleteSessionBtn.addEventListener('click', executeDelete);
    exportThreadBtn.addEventListener('click', exportCurrentThread);
    
    // Chat text area inputs
    chatInput.addEventListener('input', () => {
        if (!isGenerating) {
            sendBtn.disabled = chatInput.value.trim() === '';
        }
        autoResizeInput();
    });
    
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isGenerating) {
                sendMessage();
            }
        }
    });
    
    sendBtn.addEventListener('click', sendMessage);
    
    // Microphone buttons
    micBtn.addEventListener('click', toggleVoiceRecording);
    
    // PDF upload drag-and-drop zone
    uploadZone.addEventListener('click', () => pdfInput.click());
    
    pdfInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            uploadFilename.textContent = selectedFile.name;
            uploadBtn.disabled = false;
            uploadStatus.textContent = '';
        }
    });
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            selectedFile = e.dataTransfer.files[0];
            const name = selectedFile.name.toLowerCase();
            if (name.endsWith('.pdf') || name.endsWith('.xlsx') || name.endsWith('.xls') || name.endsWith('.csv') || name.endsWith('.docx')) {
                uploadFilename.textContent = selectedFile.name;
                uploadBtn.disabled = false;
                uploadStatus.textContent = '';
            } else {
                alert("Only PDF, Word (.docx), Excel (.xlsx, .xls) and CSV (.csv) files are supported.");
                selectedFile = null;
            }
        }
    });
    
    uploadBtn.addEventListener('click', uploadPdfFile);
}

// Confirmation modal helper
function showConfirmModal(message, onConfirm) {
    const confirmModal = document.getElementById('confirm-modal');
    const confirmMessage = document.getElementById('confirm-modal-message');
    const confirmSaveBtn = document.getElementById('confirm-save-btn');
    const confirmCancelBtn = document.getElementById('confirm-cancel-btn');
    
    confirmMessage.textContent = message;
    confirmModal.classList.add('open');
    
    // Clear old event listeners to prevent duplicate triggers
    const newSaveBtn = confirmSaveBtn.cloneNode(true);
    confirmSaveBtn.parentNode.replaceChild(newSaveBtn, confirmSaveBtn);
    
    const newCancelBtn = confirmCancelBtn.cloneNode(true);
    confirmCancelBtn.parentNode.replaceChild(newCancelBtn, confirmCancelBtn);
    
    newSaveBtn.addEventListener('click', () => {
        confirmModal.classList.remove('open');
        onConfirm();
    });
    
    newCancelBtn.addEventListener('click', () => {
        confirmModal.classList.remove('open');
    });
}

// Remove Source endpoint call and UI updates
window.removeSource = function(event, docName) {
    event.stopPropagation();
    const currentThreadId = activeThreadId || 'default';
    
    showConfirmModal(`Remove ${docName}? This cannot be undone.`, async () => {
        // Find the docItem in the DOM to show error or delete it
        const docItems = document.querySelectorAll('.kb-file-item');
        let targetDocItem = null;
        for (let item of docItems) {
            const span = item.querySelector('span');
            if (span && span.textContent.trim().startsWith(docName)) {
                targetDocItem = item;
                break;
            }
        }
        
        try {
            const response = await fetch(`/api/files/${encodeURIComponent(docName)}?thread_id=${encodeURIComponent(currentThreadId)}`, {
                method: 'DELETE'
            });
            
            const resData = await response.json();
            if (response.ok && resData.success) {
                // Clear active upload status text if it refers to the deleted file
                if (uploadStatus && uploadStatus.textContent.includes(docName)) {
                    uploadStatus.textContent = '';
                    uploadStatus.className = 'upload-status';
                }
                // Remove the item from UI immediately without page reload
                if (targetDocItem) {
                    targetDocItem.remove();
                    // If no items are left, hide the section
                    const remainingItems = document.querySelectorAll('.kb-file-item');
                    if (remainingItems.length === 0) {
                        const kbFilesSection = document.getElementById('kb-files-section');
                        if (kbFilesSection) kbFilesSection.style.display = 'none';
                    }
                }
            } else {
                throw new Error(resData.detail || resData.message || 'Failed to remove source');
            }
        } catch (err) {
            console.error("Error removing file:", err);
            if (targetDocItem) {
                // Remove any existing error message first
                const oldErr = targetDocItem.querySelector('.kb-file-error');
                if (oldErr) oldErr.remove();
                
                const errSpan = document.createElement('span');
                errSpan.className = 'kb-file-error';
                errSpan.style.color = 'var(--accent-red)';
                errSpan.style.fontSize = '10px';
                errSpan.style.marginLeft = '8px';
                errSpan.textContent = ' Error';
                errSpan.title = err.message;
                
                const btn = targetDocItem.querySelector('.kb-file-remove-btn');
                if (btn) {
                    targetDocItem.insertBefore(errSpan, btn);
                } else {
                    targetDocItem.appendChild(errSpan);
                }
                
                setTimeout(() => {
                    errSpan.remove();
                }, 4000);
            }
        }
    });
};
