document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const dropzone = document.getElementById('dropzone');
    const uploadStatus = document.getElementById('uploadStatus');
    
    const questionInput = document.getElementById('questionInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatHistory = document.getElementById('chatHistory');

    // --- Auto-resize textarea ---
    questionInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if(this.value.trim() === '') {
            sendBtn.disabled = true;
        } else {
            sendBtn.disabled = false;
        }
    });

    questionInput.addEventListener('keydown', (e) => {
        if(e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.disabled = true;
    sendBtn.addEventListener('click', sendMessage);

    // --- Drag and drop handling ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    // --- File Upload Logic ---
    async function handleFiles(files) {
        if (!files || files.length === 0) return;
        
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }

        uploadStatus.className = 'upload-status loading';
        uploadStatus.innerText = 'Uploading and processing documents...';

        try {
            const response = await fetch('/api/v1/rag/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                uploadStatus.className = 'upload-status success';
                uploadStatus.innerText = `Success! Processed ${data.total_chunks || 'multiple'} chunks.`;
                setTimeout(() => uploadStatus.innerText = '', 5000);
            } else {
                throw new Error(data.detail || 'Upload failed');
            }
        } catch (error) {
            uploadStatus.className = 'upload-status error';
            uploadStatus.innerText = `Error: ${error.message}`;
        }
        
        // Reset file input
        fileInput.value = '';
    }

    // --- Chat Logic ---
    async function sendMessage() {
        const text = questionInput.value.trim();
        if(!text) return;

        // Add user message
        appendMessage(text, 'user');
        
        // Clear input
        questionInput.value = '';
        questionInput.style.height = 'auto';
        sendBtn.disabled = true;

        // Add loading bot message
        const loadingId = appendLoadingMessage();

        try {
            const response = await fetch('/api/v1/rag/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: text })
            });

            const data = await response.json();
            
            // Remove loading message
            document.getElementById(loadingId).remove();

            if (response.ok) {
                appendBotMessage(data.answer, data.sources);
            } else {
                appendMessage(`Error: ${data.detail || 'Something went wrong.'}`, 'bot');
            }

        } catch (error) {
            document.getElementById(loadingId).remove();
            appendMessage(`Connection error: ${error.message}`, 'bot');
        }
    }

    function appendMessage(content, type) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        // Basic escaping (in a real app, use a proper markdown/sanitization library)
        contentDiv.textContent = content; 
        
        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
    }

    function appendBotMessage(answer, sources) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message bot`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Simple line break to <br> for basic formatting
        let formattedAnswer = answer.replace(/\n/g, '<br>');
        contentDiv.innerHTML = formattedAnswer;
        
        // Append sources if available
        if (sources && sources.length > 0) {
            const sourcesContainer = document.createElement('div');
            sourcesContainer.className = 'sources-container';
            
            // Limit to top 2 sources for UI neatness
            const topSources = sources.slice(0, 2);
            
            topSources.forEach((src, idx) => {
                const srcItem = document.createElement('div');
                srcItem.className = 'source-item';
                
                const meta = src.metadata?.source || `Source ${idx+1}`;
                const filename = meta.split(/[\\/]/).pop();
                
                srcItem.innerHTML = `<strong>${filename}</strong> ${src.content.substring(0, 100)}...`;
                sourcesContainer.appendChild(srcItem);
            });
            
            contentDiv.appendChild(sourcesContainer);
        }

        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
    }

    function appendLoadingMessage() {
        const id = 'loading-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.className = `message bot`;
        msgDiv.id = id;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = '<span style="animation: pulse 1s infinite alternate; opacity: 0.5;">Thinking...</span>';
        
        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        scrollToBottom();
        return id;
    }

    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});
