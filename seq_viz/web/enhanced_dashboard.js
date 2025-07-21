// WebSocket connection
let ws = null;
let reconnectInterval = null;

// DOM elements
const connectionStatus = document.getElementById('connection-status');
const statusText = document.getElementById('status-text');
const lossElement = document.getElementById('loss-value');
const perplexityElement = document.getElementById('perplexity-value');
const stepElement = document.getElementById('step-value');
const sequencesContainer = document.getElementById('sequences-container');

// State
let currentData = null;
let selectedSequence = 0;
let selectedPosition = 0;

// History for graphs
const history = {
    loss: [],
    perplexity: [],
    maxPoints: 50
};

// Connect to WebSocket server
function connectWebSocket() {
    // Use relative WebSocket URL - defaults to same host as page
    const wsUrl = window.location.hostname === 'file' || window.location.protocol === 'file:' 
        ? 'ws://localhost:8765' 
        : `ws://${window.location.hostname}:8765`;
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('Connected to server');
        statusText.textContent = 'Connected';
        connectionStatus.classList.remove('disconnected');
        connectionStatus.classList.add('connected');
        
        if (reconnectInterval) {
            clearInterval(reconnectInterval);
            reconnectInterval = null;
        }
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'training_update' || data.step !== undefined) {
                currentData = data;
                updateDashboard(data);
            }
        } catch (e) {
            console.error('Error parsing message:', e);
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
        console.log('Disconnected from server');
        statusText.textContent = 'Disconnected';
        connectionStatus.classList.remove('connected');
        connectionStatus.classList.add('disconnected');
        
        // Attempt to reconnect every 3 seconds
        if (!reconnectInterval) {
            reconnectInterval = setInterval(() => {
                console.log('Attempting to reconnect...');
                connectWebSocket();
            }, 3000);
        }
    };
}


// Create mini probability chart
function createMiniChart(predictions, targetTokenId) {
    const chart = document.createElement('div');
    chart.className = 'mini-chart';
    
    // Get top 20 predictions
    const top20 = predictions.top_20 || predictions.top_k || [];
    if (top20.length === 0) return chart;
    
    // Find max probability for scaling
    const maxProb = Math.max(...top20.map(p => p.prob));
    
    top20.forEach((pred) => {
        const bar = document.createElement('div');
        bar.className = 'mini-bar';
        
        // Check if this is the ground truth
        if (pred.token_id === targetTokenId) {
            bar.classList.add('ground-truth');
        }
        
        // Set height based on probability
        bar.style.height = `${(pred.prob / maxProb) * 100}%`;
        bar.title = `${pred.token_str}: ${(pred.prob * 100).toFixed(1)}%`;
        
        chart.appendChild(bar);
    });
    
    return chart;
}

// Update metric graphs
function updateGraph(canvasId, data, color = '#60a5fa') {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    if (data.length < 2) return;
    
    // Find min and max values
    const minVal = Math.min(...data);
    const maxVal = Math.max(...data);
    const range = maxVal - minVal || 1;
    
    // Draw line
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    data.forEach((value, index) => {
        const x = (index / (data.length - 1)) * width;
        const y = height - ((value - minVal) / range) * height;
        
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    
    ctx.stroke();
}

// Create token column
function createTokenColumn(token, predictions, position, isSelected) {
    const column = document.createElement('div');
    column.className = 'token-column';
    
    // Current token
    const tokenEl = document.createElement('div');
    tokenEl.className = 'current-token';
    if (isSelected) {
        tokenEl.classList.add('selected');
    }
    tokenEl.textContent = token;
    
    // Scale font size based on token length
    const baseSize = 1.1; // em
    const maxChars = 12; // characters that fit comfortably at base size
    if (token.length > maxChars) {
        const scale = Math.max(0.6, maxChars / token.length);
        tokenEl.style.fontSize = `${baseSize * scale}em`;
    }
    
    tokenEl.onclick = () => {
        selectedPosition = position;
        updateSequenceDisplay();
    };
    column.appendChild(tokenEl);
    
    // Predictions container
    const predsContainer = document.createElement('div');
    predsContainer.className = 'predictions-container';
    
    if (predictions) {
        // Entropy display
        const entropyBox = document.createElement('div');
        entropyBox.className = 'entropy-box';
        entropyBox.style.display = 'flex';
        entropyBox.style.justifyContent = 'center';
        entropyBox.style.alignItems = 'center';
        entropyBox.style.paddingBottom = '24px'; // Make room for mini chart
        
        const entropyValue = document.createElement('span');
        entropyValue.style.fontSize = '1em';
        entropyValue.style.fontWeight = '600';
        entropyValue.style.color = '#a78bfa';
        entropyValue.style.fontFamily = 'JetBrains Mono, monospace';
        entropyValue.textContent = predictions.entropy.toFixed(3);
        
        entropyBox.appendChild(entropyValue);
        entropyBox.appendChild(createMiniChart(predictions, predictions.target_token_id));
        predsContainer.appendChild(entropyBox);
        
        // Top predictions
        const topK = predictions.top_k || [];
        topK.forEach((pred, idx) => {
            const item = document.createElement('div');
            item.className = 'prediction-item';
            
            // Check if this is the ground truth
            if (pred.token_id === predictions.target_token_id) {
                item.classList.add('ground-truth');
            }
            
            // Background bar
            const barBg = document.createElement('div');
            barBg.className = 'prediction-bar-bg';
            barBg.style.width = `${pred.prob * 100}%`;
            item.appendChild(barBg);
            
            // Content
            const content = document.createElement('div');
            content.className = 'prediction-content';
            
            const rank = document.createElement('span');
            rank.className = 'prediction-rank';
            rank.textContent = `${idx + 1}.`;
            content.appendChild(rank);
            
            const tokenSpan = document.createElement('span');
            tokenSpan.className = 'prediction-token';
            tokenSpan.textContent = pred.token_str;
            content.appendChild(tokenSpan);
            
            const probSpan = document.createElement('span');
            probSpan.className = 'prediction-prob';
            probSpan.textContent = `${(pred.prob * 100).toFixed(1)}%`;
            content.appendChild(probSpan);
            
            item.appendChild(content);
            predsContainer.appendChild(item);
        });
    }
    
    column.appendChild(predsContainer);
    return column;
}

// Update sequence display
function updateSequenceDisplay() {
    if (!currentData || !currentData.sequences) return;
    
    const sequence = currentData.sequences[selectedSequence];
    if (!sequence) return;
    
    // Find or create sequence container
    let seqContainer = document.querySelector('.sequence-container');
    if (!seqContainer) {
        seqContainer = document.createElement('div');
        seqContainer.className = 'sequence-container';
        sequencesContainer.appendChild(seqContainer);
    }
    
    // Clear and rebuild
    seqContainer.innerHTML = '';
    
    // Header
    const header = document.createElement('div');
    header.className = 'sequence-header';
    
    const title = document.createElement('div');
    title.className = 'sequence-title';
    title.textContent = `Sequence ${selectedSequence + 1} of ${currentData.sequences.length}`;
    header.appendChild(title);
    
    // Sequence controls
    if (currentData.sequences.length > 1) {
        const controls = document.createElement('div');
        controls.className = 'sequence-controls';
        
        for (let i = 0; i < currentData.sequences.length; i++) {
            const btn = document.createElement('button');
            btn.className = 'sequence-button';
            if (i === selectedSequence) {
                btn.classList.add('active');
            }
            btn.textContent = `Seq ${i + 1}`;
            btn.onclick = () => {
                selectedSequence = i;
                selectedPosition = 0;
                updateSequenceDisplay();
            };
            controls.appendChild(btn);
        }
        
        header.appendChild(controls);
    }
    
    seqContainer.appendChild(header);
    
    // Token columns
    const scroll = document.createElement('div');
    scroll.className = 'sequence-scroll';
    
    sequence.tokens.forEach((token, idx) => {
        // Shift predictions by 1: prediction at position i-1 predicts token at position i
        // First token has no predictions (no previous token to predict from)
        let predictions = null;
        if (idx > 0 && sequence.predictions && sequence.predictions[idx - 1]) {
            predictions = sequence.predictions[idx - 1];
        }
        const column = createTokenColumn(token, predictions, idx, idx === selectedPosition);
        scroll.appendChild(column);
    });
    
    seqContainer.appendChild(scroll);
}

// Update dashboard with new data
function updateDashboard(data) {
    // Update metrics
    stepElement.textContent = data.step || '-';
    
    if (data.loss !== undefined) {
        lossElement.textContent = data.loss.toFixed(4);
        history.loss.push(data.loss);
        if (history.loss.length > history.maxPoints) {
            history.loss.shift();
        }
        updateGraph('loss-graph', history.loss, '#ef4444');
    }
    
    if (data.perplexity !== undefined) {
        perplexityElement.textContent = data.perplexity.toFixed(2);
        history.perplexity.push(data.perplexity);
        if (history.perplexity.length > history.maxPoints) {
            history.perplexity.shift();
        }
        updateGraph('perplexity-graph', history.perplexity, '#22c55e');
    }
    
    // Update sequences
    updateSequenceDisplay();
}

// Initialize
connectWebSocket();