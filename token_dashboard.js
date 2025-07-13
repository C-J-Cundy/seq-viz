// WebSocket connection
let ws = null;
let reconnectInterval = null;

// Connection status element
const connectionStatus = document.getElementById('connection-status');

// Metric elements
const lossElement = document.getElementById('loss-value');
const perplexityElement = document.getElementById('perplexity-value');
const stepElement = document.getElementById('step-value');

// Sequences wrapper
const sequencesWrapper = document.getElementById('sequences-wrapper');

// Store current data and state
let currentData = null;
let isPaused = false;

// History for graphs
const history = {
    perplexity: [],
    loss: [],
    maxPoints: 100
};

function connectWebSocket() {
    ws = new WebSocket('ws://localhost:8765');
    
    ws.onopen = () => {
        console.log('Connected to transformer server');
        connectionStatus.textContent = 'Connected';
        connectionStatus.classList.remove('disconnected');
        connectionStatus.classList.add('connected');
        
        if (reconnectInterval) {
            clearInterval(reconnectInterval);
            reconnectInterval = null;
        }
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'training_update') {
            updateDashboard(data);
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
        console.log('Disconnected from server');
        connectionStatus.textContent = 'Disconnected';
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

function createEntropyClock(entropy, maxEntropy) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 140 50');
    
    // Rectangular clock face
    const face = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    face.setAttribute('x', '5');
    face.setAttribute('y', '5');
    face.setAttribute('width', '130');
    face.setAttribute('height', '40');
    face.setAttribute('rx', '4');
    face.setAttribute('class', 'entropy-clock-face');
    svg.appendChild(face);
    
    // Calculate position along rectangle (0 = left, max = right)
    const progress = entropy / maxEntropy;
    const handX = 10 + (progress * 120); // 10px padding, 120px width
    
    // Clock hand (vertical line)
    const hand = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    hand.setAttribute('x1', handX);
    hand.setAttribute('y1', '10');
    hand.setAttribute('x2', handX);
    hand.setAttribute('y2', '40');
    hand.setAttribute('class', 'entropy-clock-hand');
    svg.appendChild(hand);
    
    // Center dot on hand
    const center = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    center.setAttribute('cx', handX);
    center.setAttribute('cy', '25');
    center.setAttribute('r', '3');
    center.setAttribute('class', 'entropy-clock-center');
    svg.appendChild(center);
    
    return svg;
}

function createMiniChart(top20Probs, top20Tokens, targetToken) {
    const chart = document.createElement('div');
    chart.className = 'mini-chart';
    
    // Find max probability for scaling
    const maxProb = Math.max(...top20Probs);
    
    top20Probs.forEach((prob, idx) => {
        const bar = document.createElement('div');
        bar.className = 'mini-bar';
        
        // Check if this is the ground truth
        if (top20Tokens[idx] === targetToken) {
            bar.classList.add('ground-truth');
        }
        
        // Set height based on probability
        bar.style.height = `${(prob / maxProb) * 100}%`;
        bar.title = `${(prob * 100).toFixed(1)}%`;
        
        chart.appendChild(bar);
    });
    
    return chart;
}

function updateGraph(canvasId, data, color) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (data.length < 2) return;
    
    // Find min and max values
    const minVal = Math.min(...data);
    const maxVal = Math.max(...data);
    const range = maxVal - minVal || 1;
    
    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = (i / 4) * canvas.height;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
    
    // Draw data line
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    data.forEach((value, index) => {
        const x = (index / (data.length - 1)) * canvas.width;
        const y = canvas.height - ((value - minVal) / range) * canvas.height;
        
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    
    ctx.stroke();
    
    // Draw latest value
    if (data.length > 0) {
        const latestValue = data[data.length - 1];
        ctx.fillStyle = color;
        ctx.font = '11px JetBrains Mono';
        ctx.textAlign = 'right';
        ctx.fillText(latestValue.toFixed(2), canvas.width - 5, 15);
    }
}

function updateDashboard(data) {
    if (isPaused) {
        // Store the data but don't update the display
        currentData = data;
        return;
    }
    
    currentData = data;
    
    // Update metrics
    lossElement.textContent = data.loss.toFixed(4);
    perplexityElement.textContent = data.perplexity.toFixed(2);
    stepElement.textContent = data.step;
    
    // Update history
    history.perplexity.push(data.perplexity);
    history.loss.push(data.loss);
    
    // Keep history size limited
    if (history.perplexity.length > history.maxPoints) {
        history.perplexity.shift();
        history.loss.shift();
    }
    
    // Update graphs
    updateGraph('perplexity-graph', history.perplexity, '#a78bfa');
    updateGraph('loss-graph', history.loss, '#60a5fa');
    
    // Update sequences
    updateSequences(data.sequences);
}

function updateSequences(sequences) {
    // Clear existing sequences
    sequencesWrapper.innerHTML = '';
    
    // Create a container for each sequence in the batch
    sequences.forEach((seq, seqIdx) => {
        const sequenceGroup = document.createElement('div');
        sequenceGroup.className = 'sequence-group';
        
        // No sequence labels
        
        // Create sequence container
        const sequenceContainer = document.createElement('div');
        sequenceContainer.className = 'sequence-container';
        
        const sequenceScroll = document.createElement('div');
        sequenceScroll.className = 'sequence-scroll';
        
        // Process tokens and predictions
        for (let i = 0; i < seq.tokens.length; i++) {
            // Get prediction from previous position (or null for first token)
            const prediction = i > 0 ? seq.predictions[i - 1] : null;
            const tokenColumn = createTokenColumn(seq.tokens[i], prediction, i);
            sequenceScroll.appendChild(tokenColumn);
        }
        
        sequenceContainer.appendChild(sequenceScroll);
        sequenceGroup.appendChild(sequenceContainer);
        sequencesWrapper.appendChild(sequenceGroup);
    });
}

function createTokenColumn(token, prediction, index) {
    const column = document.createElement('div');
    column.className = 'token-column';
    
    // Current token
    const tokenElement = document.createElement('div');
    tokenElement.className = 'current-token';
    tokenElement.textContent = token || '�';
    column.appendChild(tokenElement);
    
    // Predictions container (shows predictions FROM the previous token)
    if (prediction && prediction.top_k_strings) {
        const predictionsContainer = document.createElement('div');
        predictionsContainer.className = 'predictions-container';
        
        // Add entropy box with clock and mini chart
        if (prediction.full_entropy !== undefined && prediction.top_20_probs) {
            const entropyBox = document.createElement('div');
            entropyBox.className = 'entropy-box';
            entropyBox.title = `Entropy: H=${prediction.full_entropy.toFixed(3)}`;
            
            // Entropy clock
            const clockContainer = document.createElement('div');
            clockContainer.className = 'entropy-clock';
            const maxEntropy = Math.log2(prediction.vocab_size || 128256); // Llama vocab size
            clockContainer.appendChild(createEntropyClock(prediction.full_entropy, maxEntropy));
            entropyBox.appendChild(clockContainer);
            
            // Mini chart overlay
            const miniChart = createMiniChart(
                prediction.top_20_probs,
                prediction.top_20_tokens,
                prediction.target
            );
            entropyBox.appendChild(miniChart);
            
            predictionsContainer.appendChild(entropyBox);
        }
        
        // Sort predictions by probability
        const sortedPredictions = prediction.top_k_strings.map((token, idx) => ({
            token: token,
            prob: prediction.top_k_probs[idx],
            tokenId: prediction.top_k_tokens[idx]
        })).sort((a, b) => b.prob - a.prob);
        
        // Find max probability for scaling bars
        const maxProb = Math.max(...prediction.top_k_probs);
        
        sortedPredictions.forEach((pred, idx) => {
            const predItem = document.createElement('div');
            predItem.className = 'prediction-item';
            
            // Check if this is the ground truth (what token actually came next)
            const isGroundTruth = pred.tokenId === prediction.target;
            if (isGroundTruth) {
                predItem.classList.add('ground-truth');
            }
            
            // Probability bar background
            const barBg = document.createElement('div');
            barBg.className = 'prediction-bar-bg';
            barBg.style.width = `${(pred.prob / maxProb) * 100}%`;
            predItem.appendChild(barBg);
            
            // Content container
            const content = document.createElement('div');
            content.className = 'prediction-content';
            
            const predToken = document.createElement('span');
            predToken.className = 'prediction-token';
            predToken.textContent = pred.token || '�';
            
            const predProb = document.createElement('span');
            predProb.className = 'prediction-prob';
            predProb.textContent = `${(pred.prob * 100).toFixed(1)}%`;
            
            content.appendChild(predToken);
            content.appendChild(predProb);
            predItem.appendChild(content);
            
            predictionsContainer.appendChild(predItem);
        });
        
        column.appendChild(predictionsContainer);
    }
    
    return column;
}

// Control button functionality
const controlButton = document.getElementById('control-button');
const controlIcon = document.getElementById('control-icon');

// Initialize connection
connectWebSocket();

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, close connection to save resources
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.close();
        }
    } else {
        // Page is visible again, reconnect if needed
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            connectWebSocket();
        }
    }
});

function togglePause() {
    isPaused = !isPaused;
    
    if (isPaused) {
        controlButton.classList.add('paused');
        controlIcon.textContent = '▶';
        controlButton.title = 'Resume (P)';
    } else {
        controlButton.classList.remove('paused');
        controlIcon.textContent = '⏸';
        controlButton.title = 'Pause (P)';
        
        // Update with latest data if available
        if (currentData) {
            updateDashboard(currentData);
        }
    }
}

controlButton.addEventListener('click', togglePause);

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'p' || e.key === 'P') {
        togglePause();
    }
});