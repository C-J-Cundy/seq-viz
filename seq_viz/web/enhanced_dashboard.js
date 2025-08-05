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

// Create floating particles
function createParticles() {
    const particlesContainer = document.getElementById('particles');
    const colors = ['cyan', 'magenta', 'green'];
    
    // Create 30 particles
    for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        particle.className = `particle ${colors[Math.floor(Math.random() * colors.length)]}`;
        
        // Random horizontal position
        particle.style.left = Math.random() * 100 + '%';
        
        // Random animation delay
        particle.style.animationDelay = Math.random() * 15 + 's';
        
        // Random animation duration (12-18s)
        particle.style.animationDuration = (12 + Math.random() * 6) + 's';
        
        particlesContainer.appendChild(particle);
    }
}

// Initialize particles on load
window.addEventListener('DOMContentLoaded', createParticles);

// State
let currentData = null;
let selectedSequence = 0;
let selectedPosition = 0;

// History for graphs
const history = {
    loss: [],
    perplexity: [],
    entropy: [],
    maxPoints: 100,  // More points for larger plots
    timestamps: []
};

// Plot interaction state
const plotState = {
    lossPoints: [],
    entropyPoints: [],
    tooltip: null
};

// Connect to WebSocket server
function connectWebSocket() {
    // Don't create a new connection if one already exists
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
        return;
    }
    
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
    // Debug: log only if we're getting unexpected data
    if (predictions.top_20 && predictions.top_20.length !== 20) {
        console.warn('Unexpected top_20 length:', predictions.top_20.length, 'at position', predictions.position);
    }
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

// Calculate simple moving average
function calculateSMA(data, window = 5) {
    const sma = [];
    for (let i = 0; i < data.length; i++) {
        const start = Math.max(0, i - window + 1);
        const subset = data.slice(start, i + 1);
        const avg = subset.reduce((a, b) => a + b, 0) / subset.length;
        sma.push(avg);
    }
    return sma;
}

// Draw minimalist background plot
function drawMinimalistPlot(canvasId, data, color = '#00D4FF', options = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = 60; // Generous padding
    const plotWidth = width - padding * 2;
    const plotHeight = height - padding * 2;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    if (data.length < 2) return;
    
    // Find min and max values
    const minVal = Math.min(...data);
    const maxVal = Math.max(...data);
    const range = maxVal - minVal || 1;
    
    // Add some margin to the range for visual appeal
    const margin = range * 0.1;
    const adjustedMin = minVal - margin;
    const adjustedMax = maxVal + margin;
    const adjustedRange = adjustedMax - adjustedMin;
    
    // Store points for interaction
    const points = [];
    data.forEach((value, index) => {
        const x = padding + (index / (data.length - 1)) * plotWidth;
        const y = height - padding - ((value - adjustedMin) / adjustedRange) * plotHeight;
        points.push({ x, y, value, index, step: history.timestamps[index] || index });
    });
    
    // Store points based on plot type
    if (options.isLoss) {
        plotState.lossPoints = points;
    } else {
        plotState.entropyPoints = points;
    }
    
    // Common smoothing for both plots
    const smoothed = calculateSMA(data, Math.max(5, Math.floor(data.length / 20)));
    
    // Special handling for loss plot
    if (options.isLoss) {
        // Draw faint raw data line
        ctx.strokeStyle = '#00D4FF33'; // Very faint neon blue
        ctx.lineWidth = 1;
        ctx.beginPath();
        
        data.forEach((value, index) => {
            const x = padding + (index / (data.length - 1)) * plotWidth;
            const y = height - padding - ((value - adjustedMin) / adjustedRange) * plotHeight;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Draw the smoothed line with glow
        ctx.shadowColor = '#00D4FF';
        ctx.shadowBlur = 15;
        ctx.strokeStyle = '#00D4FF'; // Bright neon blue
        ctx.lineWidth = 2.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        
        smoothed.forEach((value, index) => {
            const x = padding + (index / (smoothed.length - 1)) * plotWidth;
            const y = height - padding - ((value - adjustedMin) / adjustedRange) * plotHeight;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        ctx.shadowBlur = 0;
        
        // Draw subtle dots at data points (like entropy plot)
        ctx.fillStyle = '#00D4FF';
        const dotInterval = Math.max(1, Math.floor(data.length / 20)); // Show max 20 dots
        
        data.forEach((value, index) => {
            if (index % dotInterval === 0 || index === data.length - 1) {
                const x = padding + (index / (data.length - 1)) * plotWidth;
                const y = height - padding - ((value - adjustedMin) / adjustedRange) * plotHeight;
                
                // Outer glow
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI * 2);
                ctx.fillStyle = '#00D4FF44';
                ctx.fill();
                
                // Inner dot
                ctx.beginPath();
                ctx.arc(x, y, 2, 0, Math.PI * 2);
                ctx.fillStyle = '#00D4FF';
                ctx.fill();
            }
        });
        
    } else {
        // For entropy plot - draw faint raw data first
        ctx.strokeStyle = `${color}33`; // Very faint
        ctx.lineWidth = 1;
        ctx.beginPath();
        
        data.forEach((value, index) => {
            const x = padding + (index / (data.length - 1)) * plotWidth;
            const y = height - padding - ((value - adjustedMin) / adjustedRange) * plotHeight;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Draw the smoothed line with glow
        ctx.shadowColor = color;
        ctx.shadowBlur = 15;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        
        smoothed.forEach((value, index) => {
            const x = padding + (index / (smoothed.length - 1)) * plotWidth;
            const y = height - padding - ((value - adjustedMin) / adjustedRange) * plotHeight;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        ctx.shadowBlur = 0;
        
        // Draw subtle dots at data points
        ctx.fillStyle = color;
        const dotInterval = Math.max(1, Math.floor(data.length / 20)); // Show max 20 dots
        
        data.forEach((value, index) => {
            if (index % dotInterval === 0 || index === data.length - 1) {
                const x = padding + (index / (data.length - 1)) * plotWidth;
                const y = height - padding - ((value - adjustedMin) / adjustedRange) * plotHeight;
                
                // Outer glow
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI * 2);
                ctx.fillStyle = `${color}44`;
                ctx.fill();
                
                // Inner dot
                ctx.beginPath();
                ctx.arc(x, y, 2, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
            }
        });
    }
}

// Animation state for Joy plot
const joyPlotAnimation = {
    previousData: null,
    currentData: null,
    animationProgress: 0,
    animationId: null,
    startTime: null
};

// Interpolate between two loss values with spatial wave effect
function interpolateLoss(oldLoss, newLoss, progress, position, totalPositions) {
    // Create a sigmoid wave that moves from left to right
    const wavePosition = progress * 1.4 - 0.2; // Wave starts off-screen left, ends off-screen right
    const normalizedPosition = position / totalPositions;
    
    // Sigmoid function centered at the wave position
    const waveWidth = 0.2; // Controls how sharp the transition is
    const distance = normalizedPosition - wavePosition;
    const sigmoid = 1 / (1 + Math.exp(-distance / waveWidth * 10));
    
    // Interpolate based on the sigmoid value
    return oldLoss + (newLoss - oldLoss) * sigmoid;
}

// Draw Joy Division style sparkline plot
function drawJoyPlotSparkline(canvas, allSequences, selectedSeqIndex, animate = true) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Scale for device pixel ratio
    const dpr = window.devicePixelRatio || 1;
    ctx.scale(dpr, dpr);
    
    // Use actual dimensions for drawing
    const drawWidth = width / dpr;
    const drawHeight = height / dpr;
    
    // Calculate layout for multiple sequences
    const padding = 40;
    const plotWidth = drawWidth - padding * 2;
    const sequenceCount = allSequences.length;
    const sequenceHeight = (drawHeight - padding * 2) / sequenceCount;
    const overlap = sequenceHeight * 0.3; // 30% overlap for Joy Division effect
    
    // Find global min/max across all sequences for consistent scaling
    let globalMinLoss = Infinity;
    let globalMaxLoss = -Infinity;
    
    allSequences.forEach(sequence => {
        if (sequence.predictions) {
            sequence.predictions.forEach(pred => {
                const loss = pred.loss || 0;
                globalMinLoss = Math.min(globalMinLoss, loss);
                globalMaxLoss = Math.max(globalMaxLoss, loss);
            });
        }
    });
    
    const globalRange = globalMaxLoss - globalMinLoss || 1;
    
    // Draw sequences from back to front (reverse order)
    for (let seqIdx = sequenceCount - 1; seqIdx >= 0; seqIdx--) {
        const sequence = allSequences[seqIdx];
        if (!sequence.predictions || sequence.predictions.length === 0) continue;
        
        // Extract loss values
        let losses = sequence.predictions.map(pred => pred.loss || 0);
        
        // Apply animation if we have previous data
        if (animate && joyPlotAnimation.previousData && 
            joyPlotAnimation.animationProgress < 1 &&
            joyPlotAnimation.previousData[seqIdx] &&
            joyPlotAnimation.previousData[seqIdx].predictions) {
            
            const oldLosses = joyPlotAnimation.previousData[seqIdx].predictions.map(pred => pred.loss || 0);
            
            
            // Interpolate between old and new losses with wave effect
            losses = losses.map((newLoss, idx) => {
                if (idx < oldLosses.length) {
                    return interpolateLoss(oldLosses[idx], newLoss, 
                                         joyPlotAnimation.animationProgress, 
                                         idx, losses.length - 1);
                }
                return newLoss;
            });
        }
        
        // Calculate vertical position
        const baseY = padding + seqIdx * (sequenceHeight - overlap);
        const waveHeight = sequenceHeight * 0.8; // Use 80% of allocated height for the wave
        
        // Set style based on whether this is the selected sequence
        const isSelected = seqIdx === selectedSeqIndex;
        ctx.strokeStyle = isSelected ? '#00D4FF' : '#00D4FF66';
        ctx.lineWidth = isSelected ? 2.5 : 1.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        // Add glow for selected sequence
        if (isSelected) {
            ctx.shadowColor = '#00D4FF';
            ctx.shadowBlur = 15;
        } else {
            ctx.shadowColor = '#00D4FF';
            ctx.shadowBlur = 5;
        }
        
        // Create the path
        ctx.beginPath();
        
        losses.forEach((loss, index) => {
            const x = padding + (index / (losses.length - 1)) * plotWidth;
            const normalizedLoss = (loss - globalMinLoss) / globalRange;
            const y = baseY + waveHeight - (normalizedLoss * waveHeight);
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Fill area under the curve - extend to next sequence
        ctx.shadowBlur = 0;
        
        // Calculate where the fill should extend to
        // Use consistent gradient height for all sequences
        const gradientHeight = sequenceHeight * 1.5; // Extend gradient 50% beyond the wave height
        const fillBottomY = baseY + gradientHeight;
        
        // Create gradient that fades as it goes down
        const gradient = ctx.createLinearGradient(0, baseY, 0, fillBottomY);
        if (isSelected) {
            gradient.addColorStop(0, 'rgba(0, 212, 255, 0.3)');
            gradient.addColorStop(0.3, 'rgba(0, 212, 255, 0.15)');
            gradient.addColorStop(0.6, 'rgba(0, 212, 255, 0.05)');
            gradient.addColorStop(0.85, 'rgba(0, 212, 255, 0.01)');
            gradient.addColorStop(1, 'rgba(0, 212, 255, 0)');
        } else {
            gradient.addColorStop(0, 'rgba(0, 212, 255, 0.15)');
            gradient.addColorStop(0.3, 'rgba(0, 212, 255, 0.08)');
            gradient.addColorStop(0.6, 'rgba(0, 212, 255, 0.02)');
            gradient.addColorStop(0.85, 'rgba(0, 212, 255, 0.005)');
            gradient.addColorStop(1, 'rgba(0, 212, 255, 0)');
        }
        
        ctx.fillStyle = gradient;
        ctx.lineTo(drawWidth - padding, fillBottomY);
        ctx.lineTo(padding, fillBottomY);
        ctx.closePath();
        ctx.fill();
    }
    
    // Reset transform
    ctx.setTransform(1, 0, 0, 1, 0, 0);
}

// Animate Joy plot when new data arrives
function animateJoyPlot(canvas, newSequences, selectedSeqIndex) {
    // Cancel any existing animation
    if (joyPlotAnimation.animationId) {
        cancelAnimationFrame(joyPlotAnimation.animationId);
    }
    
    // Store the new data as target
    joyPlotAnimation.currentData = newSequences;
    joyPlotAnimation.animationProgress = 0;
    joyPlotAnimation.startTime = performance.now();
    
    const duration = 3000; // 3 second animation - much longer to see the effect
    
    function animate(currentTime) {
        const elapsed = currentTime - joyPlotAnimation.startTime;
        joyPlotAnimation.animationProgress = Math.min(elapsed / duration, 1);
        
        // Draw with current animation progress
        drawJoyPlotSparkline(canvas, newSequences, selectedSeqIndex, true);
        
        if (joyPlotAnimation.animationProgress < 1) {
            joyPlotAnimation.animationId = requestAnimationFrame(animate);
        } else {
            // Animation complete, update previous data for next time
            joyPlotAnimation.previousData = JSON.parse(JSON.stringify(newSequences));
            joyPlotAnimation.animationId = null;
        }
    }
    
    joyPlotAnimation.animationId = requestAnimationFrame(animate);
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
    
    // Check if this token was predicted with high probability
    // Look at the previous position's predictions
    if (position > 0 && currentData && currentData.sequences[selectedSequence]) {
        const prevPredictions = currentData.sequences[selectedSequence].predictions[position - 1];
        if (prevPredictions && prevPredictions.top_k[0]) {
            const topPrediction = prevPredictions.top_k[0];
            // If this token was the top prediction with > 50% probability, mark it as high-prob
            if (topPrediction.token_str === token && topPrediction.prob > 0.5) {
                tokenEl.classList.add('high-prob');
            }
        }
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
        // Entropy display with mini chart
        const entropyBox = document.createElement('div');
        entropyBox.className = 'entropy-box';
        entropyBox.style.position = 'relative';
        entropyBox.style.paddingTop = '20px'; // Space for entropy value
        
        const entropyValue = document.createElement('div');
        entropyValue.style.position = 'absolute';
        entropyValue.style.top = '4px';
        entropyValue.style.left = '0';
        entropyValue.style.right = '0';
        entropyValue.style.textAlign = 'center';
        entropyValue.style.fontSize = '1em';
        entropyValue.style.fontWeight = '600';
        entropyValue.style.fontFamily = 'Share Tech Mono, monospace';
        
        // Create loss and entropy display
        const lossSpan = document.createElement('span');
        lossSpan.style.color = '#FFB800';
        lossSpan.style.textShadow = '0 0 5px rgba(255, 184, 0, 0.5)';
        lossSpan.textContent = predictions.loss ? predictions.loss.toFixed(2) : '0.00';
        
        const separator = document.createElement('span');
        separator.style.color = '#666';
        separator.style.margin = '0 6px';
        separator.textContent = '|';
        
        const entropySpan = document.createElement('span');
        entropySpan.style.color = '#FF0080';
        entropySpan.style.textShadow = '0 0 5px rgba(255, 0, 128, 0.5)';
        entropySpan.textContent = predictions.entropy.toFixed(2);
        
        entropyValue.appendChild(lossSpan);
        entropyValue.appendChild(separator);
        entropyValue.appendChild(entropySpan);
        
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
            
            // Add data stream effect for high probability predictions
            if (pred.prob > 0.7 && idx === 0) {
                const stream = document.createElement('div');
                stream.className = 'data-stream';
                stream.style.position = 'absolute';
                stream.style.left = '10px';
                stream.style.top = '100%';
                stream.style.animationDelay = Math.random() * 2 + 's';
                item.appendChild(stream);
                item.style.position = 'relative';
                item.style.overflow = 'visible';
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
function updateSequenceDisplay(animateJoy = false, previousSequences = null) {
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
    
    // Keep Joy plot canvas if it exists
    const existingJoyCanvas = document.getElementById('joy-plot-canvas');
    
    // Clear and rebuild (but preserve Joy plot)
    seqContainer.innerHTML = '';
    
    // Header
    const header = document.createElement('div');
    header.className = 'sequence-header';
    
    const title = document.createElement('div');
    title.className = 'sequence-title';
    title.textContent = `${selectedSequence + 1}/${currentData.sequences.length}`;
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
    
    // Joy plot sparkline - fixed at bottom of screen
    let sparklineContainer = document.getElementById('joy-plot-container');
    let sparklineCanvas = document.getElementById('joy-plot-canvas');
    
    if (!sparklineContainer) {
        sparklineContainer = document.createElement('div');
        sparklineContainer.id = 'joy-plot-container';
        
        sparklineCanvas = document.createElement('canvas');
        sparklineCanvas.id = 'joy-plot-canvas';
        sparklineCanvas.style.width = '100%';
        sparklineCanvas.style.height = '100%';
        sparklineCanvas.style.display = 'block';
        sparklineCanvas.style.pointerEvents = 'none'; // Allow clicks to pass through
        
        sparklineContainer.appendChild(sparklineCanvas);
        // Append to body instead of sequence container so it stays fixed
        document.body.appendChild(sparklineContainer);
    }
    
    // Set canvas resolution (will be adjusted on resize)
    const updateCanvasSize = (animate = false) => {
        const rect = sparklineContainer.getBoundingClientRect();
        sparklineCanvas.width = rect.width * window.devicePixelRatio;
        sparklineCanvas.height = 300 * window.devicePixelRatio;
        
        // Use animation for data updates, but not for resize
        if (animate && previousSequences) {
            // Set previous data before animating
            joyPlotAnimation.previousData = previousSequences;
            animateJoyPlot(sparklineCanvas, currentData.sequences, selectedSequence);
        } else {
            drawJoyPlotSparkline(sparklineCanvas, currentData.sequences, selectedSequence, false);
            // Initialize previous data if not set
            if (!joyPlotAnimation.previousData) {
                joyPlotAnimation.previousData = JSON.parse(JSON.stringify(currentData.sequences));
            }
        }
    };
    
    // Update canvas after DOM is ready
    setTimeout(() => {
        const rect = sparklineContainer.getBoundingClientRect();
        sparklineCanvas.width = rect.width * window.devicePixelRatio;
        sparklineCanvas.height = 300 * window.devicePixelRatio;
        
        if (animateJoy && previousSequences) {
            joyPlotAnimation.previousData = previousSequences;
            animateJoyPlot(sparklineCanvas, currentData.sequences, selectedSequence);
        } else {
            drawJoyPlotSparkline(sparklineCanvas, currentData.sequences, selectedSequence, false);
            if (!joyPlotAnimation.previousData) {
                joyPlotAnimation.previousData = JSON.parse(JSON.stringify(currentData.sequences));
            }
        }
    }, 0);
}

// Create burst of particles for data update
function createDataUpdateBurst() {
    const particlesContainer = document.getElementById('particles');
    const colors = ['cyan', 'magenta', 'green', 'yellow'];
    
    // Use DocumentFragment for better performance
    const fragment = document.createDocumentFragment();
    const burstParticles = [];
    
    // Reduce particle count and stagger creation
    const particleCount = 50; // Reduced from 100
    const batchSize = 10;
    let created = 0;
    
    function createBatch() {
        for (let i = 0; i < batchSize && created < particleCount; i++, created++) {
            const particle = document.createElement('div');
            const color = colors[Math.floor(Math.random() * colors.length)];
            particle.className = `particle ${color}`;
            
            // Add yellow style if needed
            if (color === 'yellow') {
                particle.style.background = 'radial-gradient(circle at 30% 30%, #FFD700, #FFA500)';
                particle.style.boxShadow = '0 0 10px #FFD700';
            }
            
            // Random horizontal position
            particle.style.left = Math.random() * 100 + '%';
            
            // Use CSS transform for better performance
            particle.style.transform = 'translateZ(0)'; // Enable hardware acceleration
            
            // Use same slow drift animation as regular particles
            particle.style.animationDuration = (12 + Math.random() * 6) + 's';
            particle.style.animationDelay = Math.random() * 2 + 's';
            
            fragment.appendChild(particle);
            burstParticles.push(particle);
        }
        
        // Add batch to DOM
        particlesContainer.appendChild(fragment);
        
        // Schedule next batch
        if (created < particleCount) {
            requestAnimationFrame(createBatch);
        }
    }
    
    // Start creating particles
    createBatch();
    
    // Remove burst particles after 20 seconds
    setTimeout(() => {
        // Remove in batches too
        let index = 0;
        function removeBatch() {
            for (let i = 0; i < 10 && index < burstParticles.length; i++, index++) {
                burstParticles[index].remove();
            }
            if (index < burstParticles.length) {
                requestAnimationFrame(removeBatch);
            }
        }
        removeBatch();
    }, 20000);
}

// Update dashboard with new data
function updateDashboard(data) {
    // Store previous data BEFORE updating currentData
    const hadPreviousData = currentData && currentData.sequences;
    const previousSequences = hadPreviousData ? JSON.parse(JSON.stringify(currentData.sequences)) : null;
    
    // Create particle burst to indicate update
    createDataUpdateBurst();
    // Update metrics
    stepElement.textContent = data.step || '-';
    
    if (data.loss !== undefined) {
        lossElement.textContent = data.loss.toFixed(4);
        history.loss.push(data.loss);
        history.timestamps.push(data.step || history.timestamps.length);
        
        if (history.loss.length > history.maxPoints) {
            history.loss.shift();
            history.timestamps.shift();
        }
        
        // Draw the large minimalist plot with neon blue
        drawMinimalistPlot('loss-plot', history.loss, '#00D4FF', { isLoss: true });
    }
    
    if (data.perplexity !== undefined) {
        perplexityElement.textContent = data.perplexity.toFixed(2);
        history.perplexity.push(data.perplexity);
        
        if (history.perplexity.length > history.maxPoints) {
            history.perplexity.shift();
        }
        
        // No small graph anymore
    }
    
    // Calculate average entropy from sequences
    if (data.sequences && data.sequences.length > 0) {
        let totalEntropy = 0;
        let entropyCount = 0;
        
        data.sequences.forEach(seq => {
            if (seq.predictions) {
                seq.predictions.forEach(pred => {
                    if (pred.entropy !== undefined) {
                        totalEntropy += pred.entropy;
                        entropyCount++;
                    }
                });
            }
        });
        
        if (entropyCount > 0) {
            const avgEntropy = totalEntropy / entropyCount;
            history.entropy.push(avgEntropy);
            
            if (history.entropy.length > history.maxPoints) {
                history.entropy.shift();
            }
            
            // Draw the entropy plot with the same color as token entropy (#a78bfa)
            drawMinimalistPlot('entropy-plot', history.entropy, '#a78bfa');
        }
    }
    
    // Update sequences - animate if we have new data
    updateSequenceDisplay(true, previousSequences);
}

// Initialize tooltip element
function initTooltip() {
    plotState.tooltip = document.getElementById('plot-tooltip');
    plotState.tooltipLabel = plotState.tooltip.querySelector('.label');
    plotState.tooltipValue = plotState.tooltip.querySelector('.value');
}

// Handle mouse movement over plots
function handlePlotMouseMove(event, canvas, points, label, isEntropy = false) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Find closest point
    let closestPoint = null;
    let minDistance = 20; // Pixel threshold
    
    points.forEach(point => {
        const distance = Math.sqrt(Math.pow(x - point.x, 2) + Math.pow(y - point.y, 2));
        if (distance < minDistance) {
            minDistance = distance;
            closestPoint = point;
        }
    });
    
    if (closestPoint) {
        // Show tooltip
        plotState.tooltip.classList.add('visible');
        if (isEntropy) {
            plotState.tooltip.classList.add('entropy');
        } else {
            plotState.tooltip.classList.remove('entropy');
        }
        
        plotState.tooltipLabel.textContent = `${label} @ step ${closestPoint.step}`;
        plotState.tooltipValue.textContent = closestPoint.value.toFixed(4);
        
        // Position tooltip
        const tooltipX = event.clientX + 15;
        const tooltipY = event.clientY - 30;
        plotState.tooltip.style.left = tooltipX + 'px';
        plotState.tooltip.style.top = tooltipY + 'px';
    } else {
        // Hide tooltip
        plotState.tooltip.classList.remove('visible');
    }
}

// Initialize mouse events for plots
function initPlotInteractions() {
    const lossCanvas = document.getElementById('loss-plot');
    const entropyCanvas = document.getElementById('entropy-plot');
    
    if (lossCanvas) {
        lossCanvas.addEventListener('mousemove', (e) => {
            handlePlotMouseMove(e, lossCanvas, plotState.lossPoints, 'loss', false);
        });
        
        lossCanvas.addEventListener('mouseleave', () => {
            plotState.tooltip.classList.remove('visible');
        });
    }
    
    if (entropyCanvas) {
        entropyCanvas.addEventListener('mousemove', (e) => {
            handlePlotMouseMove(e, entropyCanvas, plotState.entropyPoints, 'entropy', true);
        });
        
        entropyCanvas.addEventListener('mouseleave', () => {
            plotState.tooltip.classList.remove('visible');
        });
    }
}

// Initialize
connectWebSocket();

// Initialize interactions when DOM is ready
window.addEventListener('DOMContentLoaded', () => {
    initTooltip();
    initPlotInteractions();
});