# Sequence Training Visualizer

A real-time visualization system for transformer model token predictions during training. This tool helps understand and debug language model behavior by showing token-by-token predictions, probabilities, and entropy values.

## Overview

The system consists of:
- **Data Pipeline**: Convert PyTorch tensors to structured JSON format
- **File-based Storage**: JSONL files with schema validation
- **WebSocket Server**: Stream training data to visualization clients
- **Interactive Dashboard**: Real-time visualization with entropy, probability distributions, and prediction rankings

## Features

- 📊 **Real-time Visualization**: See model predictions update as training progresses
- 🎯 **Token-level Predictions**: View top-k predictions for each token position
- 📈 **Entropy Tracking**: Monitor prediction uncertainty at each position
- 🔍 **Ground Truth Highlighting**: Easily spot correct vs incorrect predictions
- 📉 **Loss/Perplexity Graphs**: Track training metrics over time
- 🎨 **Beautiful Dark Theme**: Optimized for long viewing sessions

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd seq-viz

# Install dependencies
pip install torch transformers jsonschema websockets
```

## Quick Start

### 1. Generate Training Data

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tensor_to_training_data import tensor_to_training_entry
from data_writer import TrainingDataWriter

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = 128001
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Get model outputs
inputs = tokenizer("To be or not to be", return_tensors="pt")
outputs = model(**inputs)

# Convert to visualization format
entry = tensor_to_training_entry(
    logits=outputs.logits[0],
    input_ids=inputs.input_ids[0],
    tokenizer=tokenizer,
    step=1000,
    loss=2.5,
    model_name=model_name
)

# Write to file
writer = TrainingDataWriter("training_data.jsonl")
writer.write_step(entry)
```

### 2. Start Visualization Server

```bash
python file_visualization_server.py --file training_data.jsonl
```

### 3. Open Dashboard

```bash
open enhanced_dashboard.html
```

## Architecture

### Data Format

Training data is stored in JSONL format with the following structure:

```json
{
  "timestamp": 1234567890.123,
  "step": 1000,
  "loss": 2.345,
  "perplexity": 10.44,
  "sequences": [{
    "tokens": ["To", " be", " or", " not", " to", " be"],
    "predictions": [{
      "position": 0,
      "target_token_id": 387,
      "target_token_str": " be",
      "top_k": [/* top 5 predictions */],
      "top_20": [/* top 20 predictions */],
      "entropy": 1.234
    }]
  }],
  "metadata": {
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "vocab_size": 128256,
    "batch_size": 4,
    "sequence_length": 64
  }
}
```

### Components

#### 1. Data Writer (`data_writer.py`)
- Validates data against JSON schema before writing
- Simple, focused API: `writer.write_step(data)`

#### 2. Data Reader (`data_reader.py`)
- Read all steps or iterate through them
- Get specific steps by number
- Generate summary statistics

#### 3. Tensor Converter (`tensor_to_training_data.py`)
- Convert PyTorch model outputs to visualization format
- Calculate entropy and probability distributions
- Support for batch processing

#### 4. Visualization Server (`file_visualization_server.py`)
- WebSocket server that monitors JSONL files
- Streams updates to connected clients
- Configurable update intervals

#### 5. Enhanced Dashboard (`enhanced_dashboard.html/js`)
- Interactive token visualization
- Probability distribution charts
- Real-time metric graphs
- Responsive design with auto-scaling tokens

## Usage Examples

### Live Training Integration

```python
# In your training loop
for step, batch in enumerate(dataloader):
    outputs = model(**batch)
    
    # Convert and save for visualization
    entry = tensor_to_training_entry(
        logits=outputs.logits,
        input_ids=batch['input_ids'],
        tokenizer=tokenizer,
        step=step,
        loss=loss.item(),
        model_name="my-model"
    )
    writer.write_step(entry)
```

### Batch Analysis

```python
from data_reader import TrainingDataReader

reader = TrainingDataReader("training_data.jsonl")

# Get summary statistics
summary = reader.get_summary()
print(f"Average loss: {summary['avg_loss']:.4f}")
print(f"Min perplexity: {summary['min_perplexity']:.2f}")

# Analyze specific step
step_100 = reader.get_step(100)
if step_100:
    print(f"Loss at step 100: {step_100['loss']:.4f}")
```

### Custom Visualization Server

```bash
# Different port
python file_visualization_server.py --port 8080

# Faster updates
python file_visualization_server.py --interval 1.0

# Monitor specific file
python file_visualization_server.py --file experiments/run1.jsonl
```

## Testing

```bash
# Validate data format
python validate_training_data.py training_data.jsonl

# Test tensor conversion
python test_tensor_conversion.py

# Test file server with live updates
python test_file_server.py
```

## Visualization Features

### Token Display
- Fixed-size token boxes with automatic font scaling
- Click tokens to see their predictions
- Visual shift: predictions shown under the token they predict

### Probability Visualization
- Mini bar chart showing top-20 token probabilities
- Green highlighting for correct predictions
- Probability bars for top-5 predictions

### Metrics
- Real-time loss and perplexity graphs
- Entropy values for each prediction
- Step counter

## Development

### Adding New Features

1. **New Data Fields**: Update `training_data_schema.json`
2. **New Visualizations**: Modify `enhanced_dashboard.js`
3. **New Processing**: Extend `tensor_to_training_data.py`

### File Structure

```
seq-viz/
├── data_writer.py              # Write training data to JSONL
├── data_reader.py              # Read and analyze training data
├── tensor_to_training_data.py  # Convert PyTorch tensors to data format
├── training_data_schema.json   # JSON schema for validation
├── validate_training_data.py   # Validate JSONL files
├── file_visualization_server.py # WebSocket server for file monitoring
├── enhanced_dashboard.html     # Main visualization interface
├── enhanced_dashboard.js       # Dashboard logic
├── test_tensor_conversion.py   # Test tensor conversion
├── test_file_server.py         # Test live updates
└── README.md                   # This file
```

## Troubleshooting

### "Validation error" when writing data
- Check that all required fields are present
- Run `validate_training_data.py` to debug

### Dashboard shows "Disconnected"
- Ensure the server is running: `python file_visualization_server.py`
- Check that the port (default 8765) is not in use
- Verify the WebSocket URL in the dashboard matches the server

### Tokens overlapping or text cut off
- The dashboard automatically scales font size for long tokens
- Maximum token width is 140px with minimum 60% font scaling

## Future Enhancements

- [ ] Multi-sequence comparison view
- [ ] Export visualizations as images/videos
- [ ] Attention weight visualization
- [ ] Token embedding similarity view
- [ ] Training checkpoint comparison

## License

[Your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.