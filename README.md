# seq-viz

v0 - Initial proof of concept

A real-time visualization dashboard for transformer language model training.

## Quick Start

```bash
pip install -r requirements.txt
python run_token_dashboard.py
```

Then open http://localhost:8000/token_dashboard.html

## What it does

- Shows token predictions during training
- Visualizes entropy and prediction confidence
- Displays training metrics (loss, perplexity)

## Files

- `train_llama_shakespeare.py` - Training server
- `token_dashboard.html/js` - Visualization dashboard
- `run_token_dashboard.py` - Launcher script