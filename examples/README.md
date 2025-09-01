# Seq-Viz Examples

## Main Example: Shakespeare LoRA Training

### `shakespeare_lora_training.py`

Demonstrates how to fine-tune a language model on Shakespeare text using LoRA (Low-Rank Adaptation) while visualizing the training process with seq-viz.

**Features:**
- Automatically downloads Shakespeare text corpus
- Uses LoRA for efficient training (only ~0.5% of parameters)
- Integrates with seq-viz for real-time visualization
- Generates Shakespeare-style text after training

**Usage:**
```bash
python examples/shakespeare_lora_training.py
```

Then in another terminal:
```bash
python run_server.py --file shakespeare_lora_training.jsonl
```

Open `seq_viz/web/dashboard.html` in your browser to view the visualization.

## Utility Scripts

### `generate_sample_data.py`

Generates sample visualization data for testing the dashboard without running a full training session.

```bash
python examples/generate_sample_data.py --steps 10
```

## Sample Data

### `sample.jsonl`

A small sample of visualization data (3 training steps) that can be used to quickly test the visualization dashboard:

```bash
python run_server.py --file examples/sample.jsonl
```

This is useful for:
- Testing the dashboard without running training
- Understanding the data format
- Quick demos