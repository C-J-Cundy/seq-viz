# Shakespeare Training Examples

These examples demonstrate how to fine-tune language models on Shakespeare text while visualizing the training process with seq-viz.

## Prerequisites

```bash
# Core dependencies
pip install torch transformers datasets

# For LoRA example (optional but recommended)
pip install peft bitsandbytes accelerate
```

## Examples

### 1. Basic Fine-tuning (`shakespeare_training.py`)

Fine-tunes the full Llama-3.2-1B model on Shakespeare text:

```bash
python examples/shakespeare_training.py
```

Features:
- Downloads Shakespeare text automatically
- Fine-tunes all model parameters
- Uses mixed precision (fp16) for efficiency
- Generates sample text after training

### 2. LoRA Fine-tuning (`shakespeare_lora_training.py`)

More efficient fine-tuning using Low-Rank Adaptation (LoRA):

```bash
python examples/shakespeare_lora_training.py
```

Features:
- Uses 8-bit quantization to reduce memory usage
- Only trains ~0.5% of parameters with LoRA
- Faster training with similar quality results
- Can use larger batch sizes

## Viewing the Visualization

1. Start the visualization server:
```bash
python run_server.py --file shakespeare_training.jsonl
```

2. Open the dashboard in your browser:
```bash
open seq_viz/web/enhanced_dashboard.html
```

## What to Look For

During training, the visualization will show:

1. **Token Predictions**: Watch how the model learns Shakespeare's language patterns
2. **Entropy Changes**: See uncertainty decrease as the model learns
3. **Loss Curves**: Track training progress in real-time
4. **Pattern Recognition**: Observe when the model starts generating proper iambic pentameter

## Sample Output

After training, the model will generate Shakespeare-style text:

```
Prompt: To be or not to be, that is the question:
Generated: To be or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles...
```

## Tips

1. **Memory Usage**: Use the LoRA version if you have limited GPU memory
2. **Training Time**: Full fine-tuning takes ~30 minutes on a decent GPU; LoRA takes ~10 minutes
3. **Quality**: Both methods produce good results, but LoRA is more efficient
4. **Customization**: Adjust `num_train_epochs` and `learning_rate` for different results