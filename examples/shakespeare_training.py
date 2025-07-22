"""
Train a language model on Shakespeare text using HuggingFace Transformers.

This example demonstrates:
1. Loading and preprocessing Shakespeare text
2. Fine-tuning Llama-3.2-1B on Shakespeare
3. Visualizing training progress with seq-viz
"""

import os
import requests
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# Import our visualization callback
from seq_viz.integrations import VisualizationCallback


def download_shakespeare():
    """Download Shakespeare text if not already present."""
    shakespeare_path = Path("shakespeare.txt")
    
    if not shakespeare_path.exists():
        print("Downloading Shakespeare text...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        with open(shakespeare_path, "w") as f:
            f.write(response.text)
        print(f"Downloaded {len(response.text)} characters of Shakespeare")
    
    return shakespeare_path


def prepare_dataset(text_path, tokenizer, block_size=128):
    """Prepare dataset for training."""
    print("Preparing dataset...")
    
    # Read text
    with open(text_path, "r") as f:
        text = f.read()
    
    # Split into training chunks
    # For simplicity, we'll just split by newlines and filter short sequences
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 10]
    
    # Create chunks of reasonable length
    chunks = []
    current_chunk = ""
    
    for line in lines:
        if len(current_chunk) + len(line) < 500:  # Approximate chunk size
            current_chunk += line + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    print(f"Created {len(chunks)} text chunks")
    
    # Tokenize chunks
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=block_size,
        )
        # For language modeling, labels are the same as input_ids
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    # Create dataset
    dataset = Dataset.from_dict({"text": chunks})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    # Split into train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    
    return split_dataset["train"], split_dataset["test"]


def main():
    # Model name
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Download Shakespeare text
    shakespeare_path = download_shakespeare()
    
    # Initialize tokenizer
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_dataset(shakespeare_path, tokenizer)
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./shakespeare_output",
        overwrite_output_dir=True,
        num_train_epochs=1,  # Just 1 epoch for demonstration
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
        learning_rate=2e-5,
        fp16=True,  # Use mixed precision
        gradient_accumulation_steps=2,  # Accumulate gradients
    )
    
    # Create visualization callback
    viz_callback = VisualizationCallback(
        output_file="shakespeare_training.jsonl",
        max_sequences_per_eval=4,
        tokenizer=tokenizer,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
        callbacks=[viz_callback],
    )
    
    # Start training
    print("\n" + "="*50)
    print("Starting Shakespeare fine-tuning with visualization!")
    print("="*50)
    print("\nTo view the visualization:")
    print("1. In another terminal: python run_server.py --file shakespeare_training.jsonl")
    print("2. Open seq_viz/web/enhanced_dashboard.html in your browser")
    print("\nTraining will start in a moment...\n")
    
    # Train
    trainer.train()
    
    # Save the final model
    trainer.save_model("./shakespeare_llama")
    tokenizer.save_pretrained("./shakespeare_llama")
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    print(f"Model saved to: ./shakespeare_llama")
    print(f"Visualization data saved to: shakespeare_training.jsonl")
    
    # Generate some sample text
    print("\nGenerating sample text...")
    model.eval()
    
    # Generate a few samples
    prompts = [
        "To be or not to be",
        "O Romeo, Romeo! wherefore art thou",
        "All the world's a stage",
        "Friends, Romans, countrymen",
    ]
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=80,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")


if __name__ == "__main__":
    main()