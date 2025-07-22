"""
Train Llama on Shakespeare text using LoRA (Low-Rank Adaptation).

This example demonstrates:
1. Loading and preprocessing Shakespeare text
2. Fine-tuning Llama-3.2-1B with LoRA for efficient training
3. Visualizing training progress with seq-viz
"""

import os
from pathlib import Path

import requests
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
# Import our visualization callback
from seq_viz.integrations import VisualizationCallback
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)


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


def prepare_dataset(text_path, tokenizer, block_size=256):
    """Prepare dataset for training."""
    print("Preparing dataset...")

    # Read text
    with open(text_path, "r") as f:
        text = f.read()

    # Split into training chunks
    # Create overlapping chunks for better context
    chunks = []
    chunk_size = 1000
    overlap = 200

    for i in range(0, len(text) - chunk_size, chunk_size - overlap):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    print(f"Created {len(chunks)} text chunks")

    # Add Shakespeare-style prompts to some chunks
    shakespeare_prompts = [
        "Speak the speech, I pray you:\n",
        "What follows is a soliloquy:\n",
        "Hark! Listen to these words:\n",
        "In fair Verona, where we lay our scene:\n",
        "Once more unto the breach:\n",
    ]

    # Tokenize chunks
    def tokenize_function(examples):
        # Add some variety by occasionally prepending prompts
        import random

        texts = []
        for text in examples["text"]:
            if random.random() < 0.2:  # 20% chance to add prompt
                prompt = random.choice(shakespeare_prompts)
                texts.append(prompt + text)
            else:
                texts.append(text)

        outputs = tokenizer(
            texts,
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

    # Load model with quantization for efficiency
    print(f"Loading model {model_name} with 8-bit quantization...")

    # Quantization config
    # quantization_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     bnb_8bit_compute_dtype=torch.float16,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=quantization_config,
        device_map="auto",
    )

    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention layers
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare datasets
    train_dataset, eval_dataset = prepare_dataset(shakespeare_path, tokenizer, block_size=256)
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # Training arguments - adjusted for LoRA
    training_args = TrainingArguments(
        output_dir="./shakespeare_lora_output",
        overwrite_output_dir=True,
        num_train_epochs=2,  # Can do more epochs with LoRA
        per_device_train_batch_size=8,  # Can use larger batch with LoRA
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=5,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
        learning_rate=3e-4,  # Higher LR works well with LoRA
        # bf16=True,
    )

    # Create visualization callback
    viz_callback = VisualizationCallback(
        output_file="shakespeare_lora_training.jsonl",
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
    print("\n" + "=" * 50)
    print("Starting Shakespeare LoRA fine-tuning with visualization!")
    print("=" * 50)
    print("\nTo view the visualization:")
    print("1. In another terminal: python run_server.py --file shakespeare_lora_training.jsonl")
    print("2. Open seq_viz/web/enhanced_dashboard.html in your browser")
    print("\nTraining will start in a moment...\n")

    # Train
    trainer.train()

    # Save the LoRA model
    model.save_pretrained("./shakespeare_lora_model")
    tokenizer.save_pretrained("./shakespeare_lora_model")

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    print(f"LoRA model saved to: ./shakespeare_lora_model")
    print(f"Visualization data saved to: shakespeare_lora_training.jsonl")

    # Generate some sample text
    print("\nGenerating sample Shakespeare-style text...")
    model.eval()

    # Shakespeare-style prompts
    prompts = [
        "To be or not to be, that is the question:",
        "O Romeo, Romeo! wherefore art thou Romeo?",
        "All the world's a stage, and all the men and women merely players:",
        "Friends, Romans, countrymen, lend me your ears;",
        "Is this a dagger which I see before me,",
        "But, soft! what light through yonder window breaks?",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=100,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n[Prompt] {prompt}")
        print(f"[Generated] {generated}")
        print("-" * 80)


if __name__ == "__main__":
    main()
