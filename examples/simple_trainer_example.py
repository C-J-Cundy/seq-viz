"""
Example of using VisualizationCallback with standard Trainer.

This demonstrates how to integrate the sequence visualizer with
HuggingFace's standard Trainer for language modeling.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Import our visualization callback
from seq_viz.integrations import VisualizationCallback


def main():
    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
    
    # Tokenize datasets
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Split into train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./trainer_output",
        evaluation_strategy="steps",
        eval_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        push_to_hub=False,
        logging_steps=10,
        save_steps=100,
        warmup_steps=10,
        report_to="none",
    )
    
    # Create visualization callback
    viz_callback = VisualizationCallback(
        output_file="trainer_viz.jsonl",
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
    
    # Custom compute metrics (optional)
    def compute_metrics(eval_pred):
        # This is just an example - you can compute any metrics you want
        predictions, labels = eval_pred
        # Simple accuracy-like metric
        predictions = predictions.argmax(axis=-1)
        
        # Flatten and remove padding
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        # Remove -100 labels (padding)
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        
        accuracy = (predictions == labels).mean()
        
        return {"accuracy": accuracy}
    
    # Optionally set compute_metrics
    # trainer.compute_metrics = compute_metrics
    
    # Start training
    print("\nStarting training with visualization...")
    print(f"Visualization data will be saved to: trainer_viz.jsonl")
    print("\nTo view the visualization:")
    print("1. In another terminal: python run_server.py --file trainer_viz.jsonl")
    print("2. Open seq_viz/web/enhanced_dashboard.html in your browser")
    print("-" * 50)
    
    trainer.train()
    
    print("\nTraining complete!")
    print("Visualization data saved to: trainer_viz.jsonl")


if __name__ == "__main__":
    main()