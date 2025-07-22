"""
Example of using VisualizationCallback with SFTTrainer.

This demonstrates how to integrate the sequence visualizer with
HuggingFace's SFTTrainer for supervised fine-tuning.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer

# Import our visualization callback
from seq_viz.integrations import VisualizationCallback


def main():
    # Load a small model for demonstration
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load a sample dataset
    print("Loading dataset...")
    dataset = load_dataset("imdb", split="train[:100]")  # Small subset for demo
    
    # Preprocessing function
    def preprocess_function(examples):
        # Simple prompt format
        texts = [f"Review: {text}\nSentiment:" for text in examples["text"]]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=128)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./sft_output",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb/tensorboard for this example
    )
    
    # Create visualization callback
    viz_callback = VisualizationCallback(
        output_file="sft_training_viz.jsonl",
        max_sequences_per_eval=4,  # Visualize 4 sequences per evaluation
        tokenizer=tokenizer,
    )
    
    # Create trainer with visualization callback
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset[:20],  # Small eval set
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[viz_callback],
    )
    
    # Start training
    print("Starting training with visualization...")
    print(f"Visualization data will be saved to: sft_training_viz.jsonl")
    print("Run the visualization server in another terminal:")
    print("  python run_server.py --file sft_training_viz.jsonl")
    print("Then open seq_viz/web/enhanced_dashboard.html in your browser")
    
    trainer.train()
    
    print("Training complete!")
    print("Check sft_training_viz.jsonl for visualization data")


if __name__ == "__main__":
    main()