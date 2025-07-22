"""
Quick test of the HuggingFace integration.
This script tests the visualization callback with a minimal setup.
"""

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

# Import our visualization callback
from seq_viz.integrations import VisualizationCallback


def create_dummy_dataset(tokenizer, num_samples=50):
    """Create a simple dataset for testing."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we interact with technology.",
        "Python is a versatile programming language used for many applications.",
        "Neural networks learn patterns from large amounts of data.",
        "The weather today is sunny with a chance of rain later.",
    ] * (num_samples // 5)

    # Tokenize
    encodings = tokenizer(
        texts[:num_samples],
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )

    # Create dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings.input_ids)

        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}

    return SimpleDataset(encodings)


def main():
    print("Testing HuggingFace integration...")

    # Use a tiny model for quick testing
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Small model for testing

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create dummy datasets
    print("Creating test dataset...")
    train_dataset = create_dummy_dataset(tokenizer, num_samples=50)
    eval_dataset = create_dummy_dataset(tokenizer, num_samples=10)

    # Training arguments - minimal for testing
    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=1,  # Evaluate frequently for testing
        logging_steps=5,
        save_steps=100,
        warmup_steps=0,
        report_to="none",
        max_steps=20,  # Limit steps for quick test
    )

    # Create visualization callback
    viz_callback = VisualizationCallback(
        output_file="test_viz.jsonl",
        max_sequences_per_eval=2,
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

    # Run training
    print("\nStarting test training...")
    print("This should create test_viz.jsonl with visualization data")

    try:
        trainer.train()
        print("\n✅ Test completed successfully!")
        print("Check test_viz.jsonl for visualization data")

        # Quick check if file was created
        import os

        if os.path.exists("test_viz.jsonl"):
            size = os.path.getsize("test_viz.jsonl")
            print(f"Visualization file created: {size} bytes")

            # Show first few lines
            with open("test_viz.jsonl", "r") as f:
                lines = f.readlines()
                print(f"Number of evaluation steps recorded: {len(lines)}")
                if lines:
                    import json

                    first_entry = json.loads(lines[0])
                    print(f"First entry has {len(first_entry['sequences'])} sequences")

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
