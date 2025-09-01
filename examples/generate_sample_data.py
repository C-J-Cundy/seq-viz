#!/usr/bin/env python3
"""Generate properly formatted training data with Llama model."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from seq_viz.core import tensor_to_training_entry, TrainingDataWriter
import random

def generate_training_data(output_file="training_data.jsonl", num_steps=10):
    """Generate training data with proper top_k=5 and top_20=20."""
    
    print("Loading Llama 3.2 1B model...")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 128001
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Shakespeare quotes for variety
    texts = [
        "To be or not to be, that is the question",
        "All the world's a stage, and all the men and women merely players",
        "Romeo, Romeo, wherefore art thou Romeo?",
        "What's in a name? That which we call a rose",
        "The course of true love never did run smooth",
        "We are such stuff as dreams are made on",
        "How sharper than a serpent's tooth it is to have a thankless child",
        "Lord, what fools these mortals be!",
        "The lady doth protest too much, methinks",
        "Something is rotten in the state of Denmark"
    ]
    
    writer = TrainingDataWriter(output_file)
    device = next(model.parameters()).device
    
    print(f"Generating {num_steps} training steps...")
    
    for step in range(1000, 1000 + num_steps):
        # Pick random texts for this batch
        batch_texts = random.sample(texts, min(4, len(texts)))
        
        # Tokenize
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=30, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
        
        # Convert to training entry - sample 1-2 sequences per batch
        num_sequences = random.randint(1, 2)
        entry = tensor_to_training_entry(
            logits[:num_sequences].cpu(),
            input_ids[:num_sequences].cpu(),
            tokenizer,
            step=step,
            loss=loss.item(),
            model_name=model_name,
            top_k=5,      # Exactly 5 top predictions
            top_20=20     # Exactly 20 predictions for histogram
        )
        
        # Write to file
        success = writer.write_step(entry)
        
        if success:
            print(f"  Step {step}: Loss={loss.item():.4f}, Sequences={len(entry['sequences'])}")
        else:
            print(f"  Step {step}: Failed validation!")
    
    print(f"\nTraining data saved to {output_file}")
    print("Run validation to verify: python -m seq_viz.core.validate_training_data " + output_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training data for visualization")
    parser.add_argument("--output", default="training_data.jsonl", help="Output file path")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to generate")
    
    args = parser.parse_args()
    generate_training_data(args.output, args.steps)