import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from seq_viz.core import tensor_to_training_entry, TrainingDataWriter


def simulate_training_writes():
    """Simulate training by writing steps to a file periodically."""
    print("Setting up model and tokenizer...")
    
    # Initialize model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 128001
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Sample texts
    texts = [
        "To be or not to be, that is the question",
        "All the world's a stage, and all the men",
        "Romeo, Romeo, wherefore art thou Romeo?",
        "What's in a name? That which we call a rose"
    ]
    
    # Initialize writer
    writer = TrainingDataWriter("../live_training_data.jsonl")
    
    print("Starting simulated training...")
    print("Run file_visualization_server.py in another terminal to see updates")
    
    # Simulate 10 training steps
    for step in range(1000, 1010):
        print(f"\nStep {step}...")
        
        # Tokenize
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=20, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # Move to device
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            
            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
        
        # Convert to training entry (sample 2 sequences from batch)
        entry = tensor_to_training_entry(
            logits[0].cpu(),  # Just first sequence for this test
            input_ids[0].cpu(),
            tokenizer,
            step=step,
            loss=loss.item(),
            model_name=model_name,
            top_k=5,
            top_20=20
        )
        
        # Write to file
        success = writer.write_step(entry)
        print(f"  Written: {success}, Loss: {loss.item():.4f}")
        
        # Wait before next step
        time.sleep(3)
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    simulate_training_writes()