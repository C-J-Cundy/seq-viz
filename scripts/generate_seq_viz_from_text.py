#!/usr/bin/env python3
"""
Generate a seq viz file from input text using a real HuggingFace model.
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from seq_viz.core.tensor_to_training_data import tensor_to_training_entry


def main():
    parser = argparse.ArgumentParser(
        description="Generate seq viz file from text using HuggingFace model"
    )
    parser.add_argument("input_file", help="Path to input text file")
    parser.add_argument(
        "-m", "--model", 
        default="gpt2",
        help="HuggingFace model name (default: gpt2)"
    )
    parser.add_argument(
        "-o", "--output", 
        default="text_viz.jsonl",
        help="Output JSONL file path (default: text_viz.jsonl)"
    )
    
    args = parser.parse_args()
    
    # Read input text
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found")
        return 1
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if not text.strip():
        print("Error: Input file is empty")
        return 1
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = inputs.input_ids.to(device)
    
    print(f"Text tokenized into {input_ids.shape[1]} tokens")
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Calculate loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
    
    # Convert to training entry format
    entry = tensor_to_training_entry(
        logits=logits,
        input_ids=input_ids,
        tokenizer=tokenizer,
        step=0,
        loss=loss.item(),
        model_name=args.model,
        top_k=5,
        top_20=20
    )
    
    # Write to file
    with open(args.output, 'w') as f:
        f.write(json.dumps(entry) + '\n')
    
    print(f"\nSeq viz file generated: {args.output}")
    print(f"You can now visualize it using: python run_server.py {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())