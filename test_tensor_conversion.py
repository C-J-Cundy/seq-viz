import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tensor_to_training_data import tensor_to_training_entry, extract_batch_sample
from data_writer import TrainingDataWriter
import json


def test_tensor_conversion():
    """Test the tensor to training data conversion with real model outputs."""
    print("Testing tensor to training data conversion with Llama 3.2 1B...")
    
    # Initialize Llama 3.2 1B Instruct
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Loading tokenizer and model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 128001  # Set padding token ID
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 to save memory
        device_map="auto"
    )
    
    # Sample Shakespeare text
    texts = [
        "To be or not to be, that is the question",
        "All the world's a stage, and all the men",
        "Romeo, Romeo, wherefore art thou Romeo?",
        "What's in a name? That which we call a rose"
    ]
    
    # Tokenize
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=20, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Move to same device as model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    print(f"\nInput shape: {input_ids.shape}")
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        
        # Calculate loss using cross entropy
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
    
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test single sequence conversion
    print("\n1. Testing single sequence conversion:")
    entry = tensor_to_training_entry(
        logits[0].cpu(),  # Move to CPU for processing
        input_ids[0].cpu(),
        tokenizer,
        step=1000,
        loss=loss.item(),
        model_name=model_name,
        top_k=5,
        top_20=20
    )
    
    print(f"Entry keys: {list(entry.keys())}")
    print(f"Number of sequences: {len(entry['sequences'])}")
    print(f"Number of predictions: {len(entry['sequences'][0]['predictions'])}")
    print(f"First few tokens: {entry['sequences'][0]['tokens'][:5]}")
    
    # Show first few predictions
    for i in range(min(3, len(entry['sequences'][0]['predictions']))):
        pred = entry['sequences'][0]['predictions'][i]
        print(f"\nPrediction at position {pred['position']}:")
        print(f"  Current token: '{entry['sequences'][0]['tokens'][pred['position']]}'")
        print(f"  Target token: '{pred['target_token_str']}' (ID: {pred['target_token_id']})")
        print(f"  Top 3 predictions:")
        for j, top_pred in enumerate(pred['top_k'][:3]):
            print(f"    {j+1}. '{top_pred['token_str']}' (prob: {top_pred['prob']:.3f})")
        print(f"  Entropy: {pred['entropy']:.3f}")
    
    # Test batch sampling
    print("\n2. Testing batch sampling:")
    batch_entry = extract_batch_sample(
        logits.cpu(),
        input_ids.cpu(),
        tokenizer,
        step=1001,
        loss=loss.item(),
        model_name=model_name,
        num_sequences=2,  # Sample 2 sequences from batch
        top_k=5,
        top_20=20
    )
    
    print(f"Sampled {len(batch_entry['sequences'])} sequences from batch of {batch_entry['metadata']['batch_size']}")
    
    # Write to file
    print("\n3. Testing write to file:")
    writer = TrainingDataWriter("test_tensor_output.jsonl")
    success = writer.write_step(entry)
    print(f"Write successful: {success}")
    
    # Pretty print one prediction for inspection
    print("\n4. Sample prediction structure:")
    display_pred = entry['sequences'][0]['predictions'][0].copy()
    display_pred['top_k'] = display_pred['top_k'][:3]  # Just top 3
    display_pred['top_20'] = display_pred['top_20'][:5]  # Just top 5
    
    print(json.dumps(display_pred, indent=2))


if __name__ == "__main__":
    test_tensor_conversion()