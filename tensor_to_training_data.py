import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import time


def tensor_to_training_entry(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    step: int,
    loss: float,
    model_name: str,
    top_k: int = 5,
    top_20: int = 20,
    sequence_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convert model outputs to training data entry format.
    
    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size) or (seq_len, vocab_size)
        input_ids: Tensor of shape (batch_size, seq_len) or (seq_len,)
        tokenizer: Tokenizer with decode method
        step: Training step number
        loss: Training loss value
        model_name: Name of the model
        top_k: Number of top predictions to include in top_k field
        top_20: Number of top predictions to include in top_20 field
        sequence_idx: If provided, only process this sequence from the batch
    
    Returns:
        Dictionary matching the training data schema
    """
    # Handle both batched and single sequence inputs
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
        input_ids = input_ids.unsqueeze(0)
    
    batch_size, seq_len, vocab_size = logits.shape
    
    # Calculate perplexity from loss
    perplexity = torch.exp(torch.tensor(loss)).item()
    
    # Process sequences
    sequences = []
    
    # Determine which sequences to process
    if sequence_idx is not None:
        indices = [sequence_idx]
    else:
        indices = range(batch_size)
    
    for batch_idx in indices:
        # Get tokens for this sequence
        tokens = [tokenizer.decode(token_id, skip_special_tokens=False) 
                  for token_id in input_ids[batch_idx]]
        
        # Process predictions for each position
        predictions = []
        
        # We predict the next token, so we look at positions 0 to seq_len-1
        for pos in range(seq_len - 1):
            # Get logits for this position
            pos_logits = logits[batch_idx, pos]
            
            # Convert to probabilities
            probs = F.softmax(pos_logits, dim=-1)
            
            # Calculate entropy
            # Use torch.where to handle zero probabilities properly
            # When prob is 0, the contribution to entropy should be 0
            log_probs = torch.where(probs > 0, torch.log(probs), torch.zeros_like(probs))
            entropy_terms = torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs))
            entropy = -torch.sum(entropy_terms).item()
            
            # Get target token (next token in sequence)
            target_token_id = input_ids[batch_idx, pos + 1].item()
            target_token_str = tokenizer.decode(target_token_id, skip_special_tokens=False)
            
            # Get top-k predictions
            top_k_probs, top_k_indices = torch.topk(probs, min(top_k, vocab_size))
            top_k_predictions = []
            for prob, idx in zip(top_k_probs, top_k_indices):
                top_k_predictions.append({
                    "token_id": idx.item(),
                    "token_str": tokenizer.decode(idx.item(), skip_special_tokens=False),
                    "prob": prob.item()
                })
            
            # Get top-20 predictions
            top_20_probs, top_20_indices = torch.topk(probs, min(top_20, vocab_size))
            top_20_predictions = []
            for prob, idx in zip(top_20_probs, top_20_indices):
                top_20_predictions.append({
                    "token_id": idx.item(),
                    "token_str": tokenizer.decode(idx.item(), skip_special_tokens=False),
                    "prob": prob.item()
                })
            
            predictions.append({
                "position": pos,
                "target_token_id": target_token_id,
                "target_token_str": target_token_str,
                "top_k": top_k_predictions,
                "top_20": top_20_predictions,
                "entropy": entropy
            })
        
        sequences.append({
            "tokens": tokens,
            "predictions": predictions
        })
    
    # Build the complete entry
    entry = {
        "timestamp": time.time(),
        "step": step,
        "loss": loss,
        "perplexity": perplexity,
        "sequences": sequences,
        "metadata": {
            "model_name": model_name,
            "vocab_size": vocab_size,
            "batch_size": batch_size,
            "sequence_length": seq_len
        }
    }
    
    return entry


def extract_batch_sample(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    step: int,
    loss: float,
    model_name: str,
    num_sequences: int = 1,
    top_k: int = 5,
    top_20: int = 20
) -> Dict[str, Any]:
    """
    Extract a sample of sequences from a batch for visualization.
    
    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
        input_ids: Tensor of shape (batch_size, seq_len)
        tokenizer: Tokenizer with decode method
        step: Training step number
        loss: Training loss value
        model_name: Name of the model
        num_sequences: Number of sequences to sample from batch
        top_k: Number of top predictions to include in top_k field
        top_20: Number of top predictions to include in top_20 field
    
    Returns:
        Dictionary matching the training data schema
    """
    batch_size = logits.shape[0]
    
    # Sample random sequences from the batch
    if num_sequences >= batch_size:
        indices = list(range(batch_size))
    else:
        indices = torch.randperm(batch_size)[:num_sequences].tolist()
    
    # Process only selected sequences
    sequences = []
    for idx in indices:
        entry = tensor_to_training_entry(
            logits[idx].unsqueeze(0),
            input_ids[idx].unsqueeze(0),
            tokenizer,
            step,
            loss,
            model_name,
            top_k,
            top_20
        )
        sequences.extend(entry["sequences"])
    
    # Update the entry with sampled sequences
    entry["sequences"] = sequences
    entry["metadata"]["batch_size"] = batch_size  # Keep original batch size
    
    return entry