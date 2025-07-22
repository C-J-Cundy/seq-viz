"""
Generate a test sequence of training data with realistic loss curves
to test the visualization vibes.
"""

import json
import time
import math
import random

def generate_test_data():
    """Generate test training data with a realistic loss curve."""
    
    output_file = "test_vibes.jsonl"
    
    # Parameters for loss curve
    initial_loss = 4.5
    final_loss = 1.2
    steps = 150
    
    with open(output_file, 'w') as f:
        for step in range(steps):
            # Create a realistic loss curve with some noise
            progress = step / steps
            
            # Exponential decay with some noise
            loss = initial_loss * math.exp(-2.5 * progress) + final_loss
            loss += random.gauss(0, 0.05)  # Add some noise
            
            # Add occasional spikes (like when learning rate changes)
            if step in [30, 60, 90]:
                loss += random.uniform(0.1, 0.3)
            
            # Calculate perplexity
            perplexity = math.exp(loss)
            
            # Create dummy sequences with varying entropy
            sequences = []
            for seq_idx in range(2):
                # Generate tokens
                token_options = ["To", "be", "or", "not", "to", "be", "that", "is", "the", "question",
                               "Whether", "tis", "nobler", "in", "mind", "suffer", "slings", "arrows"]
                tokens = [random.choice(token_options) for _ in range(10)]
                
                predictions = []
                for pos in range(len(tokens) - 1):
                    # Entropy decreases as training progresses
                    base_entropy = 3.0 * (1 - progress) + 0.5
                    entropy = base_entropy + random.gauss(0, 0.2)
                    entropy = max(0.1, entropy)  # Keep positive
                    
                    # Target token
                    target_token_str = tokens[pos + 1]
                    target_token_id = hash(target_token_str) % 50000
                    
                    # Generate top 20 predictions
                    raw_probs = [math.exp(-i * 0.3) * random.uniform(0.5, 1.5) for i in range(20)]
                    total = sum(raw_probs)
                    probs = [p / total for p in raw_probs]
                    
                    # Build token lists
                    all_tokens = []
                    target_idx = random.randint(0, 3)  # Put target in top positions
                    
                    for i in range(20):
                        if i == target_idx:
                            token_str = target_token_str
                            token_id = target_token_id
                        else:
                            token_str = random.choice(token_options)
                            token_id = hash(token_str) % 50000
                        
                        all_tokens.append({
                            "token_id": token_id,
                            "token_str": token_str,
                            "prob": probs[i]
                        })
                    
                    predictions.append({
                        "position": pos,
                        "target_token_id": target_token_id,
                        "target_token_str": target_token_str,
                        "entropy": entropy,
                        "top_k": all_tokens[:5],
                        "top_20": all_tokens
                    })
                
                sequences.append({
                    "tokens": tokens,
                    "predictions": predictions
                })
            
            # Create the data entry
            entry = {
                "timestamp": time.time(),
                "step": step * 10,  # Scale up step numbers
                "loss": loss,
                "perplexity": perplexity,
                "sequences": sequences,
                "metadata": {
                    "model_name": "test-model",
                    "vocab_size": 50257,
                    "batch_size": 2,
                    "sequence_length": 10
                }
            }
            
            f.write(json.dumps(entry) + '\n')
    
    print(f"Generated {steps} test entries in {output_file}")
    print("\nTo test the visualization:")
    print("1. Run: python run_server.py --file test_vibes.jsonl")
    print("2. Open seq_viz/web/enhanced_dashboard.html in your browser")
    print("\nThe data will update every ~2 seconds to simulate training")

if __name__ == "__main__":
    generate_test_data()