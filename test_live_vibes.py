"""
Generate live streaming test data to test the visualization vibes in real-time.
"""

import json
import time
import math
import random
import argparse

def generate_live_data(output_file="test_live_vibes.jsonl", delay=2.0):
    """Generate test training data continuously."""
    
    print(f"Starting live data generation to {output_file}")
    print(f"Generating new data every {delay} seconds")
    print("Press Ctrl+C to stop")
    print("\nIn another terminal, run:")
    print(f"  python run_server.py --file {output_file}")
    print("\nThen open seq_viz/web/enhanced_dashboard.html in your browser")
    print("-" * 50)
    
    # Parameters
    initial_loss = 4.5
    target_loss = 0.8
    decay_rate = 0.01
    
    step = 0
    start_time = time.time()
    
    # Clear the file
    open(output_file, 'w').close()
    
    try:
        while True:
            # Calculate current loss with exponential decay
            elapsed = time.time() - start_time
            base_loss = target_loss + (initial_loss - target_loss) * math.exp(-decay_rate * step)
            
            # Add realistic noise and oscillations
            noise = random.gauss(0, 0.03)
            oscillation = 0.1 * math.sin(step * 0.1) * math.exp(-step * 0.002)
            loss = base_loss + noise + oscillation
            
            # Occasional spikes (gradient explosions, new data batches, etc.)
            if random.random() < 0.05:
                loss += random.uniform(0.1, 0.3)
            
            # Calculate perplexity
            perplexity = math.exp(loss)
            
            # Create sequences with time-varying entropy
            sequences = []
            for seq_idx in range(3):  # 3 sequences
                # Generate tokens for this sequence
                token_options = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
                               "Machine", "learning", "is", "transforming", "how", "we", "interact", "with", "technology"]
                num_tokens = 15
                tokens = [random.choice(token_options) for _ in range(num_tokens)]
                
                predictions = []
                
                # Base entropy decreases over time
                base_entropy = 2.5 * math.exp(-step * 0.005) + 0.5
                
                for pos in range(len(tokens) - 1):  # Predict next token
                    # Position-dependent entropy (earlier positions more certain)
                    position_factor = 1 + 0.3 * (pos / len(tokens))
                    entropy = base_entropy * position_factor + random.gauss(0, 0.15)
                    entropy = max(0.1, min(entropy, 4.0))  # Clamp to reasonable range
                    
                    # Target token (the next token in sequence)
                    target_token_str = tokens[pos + 1]
                    target_token_id = hash(target_token_str) % 50000
                    
                    # Generate probability distribution for top 20
                    raw_probs = []
                    for i in range(20):
                        # Exponential decay for realistic distribution
                        base_prob = math.exp(-i * 0.3)
                        raw_probs.append(base_prob * random.uniform(0.5, 1.5))
                    
                    # Make sure target token has decent probability
                    target_idx = random.randint(0, 4)  # Put target in top 5
                    raw_probs[target_idx] = raw_probs[0] * random.uniform(0.8, 1.2)
                    
                    # Normalize
                    total = sum(raw_probs)
                    probs = [p / total for p in raw_probs]
                    
                    # Create top_k and top_20
                    all_tokens = []
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
                        "top_k": all_tokens[:5],  # First 5
                        "top_20": all_tokens       # All 20
                    })
                
                sequences.append({
                    "tokens": tokens,
                    "predictions": predictions
                })
            
            # Create the data entry
            entry = {
                "timestamp": time.time(),
                "step": step * 50,  # Scale up for realistic step numbers
                "loss": loss,
                "perplexity": perplexity,
                "sequences": sequences,
                "metadata": {
                    "model_name": "llama-vibe-test",
                    "vocab_size": 50257,
                    "batch_size": len(sequences),
                    "sequence_length": 15
                }
            }
            
            # Append to file
            with open(output_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            
            print(f"Step {step * 50}: loss={loss:.4f}, perplexity={perplexity:.2f}, "
                  f"avg_entropy={base_entropy:.2f}")
            
            step += 1
            time.sleep(delay)
            
    except KeyboardInterrupt:
        print("\n\nStopped data generation")
        print(f"Generated {step} data points over {elapsed:.1f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate live test data for visualization")
    parser.add_argument("--file", default="test_live_vibes.jsonl", help="Output file name")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between updates (seconds)")
    
    args = parser.parse_args()
    generate_live_data(args.file, args.delay)