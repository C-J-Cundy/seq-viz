import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seq_viz.core import TrainingDataReader, TrainingDataWriter


def test_reader():
    """Test the data reader functionality."""
    print("Testing TrainingDataReader...")
    reader = TrainingDataReader("../training_data.jsonl")
    
    # Get summary
    summary = reader.get_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Get latest step
    latest = reader.get_latest()
    if latest:
        print(f"\nLatest step: {latest['step']}")
        print(f"  Loss: {latest['loss']:.4f}")
        print(f"  Perplexity: {latest['perplexity']:.2f}")
    
    # Check a specific step
    step_100 = reader.get_step(100)
    if step_100:
        print(f"\nStep 100:")
        print(f"  Loss: {step_100['loss']:.4f}")
        print(f"  Sequences: {len(step_100['sequences'])}")


def test_writer():
    """Test the data writer functionality."""
    print("\n\nTesting TrainingDataWriter...")
    writer = TrainingDataWriter("../test_output.jsonl")
    
    # Create a minimal valid entry
    test_entry = {
        "step": 9999,
        "loss": 1.234,
        "perplexity": 3.432,
        "sequences": [],
        "metadata": {
            "model_name": "test-model",
            "vocab_size": 1000,
            "batch_size": 4,
            "sequence_length": 64
        }
    }
    
    success = writer.write_step(test_entry)
    print(f"Write successful: {success}")
    
    # Try writing an invalid entry
    invalid_entry = {
        "step": 10000,
        "loss": "not a number",  # Invalid type
        "sequences": []
    }
    
    print("\nTrying to write invalid entry...")
    success = writer.write_step(invalid_entry)
    print(f"Write successful: {success}")


if __name__ == "__main__":
    test_reader()
    test_writer()