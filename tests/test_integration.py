"""Integration tests for the complete seq-viz pipeline."""

import json
import tempfile
from pathlib import Path
import shutil
import pytest
import torch
import numpy as np

from seq_viz.core import (
    tensor_to_training_entry,
    TrainingDataWriter,
    TrainingDataReader
)


class MockModel:
    """Mock language model for testing."""
    
    def __init__(self, vocab_size=100, hidden_size=64):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
    def forward(self, input_ids):
        """Generate mock logits."""
        batch_size, seq_len = input_ids.shape
        # Create logits with some structure (not random)
        # Make the model "prefer" the next token in sequence
        logits = torch.randn(batch_size, seq_len, self.vocab_size) * 0.1
        for b in range(batch_size):
            for s in range(seq_len - 1):
                next_token = input_ids[b, s + 1].item()
                if next_token >= 0:  # Not padding
                    logits[b, s, next_token] += 2.0  # Boost correct prediction
        return logits


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        
    def decode(self, token_ids, skip_special_tokens=False):
        """Convert token IDs to strings."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if isinstance(token_ids, int):
            return self._decode_single(token_ids, skip_special_tokens)
        
        return " ".join([self._decode_single(t, skip_special_tokens) for t in token_ids])
    
    def _decode_single(self, token_id, skip_special_tokens):
        """Decode a single token."""
        if token_id == self.pad_token_id:
            return "" if skip_special_tokens else "[PAD]"
        elif token_id == self.eos_token_id:
            return "" if skip_special_tokens else "[EOS]"
        elif token_id < 0:
            return "[MASK]"
        else:
            # Simple word-like tokens
            words = ["the", "cat", "sat", "on", "mat", "dog", "ran", "quick", "brown", "fox"]
            if token_id - 2 < len(words):
                return words[token_id - 2]
            return f"token_{token_id}"


@pytest.fixture
def temp_jsonl_file():
    """Create a temporary JSONL file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


def test_full_pipeline_integration(temp_jsonl_file):
    """Test the complete pipeline from model output to visualization data."""
    
    # Setup
    model = MockModel(vocab_size=100)
    tokenizer = MockTokenizer(vocab_size=100)
    writer = TrainingDataWriter(temp_jsonl_file)
    
    # Simulate training for multiple steps
    num_steps = 5
    batch_size = 2
    seq_length = 8
    
    for step in range(num_steps):
        # Create input sequences
        input_ids = torch.randint(2, 20, (batch_size, seq_length))
        # Add some padding to test padding handling
        input_ids[0, -2:] = tokenizer.pad_token_id
        input_ids[1, -3:] = tokenizer.pad_token_id
        
        # Get model predictions
        with torch.no_grad():
            logits = model.forward(input_ids)
        
        # Calculate a mock loss (decreasing over time to simulate learning)
        base_loss = 3.0 - (step * 0.4)
        loss = base_loss + np.random.random() * 0.1
        
        # Convert to visualization format
        for batch_idx in range(batch_size):
            entry = tensor_to_training_entry(
                logits=logits,
                input_ids=input_ids,
                tokenizer=tokenizer,
                step=step * batch_size + batch_idx,
                loss=loss,
                model_name="mock-model",
                sequence_idx=batch_idx,
                top_k=5,
                top_20=20,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Write to file
            writer.write_step(entry)
    
    # Now read back and verify
    reader = TrainingDataReader(temp_jsonl_file)
    
    # Check all steps were written
    all_steps = reader.read_all()
    assert len(all_steps) == num_steps * batch_size
    
    # Verify step numbers
    for i, entry in enumerate(all_steps):
        assert entry["step"] == i
        assert "timestamp" in entry
        assert entry["loss"] > 0
        assert entry["perplexity"] > 0
        
        # Check sequences structure
        assert len(entry["sequences"]) == 1
        seq = entry["sequences"][0]
        assert len(seq["tokens"]) == seq_length
        assert len(seq["token_ids"]) == seq_length
        
        # Check predictions exist for non-padding positions
        assert len(seq["predictions"]) > 0
        for pred in seq["predictions"]:
            assert "position" in pred
            assert "target_token_id" in pred
            assert "target_token_str" in pred
            assert "entropy" in pred
            assert "loss" in pred
            assert len(pred["top_k"]) == 5
            assert len(pred["top_20"]) == 20
            
            # Verify probabilities are valid (top_20 won't sum to 1, just a subset)
            total_prob = sum(p["prob"] for p in pred["top_20"])
            assert 0 < total_prob <= 1.0
            # Check probabilities are sorted
            probs = [p["prob"] for p in pred["top_20"]]
            assert probs == sorted(probs, reverse=True)
    
    # Test summary statistics
    summary = reader.get_summary()
    assert summary["total_steps"] == num_steps * batch_size
    assert summary["first_step"] == 0
    assert summary["last_step"] == num_steps * batch_size - 1
    assert summary["min_loss"] < summary["max_loss"]
    assert summary["avg_loss"] > 0
    
    # Test get_step
    step_3 = reader.get_step(3)
    assert step_3 is not None
    assert step_3["step"] == 3
    
    # Test iteration
    count = sum(1 for _ in reader.iter_steps())
    assert count == num_steps * batch_size


def test_pipeline_with_special_tokens(temp_jsonl_file):
    """Test pipeline with special tokens and edge cases."""
    
    model = MockModel(vocab_size=50)
    tokenizer = MockTokenizer(vocab_size=50)
    writer = TrainingDataWriter(temp_jsonl_file)
    
    # Test with various edge cases
    test_cases = [
        # Single token sequence
        torch.tensor([[5]]),
        # All padding
        torch.tensor([[0, 0, 0, 0]]),
        # Mix of regular and special tokens
        torch.tensor([[2, 3, 1, 0, 0]]),  # tokens, EOS, padding
    ]
    
    for i, input_ids in enumerate(test_cases):
        logits = model.forward(input_ids)
        
        entry = tensor_to_training_entry(
            logits=logits,
            input_ids=input_ids,
            tokenizer=tokenizer,
            step=i,
            loss=2.0,
            model_name="test-model",
            pad_token_id=tokenizer.pad_token_id
        )
        
        writer.write_step(entry)
    
    # Verify all entries were written correctly
    reader = TrainingDataReader(temp_jsonl_file)
    all_steps = reader.read_all()
    assert len(all_steps) == len(test_cases)


def test_pipeline_error_recovery(temp_jsonl_file):
    """Test that pipeline handles errors gracefully."""
    
    tokenizer = MockTokenizer()
    writer = TrainingDataWriter(temp_jsonl_file)
    
    # Write a valid entry
    valid_entry = tensor_to_training_entry(
        logits=torch.randn(1, 5, 100),
        input_ids=torch.tensor([[2, 3, 4, 5, 6]]),
        tokenizer=tokenizer,
        step=0,
        loss=2.5,
        model_name="test"
    )
    writer.write_step(valid_entry)
    
    # Try to write invalid entry (should raise)
    with pytest.raises(ValueError):
        invalid_entry = {"step": 1, "loss": 2.0}  # Missing required fields
        writer.write_step(invalid_entry)
    
    # Write another valid entry
    valid_entry2 = tensor_to_training_entry(
        logits=torch.randn(1, 5, 100),
        input_ids=torch.tensor([[7, 8, 9, 10, 11]]),
        tokenizer=tokenizer,
        step=2,
        loss=2.3,
        model_name="test"
    )
    writer.write_step(valid_entry2)
    
    # Verify only valid entries were written
    reader = TrainingDataReader(temp_jsonl_file)
    all_steps = reader.read_all()
    assert len(all_steps) == 2
    assert all_steps[0]["step"] == 0
    assert all_steps[1]["step"] == 2


def test_pipeline_with_real_text_simulation(temp_jsonl_file):
    """Test with more realistic text-like sequences."""
    
    model = MockModel(vocab_size=50)
    tokenizer = MockTokenizer(vocab_size=50)
    writer = TrainingDataWriter(temp_jsonl_file)
    
    # Simulate a few "sentences" as token IDs
    sentences = [
        [2, 3, 4, 5, 6, 1, 0, 0],  # "the cat sat on mat [EOS] [PAD] [PAD]"
        [2, 8, 9, 10, 4, 5, 6, 1],  # "the quick brown fox sat on mat [EOS]"
        [7, 8, 9, 1, 0, 0, 0, 0],  # "dog quick brown [EOS] [PAD] [PAD] [PAD] [PAD]"
    ]
    
    for step, tokens in enumerate(sentences):
        input_ids = torch.tensor([tokens])
        logits = model.forward(input_ids)
        
        # Simulate decreasing loss
        loss = 3.0 - (step * 0.5)
        
        entry = tensor_to_training_entry(
            logits=logits,
            input_ids=input_ids,
            tokenizer=tokenizer,
            step=step,
            loss=loss,
            model_name="text-model",
            pad_token_id=tokenizer.pad_token_id
        )
        
        writer.write_step(entry)
        
        # Verify the entry is valid for visualization
        assert "sequences" in entry
        assert len(entry["sequences"]) == 1
        
        seq = entry["sequences"][0]
        # Check tokens were decoded properly
        assert all(isinstance(t, str) for t in seq["tokens"])
        
        # Check predictions stop at EOS or first padding
        num_predictions = len(seq["predictions"])
        if tokenizer.eos_token_id in tokens:
            eos_position = tokens.index(tokenizer.eos_token_id)
            assert num_predictions <= eos_position
    
    # Verify the file can be used by visualization
    reader = TrainingDataReader(temp_jsonl_file)
    all_steps = reader.read_all()
    
    # Check that loss decreases over time (learning)
    losses = [step["loss"] for step in all_steps]
    assert losses[0] > losses[-1]
    
    # Verify all entries have valid visualization data
    for entry in all_steps:
        assert entry["loss"] > 0
        assert entry["perplexity"] == pytest.approx(np.exp(entry["loss"]), rel=0.01)
        assert len(entry["sequences"]) > 0
        
        # Each sequence should have tokens and predictions
        for seq in entry["sequences"]:
            assert len(seq["tokens"]) > 0
            assert all(isinstance(token, str) for token in seq["tokens"])
            
            # Predictions should have valid structure
            for pred in seq["predictions"]:
                assert 0 <= pred["position"] < len(seq["tokens"]) - 1
                assert pred["entropy"] >= 0
                assert len(pred["top_k"]) == 5
                assert len(pred["top_20"]) == 20
                
                # Top predictions should be sorted by probability
                probs = [p["prob"] for p in pred["top_k"]]
                assert probs == sorted(probs, reverse=True)


def test_pipeline_with_real_llama_model(temp_jsonl_file):
    """Test the pipeline with a real HuggingFace Llama model."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        pytest.skip("transformers not installed")
    
    # Load real model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu"  # Use CPU for CI compatibility
        )
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")
    
    writer = TrainingDataWriter(temp_jsonl_file)
    
    # Test with real text
    test_texts = [
        "The quick brown fox",
        "Machine learning is",
        "To be or not to be",
    ]
    
    for step, text in enumerate(test_texts):
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=20)
        input_ids = inputs["input_ids"]
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
        
        # Calculate loss (mock for this test)
        loss = 2.5 - (step * 0.3)
        
        # Convert to visualization format
        entry = tensor_to_training_entry(
            logits=logits,
            input_ids=input_ids,
            tokenizer=tokenizer,
            step=step,
            loss=loss,
            model_name=model_name,
            top_k=5,
            top_20=20,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Verify entry structure
        assert "sequences" in entry
        assert len(entry["sequences"]) == 1
        seq = entry["sequences"][0]
        
        # Check tokens match what we expect
        assert len(seq["tokens"]) == input_ids.shape[1]
        assert seq["tokens"][0] == tokenizer.decode(input_ids[0, 0])
        
        # Check predictions
        assert len(seq["predictions"]) > 0
        for pred in seq["predictions"]:
            # Verify we're getting real token predictions
            assert pred["target_token_str"] != ""
            assert pred["target_token_id"] >= 0
            assert pred["target_token_id"] < model.config.vocab_size
            
            # Check entropy is reasonable for a real model
            assert 0 <= pred["entropy"] <= 20  # Reasonable range for entropy
            
            # Verify top predictions include real tokens
            for top_pred in pred["top_k"]:
                assert top_pred["token_id"] < model.config.vocab_size
                assert top_pred["token_str"] != ""
                assert 0 <= top_pred["prob"] <= 1
        
        # Write to file
        writer.write_step(entry)
    
    # Read back and verify
    reader = TrainingDataReader(temp_jsonl_file)
    all_steps = reader.read_all()
    assert len(all_steps) == len(test_texts)
    
    # Verify the data would work with visualization
    for entry in all_steps:
        assert entry["metadata"]["model_name"] == model_name
        # Vocab size should match model's output dimension, not tokenizer's
        assert entry["metadata"]["vocab_size"] == model.config.vocab_size
        
        # Check that tokens are actual text (not just IDs)
        seq = entry["sequences"][0]
        reconstructed = "".join(seq["tokens"])
        assert len(reconstructed) > 0  # Should have actual text
        
        # Verify predictions look realistic
        if len(seq["predictions"]) > 0:
            first_pred = seq["predictions"][0]
            # Top prediction should have reasonable probability
            top_prob = first_pred["top_k"][0]["prob"]
            assert top_prob > 0.001  # Not completely random


def test_huggingface_trainer_integration(temp_jsonl_file):
    """Test the HuggingFace Trainer integration with callbacks and compute_metrics."""
    try:
        from transformers import (
            AutoTokenizer, 
            AutoModelForCausalLM,
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling
        )
        from datasets import Dataset
    except ImportError:
        pytest.skip("transformers/datasets not installed")
    
    from seq_viz.integrations import create_seq_viz_integration
    
    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")
    
    # Create a small dataset
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "To be or not to be, that is the question.",
        "Python is a versatile programming language.",
    ]
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=16,  # Keep short for testing
        )
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    
    # Split into train/eval
    split = tokenized_dataset.train_test_split(test_size=0.5, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    
    # Create visualization integration
    callback, compute_metrics = create_seq_viz_integration(
        output_file=temp_jsonl_file,
        tokenizer=tokenizer,
        max_sequences_per_eval=2,  # Only visualize 2 sequences per eval
        model_name=model_name
    )
    
    # Test that we can also chain with existing compute_metrics
    def custom_compute_metrics(eval_pred):
        """Custom metrics to test chaining."""
        return {"custom_metric": 42.0}
    
    # Create integration with existing metrics
    callback_with_custom, compute_metrics_with_custom = create_seq_viz_integration(
        output_file=temp_jsonl_file.replace('.jsonl', '_custom.jsonl'),
        tokenizer=tokenizer,
        existing_compute_metrics=custom_compute_metrics,
        max_sequences_per_eval=1,
        model_name=model_name
    )
    
    # Training arguments (minimal for testing)
    training_args = TrainingArguments(
        output_dir="./test_trainer_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=1,  # Evaluate after every step for testing
        logging_steps=1,
        save_steps=1000,  # Don't save during test
        report_to="none",
        push_to_hub=False,
    )
    
    # Create trainer with visualization
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
        callbacks=[callback],
        compute_metrics=compute_metrics,
    )
    
    # Train for one step (just to trigger evaluation)
    trainer.train(resume_from_checkpoint=False)
    
    # Check that visualization data was written
    reader = TrainingDataReader(temp_jsonl_file)
    all_steps = reader.read_all()
    
    # Should have at least one evaluation step
    assert len(all_steps) > 0
    
    for entry in all_steps:
        # Check metadata
        assert entry["metadata"]["model_name"] == model_name
        assert entry["metadata"]["vocab_size"] == model.config.vocab_size
        
        # Check sequences were captured
        assert len(entry["sequences"]) > 0
        assert len(entry["sequences"]) <= 2  # max_sequences_per_eval
        
        for seq in entry["sequences"]:
            # Verify tokens and predictions
            assert len(seq["tokens"]) > 0
            assert len(seq["token_ids"]) == len(seq["tokens"])
            
            # Check predictions exist
            if len(seq["predictions"]) > 0:
                pred = seq["predictions"][0]
                assert "position" in pred
                assert "target_token_id" in pred
                assert "entropy" in pred
                assert len(pred["top_k"]) == 5
                assert len(pred["top_20"]) == 20
    
    # Test the version with custom metrics
    trainer_with_custom = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
        callbacks=[callback_with_custom],
        compute_metrics=compute_metrics_with_custom,
    )
    
    # Evaluate to trigger compute_metrics
    eval_results = trainer_with_custom.evaluate()
    
    # Check that custom metric was preserved
    assert "eval_custom_metric" in eval_results
    assert eval_results["eval_custom_metric"] == 42.0
    
    # Check that visualization data was also written
    custom_file = temp_jsonl_file.replace('.jsonl', '_custom.jsonl')
    if Path(custom_file).exists():
        reader_custom = TrainingDataReader(custom_file)
        custom_steps = reader_custom.read_all()
        assert len(custom_steps) > 0
        
        # Clean up
        Path(custom_file).unlink(missing_ok=True)
    
    # Clean up trainer output
    import shutil
    if Path("./test_trainer_output").exists():
        shutil.rmtree("./test_trainer_output")