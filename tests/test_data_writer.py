"""Unit tests for TrainingDataWriter."""

import pytest
import json
import tempfile
from pathlib import Path
from seq_viz.core.data_writer import TrainingDataWriter


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def valid_entry():
    """Create a valid training data entry."""
    return {
        "step": 100,
        "loss": 2.5,
        "perplexity": 12.18,
        "sequences": [
            {
                "tokens": ["Hello", " world"],
                "token_ids": [1, 2],
                "predictions": [
                    {
                        "position": 0,
                        "target_token_id": 2,
                        "target_token_str": " world",
                        "top_k": [
                            {"token_id": 2, "token_str": " world", "prob": 0.8},
                            {"token_id": 3, "token_str": " there", "prob": 0.2}
                        ],
                        "top_20": [
                            {"token_id": 2, "token_str": " world", "prob": 0.8},
                            {"token_id": 3, "token_str": " there", "prob": 0.2}
                        ],
                        "entropy": 0.5
                    }
                ]
            }
        ],
        "metadata": {
            "model_name": "test-model",
            "vocab_size": 1000,
            "batch_size": 1,
            "sequence_length": 2
        }
    }


def test_write_single_entry(temp_file, valid_entry):
    """Test writing a single valid entry."""
    writer = TrainingDataWriter(temp_file)
    writer.write_step(valid_entry)
    
    # Read back and verify
    with open(temp_file, 'r') as f:
        lines = f.readlines()
    
    assert len(lines) == 1
    data = json.loads(lines[0])
    
    # Check key fields
    assert data["step"] == 100
    assert data["loss"] == 2.5
    assert "timestamp" in data  # Should be added automatically
    assert data["sequences"][0]["tokens"] == ["Hello", " world"]


def test_write_multiple_entries(temp_file, valid_entry):
    """Test writing multiple entries."""
    writer = TrainingDataWriter(temp_file)
    
    # Write 3 entries with different steps
    for i in range(3):
        entry = valid_entry.copy()
        entry["step"] = 100 + i
        writer.write_step(entry)
    
    # Read back and verify
    with open(temp_file, 'r') as f:
        lines = f.readlines()
    
    assert len(lines) == 3
    
    # Check each entry
    for i, line in enumerate(lines):
        data = json.loads(line)
        assert data["step"] == 100 + i


def test_timestamp_addition(temp_file, valid_entry):
    """Test that timestamp is automatically added."""
    writer = TrainingDataWriter(temp_file)
    
    # Entry without timestamp
    entry = valid_entry.copy()
    assert "timestamp" not in entry
    
    writer.write_step(entry)
    
    # Read back
    with open(temp_file, 'r') as f:
        data = json.load(f)
    
    assert "timestamp" in data
    assert isinstance(data["timestamp"], float)
    assert data["timestamp"] > 0


def test_invalid_entry_validation(temp_file):
    """Test that invalid entries are rejected."""
    writer = TrainingDataWriter(temp_file)
    
    # Missing required field (step)
    invalid_entry = {
        "loss": 2.5,
        "sequences": [],
        "metadata": {
            "model_name": "test",
            "vocab_size": 100,
            "batch_size": 1,
            "sequence_length": 10
        }
    }
    
    with pytest.raises(Exception) as exc_info:
        writer.write_step(invalid_entry)
    
    # Should mention the missing field
    assert "step" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()


def test_invalid_sequence_structure(temp_file):
    """Test validation of sequence structure."""
    writer = TrainingDataWriter(temp_file)
    
    # Invalid sequence - missing token_ids
    invalid_entry = {
        "step": 100,
        "loss": 2.5,
        "perplexity": 12.18,
        "sequences": [
            {
                "tokens": ["Hello", " world"],
                # "token_ids": [1, 2],  # Missing!
                "predictions": []
            }
        ],
        "metadata": {
            "model_name": "test",
            "vocab_size": 100,
            "batch_size": 1,
            "sequence_length": 2
        }
    }
    
    with pytest.raises(Exception):
        writer.write_step(invalid_entry)


def test_invalid_prediction_structure(temp_file):
    """Test validation of prediction structure."""
    writer = TrainingDataWriter(temp_file)
    
    invalid_entry = {
        "step": 100,
        "loss": 2.5,
        "perplexity": 12.18,
        "sequences": [
            {
                "tokens": ["Hello", " world"],
                "token_ids": [1, 2],
                "predictions": [
                    {
                        "position": 0,
                        # Missing required fields like target_token_id, top_k, etc.
                    }
                ]
            }
        ],
        "metadata": {
            "model_name": "test",
            "vocab_size": 100,
            "batch_size": 1,
            "sequence_length": 2
        }
    }
    
    with pytest.raises(Exception):
        writer.write_step(invalid_entry)


def test_append_mode(temp_file, valid_entry):
    """Test that writer appends to existing file."""
    # Write first entry
    writer1 = TrainingDataWriter(temp_file)
    entry1 = valid_entry.copy()
    entry1["step"] = 100
    writer1.write_step(entry1)
    
    # Create new writer and append
    writer2 = TrainingDataWriter(temp_file)
    entry2 = valid_entry.copy()
    entry2["step"] = 200
    writer2.write_step(entry2)
    
    # Should have both entries
    with open(temp_file, 'r') as f:
        lines = f.readlines()
    
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 100
    assert json.loads(lines[1])["step"] == 200


def test_empty_sequences(temp_file):
    """Test writing entry with empty sequences list."""
    writer = TrainingDataWriter(temp_file)
    
    entry = {
        "step": 100,
        "loss": 2.5,
        "perplexity": 12.18,
        "sequences": [],  # Empty is valid
        "metadata": {
            "model_name": "test",
            "vocab_size": 100,
            "batch_size": 0,
            "sequence_length": 0
        }
    }
    
    writer.write_step(entry)
    
    with open(temp_file, 'r') as f:
        data = json.loads(f.read())
    
    assert data["sequences"] == []
    assert data["step"] == 100


def test_large_vocabulary(temp_file):
    """Test handling large vocabulary sizes."""
    writer = TrainingDataWriter(temp_file)
    
    entry = {
        "step": 100,
        "loss": 2.5,
        "perplexity": 12.18,
        "sequences": [
            {
                "tokens": ["token"],
                "token_ids": [128255],  # Large token ID
                "predictions": []
            }
        ],
        "metadata": {
            "model_name": "large-model",
            "vocab_size": 128256,  # Large vocab
            "batch_size": 1,
            "sequence_length": 1
        }
    }
    
    writer.write_step(entry)
    
    with open(temp_file, 'r') as f:
        data = json.loads(f.read())
    
    assert data["metadata"]["vocab_size"] == 128256
    assert data["sequences"][0]["token_ids"][0] == 128255


def test_special_characters_in_tokens(temp_file):
    """Test handling special characters in token strings."""
    writer = TrainingDataWriter(temp_file)
    
    entry = {
        "step": 100,
        "loss": 2.5,
        "perplexity": 12.18,
        "sequences": [
            {
                "tokens": ["Hello\n", "\t world", "\"quoted\"", "\\backslash"],
                "token_ids": [1, 2, 3, 4],
                "predictions": []
            }
        ],
        "metadata": {
            "model_name": "test",
            "vocab_size": 100,
            "batch_size": 1,
            "sequence_length": 4
        }
    }
    
    writer.write_step(entry)
    
    with open(temp_file, 'r') as f:
        data = json.loads(f.read())
    
    # JSON should handle special characters correctly
    assert data["sequences"][0]["tokens"][0] == "Hello\n"
    assert data["sequences"][0]["tokens"][1] == "\t world"
    assert data["sequences"][0]["tokens"][2] == "\"quoted\""
    assert data["sequences"][0]["tokens"][3] == "\\backslash"