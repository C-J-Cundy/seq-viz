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
                        "loss": 0.223,  # Added required loss field
                        "top_k": [
                            {"token_id": 2, "token_str": " world", "prob": 0.8},
                            {"token_id": 3, "token_str": " there", "prob": 0.2},
                            {"token_id": 4, "token_str": " foo", "prob": 0.0},
                            {"token_id": 5, "token_str": " bar", "prob": 0.0},
                            {"token_id": 6, "token_str": " baz", "prob": 0.0}
                        ],
                        "top_20": [
                            {"token_id": 2, "token_str": " world", "prob": 0.8},
                            {"token_id": 3, "token_str": " there", "prob": 0.15},
                            {"token_id": 4, "token_str": " foo", "prob": 0.02},
                            {"token_id": 5, "token_str": " bar", "prob": 0.01},
                            {"token_id": 6, "token_str": " baz", "prob": 0.005},
                            {"token_id": 7, "token_str": " tok7", "prob": 0.004},
                            {"token_id": 8, "token_str": " tok8", "prob": 0.003},
                            {"token_id": 9, "token_str": " tok9", "prob": 0.002},
                            {"token_id": 10, "token_str": " tok10", "prob": 0.001},
                            {"token_id": 11, "token_str": " tok11", "prob": 0.001},
                            {"token_id": 12, "token_str": " tok12", "prob": 0.001},
                            {"token_id": 13, "token_str": " tok13", "prob": 0.001},
                            {"token_id": 14, "token_str": " tok14", "prob": 0.001},
                            {"token_id": 15, "token_str": " tok15", "prob": 0.001},
                            {"token_id": 16, "token_str": " tok16", "prob": 0.001},
                            {"token_id": 17, "token_str": " tok17", "prob": 0.001},
                            {"token_id": 18, "token_str": " tok18", "prob": 0.001},
                            {"token_id": 19, "token_str": " tok19", "prob": 0.001},
                            {"token_id": 20, "token_str": " tok20", "prob": 0.001},
                            {"token_id": 21, "token_str": " tok21", "prob": 0.001}
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
    
    # Writer should raise ValueError for invalid data
    with pytest.raises(ValueError, match="step.*required"):
        writer.write_step(invalid_entry)
    
    # File should be empty
    with open(temp_file, 'r') as f:
        content = f.read()
        assert content == ""


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
    
    # Should raise ValueError for invalid data
    with pytest.raises(ValueError, match="token_ids.*required"):
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
    
    # Should raise ValueError for invalid data
    with pytest.raises(ValueError, match="required"):
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
            "batch_size": 1,  # Must be >= 1
            "sequence_length": 1  # Must be >= 1
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