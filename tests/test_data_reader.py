"""Unit tests for TrainingDataReader."""

import pytest
import json
import tempfile
from pathlib import Path
from seq_viz.core.data_reader import TrainingDataReader


@pytest.fixture
def temp_file_with_data():
    """Create a temporary file with sample training data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # Write 3 sample entries
        entries = [
            {
                "timestamp": 1000.0,
                "step": 100,
                "loss": 3.0,
                "perplexity": 20.09,
                "sequences": [{"tokens": ["a"], "token_ids": [1], "predictions": []}],
                "metadata": {"model_name": "test", "vocab_size": 100, "batch_size": 1, "sequence_length": 1}
            },
            {
                "timestamp": 1001.0,
                "step": 200,
                "loss": 2.5,
                "perplexity": 12.18,
                "sequences": [{"tokens": ["b"], "token_ids": [2], "predictions": []}],
                "metadata": {"model_name": "test", "vocab_size": 100, "batch_size": 1, "sequence_length": 1}
            },
            {
                "timestamp": 1002.0,
                "step": 300,
                "loss": 2.0,
                "perplexity": 7.39,
                "sequences": [{"tokens": ["c"], "token_ids": [3], "predictions": []}],
                "metadata": {"model_name": "test", "vocab_size": 100, "batch_size": 1, "sequence_length": 1}
            }
        ]
        
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
        
        temp_path = f.name
    
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def empty_file():
    """Create an empty temporary file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


def test_read_all_steps(temp_file_with_data):
    """Test reading all steps from file."""
    reader = TrainingDataReader(temp_file_with_data)
    all_steps = reader.read_all()
    
    assert len(all_steps) == 3
    assert all_steps[0]["step"] == 100
    assert all_steps[1]["step"] == 200
    assert all_steps[2]["step"] == 300


def test_get_specific_step(temp_file_with_data):
    """Test getting a specific step."""
    reader = TrainingDataReader(temp_file_with_data)
    
    # Get step 200
    step_200 = reader.get_step(200)
    assert step_200 is not None
    assert step_200["step"] == 200
    assert step_200["loss"] == 2.5
    
    # Try non-existent step
    step_999 = reader.get_step(999)
    assert step_999 is None


def test_get_latest_step(temp_file_with_data):
    """Test getting the latest step."""
    reader = TrainingDataReader(temp_file_with_data)
    latest = reader.get_latest()
    
    assert latest is not None
    assert latest["step"] == 300  # Last step in file
    assert latest["loss"] == 2.0


def test_iterate_steps(temp_file_with_data):
    """Test iterating through steps."""
    reader = TrainingDataReader(temp_file_with_data)
    
    steps = list(reader.iter_steps())
    assert len(steps) == 3
    
    # Check they're in order
    assert steps[0]["step"] == 100
    assert steps[1]["step"] == 200
    assert steps[2]["step"] == 300


def test_get_summary(temp_file_with_data):
    """Test getting summary statistics."""
    reader = TrainingDataReader(temp_file_with_data)
    summary = reader.get_summary()
    
    assert summary["total_steps"] == 3
    assert summary["first_step"] == 100
    assert summary["last_step"] == 300
    assert summary["avg_loss"] == pytest.approx(2.5, 0.01)  # (3.0 + 2.5 + 2.0) / 3
    assert summary["min_loss"] == 2.0
    assert summary["max_loss"] == 3.0
    assert summary["avg_perplexity"] == pytest.approx(13.22, 0.01)  # (20.09 + 12.18 + 7.39) / 3
    assert summary["min_perplexity"] == pytest.approx(7.39, 0.01)
    assert summary["max_perplexity"] == pytest.approx(20.09, 0.01)


def test_empty_file_handling(empty_file):
    """Test handling of empty file."""
    reader = TrainingDataReader(empty_file)
    
    assert reader.read_all() == []
    assert reader.get_latest() is None
    assert reader.get_step(100) is None
    assert list(reader.iter_steps()) == []
    
    summary = reader.get_summary()
    assert summary == {}  # Empty dict for empty file


def test_nonexistent_file():
    """Test handling of non-existent file."""
    # TrainingDataReader raises FileNotFoundError for non-existent files
    with pytest.raises(FileNotFoundError):
        reader = TrainingDataReader("/path/that/does/not/exist.jsonl")


def test_corrupted_file():
    """Test handling of corrupted JSONL file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # Write valid entry
        f.write(json.dumps({"step": 100, "loss": 2.0, "sequences": [], 
                          "metadata": {"model_name": "test", "vocab_size": 100, 
                                     "batch_size": 1, "sequence_length": 1}}) + '\n')
        # Write corrupted entry
        f.write("this is not valid json\n")
        # Write another valid entry
        f.write(json.dumps({"step": 200, "loss": 3.0, "sequences": [],
                          "metadata": {"model_name": "test", "vocab_size": 100, 
                                     "batch_size": 1, "sequence_length": 1}}) + '\n')
        temp_path = f.name
    
    try:
        reader = TrainingDataReader(temp_path)
        all_steps = reader.read_all()
        
        # Should skip corrupted entry and read valid ones
        assert len(all_steps) == 2
        assert all_steps[0]["step"] == 100
        assert all_steps[1]["step"] == 200
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_get_step_range(temp_file_with_data):
    """Test getting steps within a range."""
    reader = TrainingDataReader(temp_file_with_data)
    
    # This method might not exist yet, but would be useful
    # Just test that we can filter steps manually
    all_steps = reader.read_all()
    steps_in_range = [s for s in all_steps if 150 <= s["step"] <= 250]
    
    assert len(steps_in_range) == 1
    assert steps_in_range[0]["step"] == 200


def test_large_file_iteration():
    """Test iterating through a larger file efficiently."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # Write 100 entries
        for i in range(100):
            entry = {
                "timestamp": 1000.0 + i,
                "step": i * 10,
                "loss": 3.0 - (i * 0.01),  # Decreasing loss
                "perplexity": 20.0 - (i * 0.1),
                "sequences": [],
                "metadata": {"model_name": "test", "vocab_size": 100, 
                           "batch_size": 1, "sequence_length": 1}
            }
            f.write(json.dumps(entry) + '\n')
        temp_path = f.name
    
    try:
        reader = TrainingDataReader(temp_path)
        
        # Count steps using iterator
        count = sum(1 for _ in reader.iter_steps())
        assert count == 100
        
        # Check summary
        summary = reader.get_summary()
        assert summary["total_steps"] == 100
        assert summary["first_step"] == 0
        assert summary["last_step"] == 990
        assert summary["min_loss"] == pytest.approx(2.01, 0.01)
        assert summary["max_loss"] == pytest.approx(3.0, 0.01)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_duplicate_steps(temp_file_with_data):
    """Test handling files with duplicate step numbers."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # Write entries with duplicate steps
        entries = [
            {"step": 100, "loss": 2.0, "sequences": [], "timestamp": 1000.0,
             "metadata": {"model_name": "test", "vocab_size": 100, "batch_size": 1, "sequence_length": 1}},
            {"step": 100, "loss": 2.5, "sequences": [], "timestamp": 1001.0,  # Duplicate step
             "metadata": {"model_name": "test", "vocab_size": 100, "batch_size": 1, "sequence_length": 1}},
            {"step": 200, "loss": 3.0, "sequences": [], "timestamp": 1002.0,
             "metadata": {"model_name": "test", "vocab_size": 100, "batch_size": 1, "sequence_length": 1}}
        ]
        
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
        temp_path = f.name
    
    try:
        reader = TrainingDataReader(temp_path)
        
        # get_step should return the first occurrence
        step_100 = reader.get_step(100)
        assert step_100["loss"] == 2.0  # First one
        
        # read_all should return all entries
        all_steps = reader.read_all()
        assert len(all_steps) == 3
    finally:
        Path(temp_path).unlink(missing_ok=True)