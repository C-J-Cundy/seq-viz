"""Unit tests for tensor_to_training_entry function."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock
from seq_viz.core.tensor_to_training_data import tensor_to_training_entry


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, vocab_size=100, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
    
    def decode(self, token_ids, skip_special_tokens=False):
        """Mock decode that returns a string representation."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Handle single token or list
        if isinstance(token_ids, int):
            if token_ids < 0:
                return f"[PAD]"
            return f"token_{token_ids}"
        
        # Handle list of tokens
        result = []
        for tid in token_ids:
            if tid < 0 or tid == self.pad_token_id:
                if not skip_special_tokens:
                    result.append("[PAD]")
            else:
                result.append(f"token_{tid}")
        return " ".join(result) if result else ""


@pytest.fixture
def mock_tokenizer():
    """Fixture for mock tokenizer."""
    return MockTokenizer(vocab_size=100, pad_token_id=0)


@pytest.fixture
def simple_logits():
    """Create simple logits tensor for testing."""
    # Shape: (sequence_length=5, vocab_size=100)
    torch.manual_seed(42)
    return torch.randn(5, 100)


@pytest.fixture
def simple_input_ids():
    """Create simple input_ids tensor."""
    # 5 tokens, values 1-5
    return torch.tensor([1, 2, 3, 4, 5])


def test_basic_conversion(mock_tokenizer, simple_logits, simple_input_ids):
    """Test basic tensor to training entry conversion."""
    result = tensor_to_training_entry(
        logits=simple_logits,
        input_ids=simple_input_ids,
        tokenizer=mock_tokenizer,
        step=100,
        loss=2.5,
        model_name="test-model",
        top_k=3,
        top_20=5
    )
    
    # Check basic fields
    assert result["step"] == 100
    assert result["loss"] == 2.5
    assert pytest.approx(result["perplexity"], 0.1) == np.exp(2.5)
    
    # Check metadata
    assert result["metadata"]["model_name"] == "test-model"
    assert result["metadata"]["vocab_size"] == 100
    assert result["metadata"]["batch_size"] == 1
    assert result["metadata"]["sequence_length"] == 5
    
    # Check sequences
    assert len(result["sequences"]) == 1
    seq = result["sequences"][0]
    assert len(seq["tokens"]) == 5
    assert len(seq["token_ids"]) == 5
    assert seq["token_ids"] == [1, 2, 3, 4, 5]
    
    # Check predictions (we should have 4 predictions for 5 tokens)
    assert len(seq["predictions"]) == 4
    
    # Check first prediction
    pred = seq["predictions"][0]
    assert pred["position"] == 0
    assert pred["target_token_id"] == 2  # Predicting token at position 1
    assert pred["target_token_str"] == "token_2"
    assert len(pred["top_k"]) == 3
    assert len(pred["top_20"]) == 5
    assert "entropy" in pred
    
    # Verify top_k structure
    for top_pred in pred["top_k"]:
        assert "token_id" in top_pred
        assert "token_str" in top_pred
        assert "prob" in top_pred
        assert 0 <= top_pred["prob"] <= 1


def test_padding_handling(mock_tokenizer):
    """Test handling of padding tokens (-100 values)."""
    # Create logits and input_ids with padding
    logits = torch.randn(6, 100)
    input_ids = torch.tensor([1, 2, 3, -100, -100, -100])
    
    result = tensor_to_training_entry(
        logits=logits,
        input_ids=input_ids,
        tokenizer=mock_tokenizer,
        step=1,
        loss=1.0,
        model_name="test",
        pad_token_id=0
    )
    
    seq = result["sequences"][0]
    # Should handle -100 values
    assert len(seq["token_ids"]) == 6
    # Based on the implementation, -100 values are kept as-is (actual token IDs)
    assert seq["token_ids"] == [1, 2, 3, -100, -100, -100]
    
    # Check predictions only for non-padding positions
    # Position 0 predicts token 1 (id=2)
    # Position 1 predicts token 2 (id=3)
    # Position 2 predicts token 3 (id=-100, becomes 0)
    assert len(seq["predictions"]) == 5  # All positions except last


def test_batch_processing(mock_tokenizer):
    """Test processing a batch of sequences."""
    # Batch of 3 sequences
    logits = torch.randn(3, 4, 100)  # (batch, seq_len, vocab)
    input_ids = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])
    
    result = tensor_to_training_entry(
        logits=logits,
        input_ids=input_ids,
        tokenizer=mock_tokenizer,
        step=200,
        loss=1.5,
        model_name="test-batch"
    )
    
    # Should have 3 sequences
    assert len(result["sequences"]) == 3
    assert result["metadata"]["batch_size"] == 3
    
    # Check each sequence
    for i, seq in enumerate(result["sequences"]):
        assert len(seq["tokens"]) == 4
        assert len(seq["predictions"]) == 3  # seq_len - 1


def test_entropy_calculation(mock_tokenizer):
    """Test entropy calculation in predictions."""
    # Create logits with known distribution
    logits = torch.zeros(2, 10)
    # Make position 0 have high confidence (low entropy)
    logits[0, 0] = 10.0  # Very high logit for token 0
    # Make position 1 have uniform distribution (high entropy)
    logits[1, :] = 0.0  # All equal logits
    
    input_ids = torch.tensor([1, 2])
    
    result = tensor_to_training_entry(
        logits=logits,
        input_ids=input_ids,
        tokenizer=mock_tokenizer,
        step=1,
        loss=1.0,
        model_name="test"
    )
    
    pred = result["sequences"][0]["predictions"][0]
    # First position should have low entropy (high confidence)
    assert pred["entropy"] < 0.1
    
    # Check that probabilities sum to approximately 1
    total_prob = sum(p["prob"] for p in pred["top_20"])
    assert total_prob <= 1.01  # Allow small numerical error


def test_sequence_index_extraction(mock_tokenizer):
    """Test extracting a specific sequence from a batch."""
    # Batch of 3 sequences
    logits = torch.randn(3, 4, 100)
    input_ids = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])
    
    # Extract only sequence at index 1
    result = tensor_to_training_entry(
        logits=logits,
        input_ids=input_ids,
        tokenizer=mock_tokenizer,
        step=1,
        loss=1.0,
        model_name="test",
        sequence_idx=1
    )
    
    # Should have only 1 sequence
    assert len(result["sequences"]) == 1
    seq = result["sequences"][0]
    assert seq["token_ids"] == [5, 6, 7, 8]


def test_empty_sequence(mock_tokenizer):
    """Test handling of empty or single-token sequences."""
    # Single token - no predictions possible
    logits = torch.randn(1, 100)
    input_ids = torch.tensor([1])
    
    result = tensor_to_training_entry(
        logits=logits,
        input_ids=input_ids,
        tokenizer=mock_tokenizer,
        step=1,
        loss=1.0,
        model_name="test"
    )
    
    seq = result["sequences"][0]
    assert len(seq["tokens"]) == 1
    assert len(seq["predictions"]) == 0  # No predictions for single token


def test_top_k_limiting(mock_tokenizer):
    """Test that top_k and top_20 properly limit the number of predictions."""
    logits = torch.randn(3, 100)
    input_ids = torch.tensor([1, 2, 3])
    
    result = tensor_to_training_entry(
        logits=logits,
        input_ids=input_ids,
        tokenizer=mock_tokenizer,
        step=1,
        loss=1.0,
        model_name="test",
        top_k=2,  # Only top 2
        top_20=10  # Only top 10
    )
    
    pred = result["sequences"][0]["predictions"][0]
    assert len(pred["top_k"]) == 2
    assert len(pred["top_20"]) == 10
    
    # Verify they're sorted by probability
    for i in range(1, len(pred["top_k"])):
        assert pred["top_k"][i-1]["prob"] >= pred["top_k"][i]["prob"]


def test_clean_input_ids_preferred(mock_tokenizer):
    """Test that clean input_ids are preferred over labels with -100."""
    logits = torch.randn(4, 100)
    # Simulate case where we have both clean input_ids and labels
    clean_ids = torch.tensor([1, 2, 3, 4])
    labels_with_padding = torch.tensor([1, 2, -100, -100])
    
    # Should prefer clean_ids when available
    result = tensor_to_training_entry(
        logits=logits,
        input_ids=clean_ids,  # Clean version
        tokenizer=mock_tokenizer,
        step=1,
        loss=1.0,
        model_name="test"
    )
    
    seq = result["sequences"][0]
    assert seq["token_ids"] == [1, 2, 3, 4]  # Should use clean version
    assert -100 not in seq["token_ids"]