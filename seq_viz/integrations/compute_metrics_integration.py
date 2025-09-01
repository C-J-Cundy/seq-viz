"""HuggingFace Transformers integration using compute_metrics to capture predictions."""

import logging
import time
from typing import Optional, Callable, Dict, Any

import numpy as np
import torch
from transformers import TrainerCallback

from ..core import TrainingDataWriter, tensor_to_training_entry

logger = logging.getLogger(__name__)


class SeqVizIntegration:
    """
    Integration with HuggingFace Trainers that captures model predictions during evaluation.
    
    This works by combining a lightweight callback (for tracking training state) with a
    compute_metrics wrapper (for processing predictions), connected through shared state.
    
    Args:
        output_file: Path to the JSONL file for storing visualization data
        tokenizer: The tokenizer to use for decoding tokens
        max_sequences_per_eval: Maximum number of sequences to capture per evaluation
        model_name: Optional model name to include in metadata
    """
    
    def __init__(
        self,
        output_file: str,
        tokenizer,
        max_sequences_per_eval: int = 4,
        model_name: Optional[str] = None
    ):
        self.output_file = output_file
        self.tokenizer = tokenizer
        self.max_sequences_per_eval = max_sequences_per_eval
        self.model_name = model_name
        
        # Shared state between callback and compute_metrics
        self.current_step = 0
        self.current_loss = 0.0
        self.vocab_size = None
        
        # Initialize writer
        self.writer = TrainingDataWriter(output_file)
        self._sequences_written = 0
        
    def create_callback(self) -> TrainerCallback:
        """
        Creates a callback that tracks training state during evaluation.
        """
        integration = self  # Capture self for closure
        
        class StateTrackingCallback(TrainerCallback):
            """Tracks training state for use in compute_metrics."""
            
            def _setup_model_and_args(self, args, model):
                """Common setup for model metadata and args."""
                if model is not None and integration.model_name is None:
                    # Get model name if not already set
                    if hasattr(model, "config") and hasattr(model.config, "name_or_path"):
                        integration.model_name = model.config.name_or_path
                    else:
                        integration.model_name = model.__class__.__name__
                    
                    # Get vocab size
                    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
                        integration.vocab_size = model.config.vocab_size
                
                # IMPORTANT: Enable inputs in compute_metrics so we get actual input_ids
                if args and hasattr(args, 'include_for_metrics'):
                    if 'inputs' not in args.include_for_metrics:
                        args.include_for_metrics.append('inputs')
                        logger.info("Enabled 'inputs' in include_for_metrics to get actual input_ids instead of labels")
            
            def on_init_end(self, args, state, control, model=None, **kwargs):
                """Called when Trainer is initialized."""
                self._setup_model_and_args(args, model)
            
            def on_train_begin(self, args, state, control, model=None, **kwargs):
                """Capture model metadata at training start."""
                self._setup_model_and_args(args, model)
            
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                """Update shared state with current training context."""
                integration.current_step = state.global_step
                if metrics:
                    integration.current_loss = metrics.get("eval_loss", metrics.get("loss", 0.0))
                
        return StateTrackingCallback()
    
    def create_compute_metrics(
        self, 
        original_compute_metrics: Optional[Callable] = None
    ) -> Callable:
        """
        Creates a compute_metrics function that saves predictions for visualization.
        
        This wrapper receives the predictions that Trainer has already computed,
        processes them for visualization, and calls any existing compute_metrics.
        
        Args:
            original_compute_metrics: Optional existing compute_metrics function to preserve
            
        Returns:
            A compute_metrics function compatible with HuggingFace Trainer
        """
        integration = self  # Capture self for closure
        
        def compute_metrics_with_seq_viz(eval_prediction) -> Dict[str, float]:
            """
            Process predictions for visualization and compute metrics.
            
            Args:
                eval_prediction: EvalPrediction object with:
                    - predictions: numpy array of model outputs (logits)
                    - label_ids: numpy array of input token ids
            """
            try:
                # Extract predictions from the evaluation
                predictions = eval_prediction.predictions
                label_ids = eval_prediction.label_ids
                
                # Prefer actual input_ids if available (cleaner, no -100 padding markers)
                if hasattr(eval_prediction, 'inputs') and eval_prediction.inputs is not None:
                    input_ids = eval_prediction.inputs
                    logger.info(f"Using actual input_ids from eval_prediction.inputs (shape: {input_ids.shape})")
                    logger.debug(f"Sample input_ids: {input_ids[0][:10] if len(input_ids) > 0 else 'empty'}")
                else:
                    # Fallback to label_ids when input_ids not available
                    # Note: label_ids may contain -100 to mark padding for loss masking
                    input_ids = label_ids
                    logger.info("Input_ids not available, using label_ids (may contain -100 for padding)")
                    logger.debug(f"Sample label_ids: {label_ids[0][:10] if len(label_ids) > 0 else 'empty'}")
                
                # Handle different prediction formats
                if isinstance(predictions, tuple):
                    logits = predictions[0] if len(predictions) > 0 else None
                else:
                    logits = predictions
                
                # Skip if no valid logits or wrong shape
                if logits is None or len(logits) == 0 or len(logits.shape) != 3:
                    if original_compute_metrics:
                        return original_compute_metrics(eval_prediction)
                    return {}
                
                # Convert numpy arrays to torch tensors
                logits_tensor = torch.from_numpy(logits)
                input_ids_tensor = torch.from_numpy(input_ids)
                
                # Limit the number of sequences to process
                num_sequences = min(len(logits), integration.max_sequences_per_eval)
                
                # Use tensor_to_training_entry to process the data
                # It handles all the computation: entropy, per-token loss, top-k predictions
                entry = tensor_to_training_entry(
                    logits=logits_tensor[:num_sequences],
                    input_ids=input_ids_tensor[:num_sequences],
                    tokenizer=integration.tokenizer,
                    step=integration.current_step,
                    loss=integration.current_loss,
                    model_name=integration.model_name or "unknown",
                    pad_token_id=integration.tokenizer.pad_token_id if hasattr(integration.tokenizer, 'pad_token_id') else None
                )
                
                # Write the visualization data
                if entry:
                    integration.writer.write_step(entry)
                    sequences_count = len(entry.get("sequences", []))
                    integration._sequences_written += sequences_count
                    logger.info(
                        f"Saved {sequences_count} sequences at step {integration.current_step} "
                        f"(total: {integration._sequences_written})"
                    )
                
            except Exception as e:
                # Log error but don't break training
                logger.error(f"Error in compute_metrics: {e}", exc_info=True)
            
            # Call original compute_metrics if it exists
            if original_compute_metrics:
                return original_compute_metrics(eval_prediction)
            
            # Return empty metrics if no original function
            return {}
        
        return compute_metrics_with_seq_viz


def create_seq_viz_integration(
    output_file: str,
    tokenizer,
    existing_compute_metrics: Optional[Callable] = None,
    max_sequences_per_eval: int = 4,
    model_name: Optional[str] = None
) -> tuple:
    """
    Create a sequence visualization integration for HuggingFace Trainers.
    
    This function returns a callback and compute_metrics function that work together
    to capture and save model predictions during evaluation for visualization.
    
    Args:
        output_file: Path to save visualization data
        tokenizer: Tokenizer for decoding tokens  
        existing_compute_metrics: Optional existing compute_metrics to preserve
        max_sequences_per_eval: Number of sequences to visualize per evaluation
        model_name: Optional model name for metadata
        
    Returns:
        Tuple of (callback, compute_metrics) to pass to Trainer
        
    Example:
        ```python
        from seq_viz.integrations import create_seq_viz_integration
        
        # Create integration components
        callback, compute_metrics = create_seq_viz_integration(
            output_file="training_viz.jsonl",
            tokenizer=tokenizer,
            existing_compute_metrics=my_metrics_fn  # Optional
        )
        
        # Use with any Trainer variant
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[callback],
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        ```
    """
    integration = SeqVizIntegration(
        output_file=output_file,
        tokenizer=tokenizer,
        max_sequences_per_eval=max_sequences_per_eval,
        model_name=model_name
    )
    
    return (
        integration.create_callback(),
        integration.create_compute_metrics(existing_compute_metrics)
    )