"""HuggingFace Transformers callback for sequence visualization."""

import time

import numpy as np
import torch
from transformers import TrainerCallback

from ..core import TrainingDataWriter, tensor_to_training_entry


class VisualizationCallback(TrainerCallback):
    """
    A TrainerCallback that captures model predictions during evaluation
    and saves them to a JSONL file for visualization.

    Args:
        output_file: Path to the JSONL file for storing visualization data
        max_sequences_per_eval: Maximum number of sequences to capture per evaluation
        capture_train_steps: Whether to also capture training step predictions
        train_step_interval: If capturing training, how often (every N steps)
        tokenizer: The tokenizer to use for decoding tokens (required)
    """

    def __init__(
        self,
        output_file: str,
        max_sequences_per_eval: int = 4,
        capture_train_steps: bool = False,
        train_step_interval: int = 100,
        tokenizer=None,
    ):
        self.output_file = output_file
        self.max_sequences_per_eval = max_sequences_per_eval
        self.capture_train_steps = capture_train_steps
        self.train_step_interval = train_step_interval

        self.writer = TrainingDataWriter(output_file)
        self.model = None
        self.tokenizer = tokenizer  # Allow passing tokenizer at init
        self.model_name = None
        print("initialized writer!")

    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics=None,
        model=None,
        tokenizer=None,
        eval_dataloader=None,
        **kwargs,
    ):
        """Called after evaluation phase."""
        # Store model and tokenizer references
        if model is not None:
            self.model = model
            if hasattr(model, "config") and hasattr(model.config, "name_or_path"):
                self.model_name = model.config.name_or_path
            else:
                self.model_name = model.__class__.__name__

        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif self.tokenizer is None:
            raise ValueError(
                "No tokenizer available. Please pass tokenizer when creating VisualizationCallback: "
                "VisualizationCallback(output_file='...', tokenizer=tokenizer)"
            )

        # Skip if we don't have necessary components
        if self.model is None or eval_dataloader is None:
            return

        try:
            # Debug: print metrics keys
            # if metrics:
            #     print(f"VisualizationCallback: Available metrics keys: {list(metrics.keys())}")
            #     print(f"VisualizationCallback: Metrics values: {metrics}")
            # else:
            #     print("VisualizationCallback: No metrics received!")

            # Extract current metrics - try both 'loss' and 'eval_loss'
            current_loss = 0.0
            if metrics:
                current_loss = metrics.get("eval_loss", metrics.get("loss", 0.0))
            current_step = state.global_step

            # Get vocab size from model
            vocab_size = self.model.config.vocab_size

            # Process sequences and collect all data
            all_sequences = []
            sequences_processed = 0

            # Get a few sequences from the evaluation dataloader
            for batch in eval_dataloader:
                if sequences_processed >= self.max_sequences_per_eval:
                    break

                # Move batch to device
                device = self.model.device
                input_ids = batch["input_ids"].to(device)

                # Get model predictions
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs.logits

                # Process each sequence in the batch
                batch_size = min(len(input_ids), self.max_sequences_per_eval - sequences_processed)

                for i in range(batch_size):
                    # Create entry for this sequence
                    entry = tensor_to_training_entry(
                        logits=logits[i : i + 1],  # Single sequence with batch dimension
                        input_ids=input_ids[i : i + 1],  # Single sequence with batch dimension
                        tokenizer=self.tokenizer,
                        step=current_step,
                        loss=current_loss,
                        model_name=self.model_name,
                    )

                    if entry and "sequences" in entry:
                        all_sequences.extend(entry["sequences"])
                        sequences_processed += 1

                # Break if we've processed enough sequences
                if sequences_processed >= self.max_sequences_per_eval:
                    break

            # Create combined entry for all sequences
            if all_sequences:
                combined_entry = {
                    "timestamp": time.time(),
                    "step": current_step,
                    "loss": current_loss,
                    "perplexity": float(np.exp(current_loss)),
                    "sequences": all_sequences,
                    "metadata": {
                        "model_name": self.model_name,
                        "vocab_size": vocab_size,
                        "batch_size": len(all_sequences),
                        "sequence_length": (
                            all_sequences[0]["predictions"][-1]["position"] + 1
                            if all_sequences
                            else 0
                        ),
                    },
                }

                # Write to file
                self.writer.write_step(combined_entry)
                print(
                    f"VisualizationCallback: Saved {len(all_sequences)} sequences at step {current_step}"
                )

        except Exception as e:
            print(f"VisualizationCallback error during evaluation: {e}")
            import traceback

            traceback.print_exc()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called after logging metrics."""
        # Optionally capture training steps
        if (
            self.capture_train_steps
            and state.global_step % self.train_step_interval == 0
            and logs
            and "loss" in logs
        ):

            # For training steps, we'd need to hook into the training loop differently
            # This is more complex and might require a different approach
            # For now, we'll focus on evaluation data
            pass
