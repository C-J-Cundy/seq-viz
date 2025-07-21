import asyncio
import json
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import websockets
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)

warnings.filterwarnings("ignore")


class ShakespeareDataset(Dataset):
    def __init__(self, text_path, tokenizer, max_length=128):
        with open(text_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        self.tokenizer = tokenizer
        self.max_length = max_length

        # Tokenize the entire text
        self.tokens = tokenizer.encode(self.text)
        print(f"Dataset has {len(self.tokens)} tokens")

    def __len__(self):
        return len(self.tokens) - self.max_length

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.max_length + 1]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, labels


class LlamaShakespeareTrainer:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        print(f"Loading {model_name}...")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="eager",  # To get attention weights
        )

        # Add pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Train the whole model
        for param in self.model.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"Training {trainable_params:,} / {total_params:,} parameters ({100 * trainable_params / total_params:.1f}%)"
        )

        # Create dataset
        self.dataset = ShakespeareDataset("shakespeare.txt", self.tokenizer, max_length=64)
        self.dataloader = DataLoader(
            self.dataset, batch_size=4, shuffle=True
        )  # Increased batch size
        self.data_iter = iter(self.dataloader)

        # Optimizer with lower learning rate for full model training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-5, weight_decay=0.01  # Lower LR for full model training
        )

        self.step = 0
        self.device = next(self.model.parameters()).device

    def get_batch(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            return next(self.data_iter)

    def train_step(self):
        self.model.train()

        # Get batch
        input_ids, labels = self.get_batch()
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids, labels=labels, output_attentions=True, return_dict=True
        )

        loss = outputs.loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.step += 1

        # Get visualization data
        with torch.no_grad():
            batch_size = input_ids.shape[0]
            sequences = []

            # Process each sequence in the batch
            for batch_idx in range(batch_size):
                # Decode tokens for this sequence
                token_strings = [
                    self.tokenizer.decode([tid], skip_special_tokens=False)
                    for tid in input_ids[batch_idx].cpu().tolist()
                ]

                # Get predictions for this sequence
                logits = outputs.logits[batch_idx]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=5, dim=-1)
                # Also get top 20 for visualization
                top20_probs, top20_indices = torch.topk(probs, k=20, dim=-1)

                predictions = []
                for i in range(len(token_strings)):
                    pred_tokens = [self.tokenizer.decode([idx.item()]) for idx in topk_indices[i]]
                    target = labels[batch_idx, i].item() if i < len(labels[batch_idx]) else -1

                    predictions.append(
                        {
                            "top_k_tokens": topk_indices[i].cpu().tolist(),
                            "top_k_probs": topk_probs[i].cpu().tolist(),
                            "top_k_strings": pred_tokens,
                            "target": target,
                            "top_20_tokens": top20_indices[i].cpu().tolist(),
                            "top_20_probs": top20_probs[i].cpu().tolist(),
                            "full_entropy": -(probs[i] * torch.log2(probs[i] + 1e-10)).sum().item(),
                            "vocab_size": probs.shape[-1],
                        }
                    )

                sequences.append({"tokens": token_strings, "predictions": predictions})

            # Extract attention (limit to first 4 layers for visualization)
            attention_data = []
            if outputs.attentions:
                for i in range(min(4, len(outputs.attentions))):
                    if outputs.attentions[i] is not None:
                        avg_attention = outputs.attentions[i].mean(dim=1)
                        attention_data.append(avg_attention[0].cpu().numpy().tolist())

        return {
            "loss": loss.item(),
            "perplexity": torch.exp(loss).item(),
            "step": self.step,
            "sequences": sequences,
            "attention_weights": attention_data,
            "model_config": {
                "n_layers": min(4, len(self.model.model.layers)),  # Show only first 4
                "n_heads": self.model.config.num_attention_heads,
                "d_model": self.model.config.hidden_size,
                "model_name": "Llama-3.2-1B (Shakespeare)",
            },
        }

    def generate_sample(self, prompt="ROMEO:", max_length=50):
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_length, temperature=0.8, do_sample=True, top_p=0.9
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# WebSocket server
async def visualization_handler(websocket):
    print("Client connected, initializing Llama trainer...")
    trainer = LlamaShakespeareTrainer()

    training_active = True

    async def training_loop():
        while training_active:
            try:
                # Train step
                result = trainer.train_step()

                # Send update
                await websocket.send(json.dumps({"type": "training_update", **result}))

                # Generate sample every 50 steps
                if trainer.step % 50 == 0:
                    sample = trainer.generate_sample()
                    print(f"\nStep {trainer.step}: {sample}\n")

                await asyncio.sleep(0.5)  # Update every 500ms

            except Exception as e:
                print(f"Error in training: {e}")
                break

    training_task = asyncio.create_task(training_loop())

    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get("type") == "control":
                if data.get("action") == "pause":
                    training_active = False
                elif data.get("action") == "resume":
                    training_active = True
                    if training_task.done():
                        training_task = asyncio.create_task(training_loop())

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        training_active = False
        training_task.cancel()


async def main():
    print("🎭 Starting Llama Shakespeare Fine-tuning Server")
    print("📚 Fine-tuning Llama 3.2 1B on Shakespeare")
    print("🔥 Watch the model adapt to Shakespearean style!")

    async with websockets.serve(visualization_handler, "localhost", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
