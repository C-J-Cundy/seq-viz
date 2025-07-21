import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator


class TrainingDataReader:
    """Simple reader for training data JSONL files."""
    
    def __init__(self, input_file: str = "training_data.jsonl"):
        self.input_file = Path(input_file)
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"File {input_file} not found")
    
    def read_all(self) -> List[Dict[str, Any]]:
        """Read all training steps from file."""
        steps = []
        with open(self.input_file, 'r') as f:
            for line in f:
                if line.strip():
                    steps.append(json.loads(line))
        return steps
    
    def iter_steps(self) -> Iterator[Dict[str, Any]]:
        """Iterate over training steps without loading all into memory."""
        with open(self.input_file, 'r') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    
    def get_step(self, step_number: int) -> Optional[Dict[str, Any]]:
        """Get a specific training step by step number."""
        for entry in self.iter_steps():
            if entry.get('step') == step_number:
                return entry
        return None
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the most recent training step."""
        latest = None
        for entry in self.iter_steps():
            latest = entry
        return latest
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the training data."""
        steps = []
        losses = []
        perplexities = []
        
        for entry in self.iter_steps():
            steps.append(entry['step'])
            losses.append(entry['loss'])
            perplexities.append(entry['perplexity'])
        
        if not steps:
            return {}
        
        return {
            'total_steps': len(steps),
            'first_step': min(steps),
            'last_step': max(steps),
            'avg_loss': sum(losses) / len(losses),
            'min_loss': min(losses),
            'max_loss': max(losses),
            'avg_perplexity': sum(perplexities) / len(perplexities),
            'min_perplexity': min(perplexities),
            'max_perplexity': max(perplexities)
        }