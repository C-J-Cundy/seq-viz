import json
import time
from pathlib import Path
from typing import Dict, Any
from .validate_training_data import load_schema, validate_training_entry


class TrainingDataWriter:
    """Simple writer for training data that validates against schema."""
    
    def __init__(self, output_file: str = "training_data.jsonl", 
                 schema_path: str = None):
        self.output_file = Path(output_file)
        self.schema = load_schema(schema_path)
    
    def write_step(self, step_data: Dict[str, Any]) -> bool:
        """
        Write a training step to file after validation.
        
        Returns:
            True if write was successful, False otherwise
        """
        # Add timestamp if not present
        if 'timestamp' not in step_data:
            step_data['timestamp'] = time.time()
        
        # Validate against schema
        is_valid, error_msg = validate_training_entry(step_data, self.schema)
        
        if not is_valid:
            print(f"Validation error: {error_msg}")
            return False
        
        # Write to file
        with open(self.output_file, 'a') as f:
            f.write(json.dumps(step_data) + '\n')
        
        return True