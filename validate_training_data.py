import json
import jsonschema
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_schema(schema_path: str = "training_data_schema.json") -> Dict[str, Any]:
    """Load the JSON schema from file."""
    with open(schema_path, 'r') as f:
        return json.load(f)


def validate_training_entry(entry: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate a single training data entry against the schema.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        jsonschema.validate(instance=entry, schema=schema)
        return True, ""
    except jsonschema.ValidationError as e:
        return False, str(e)


def validate_jsonl_file(file_path: str, schema_path: str = "training_data_schema.json") -> None:
    """Validate all entries in a JSONL file against the schema."""
    schema = load_schema(schema_path)
    
    valid_count = 0
    invalid_count = 0
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                is_valid, error_msg = validate_training_entry(entry, schema)
                
                if is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                    print(f"Line {line_num} - Validation error: {error_msg}")
                    
            except json.JSONDecodeError as e:
                invalid_count += 1
                print(f"Line {line_num} - JSON parse error: {e}")
    
    print(f"\nValidation complete:")
    print(f"  Valid entries: {valid_count}")
    print(f"  Invalid entries: {invalid_count}")


if __name__ == "__main__":
    import sys
    
    # Use command line argument or default to training_data.jsonl
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "training_data.jsonl"
    
    if Path(file_path).exists():
        print(f"Validating {file_path}...")
        validate_jsonl_file(file_path)
    else:
        print(f"No {file_path} file found.")