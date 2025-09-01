# Guidelines for Claude AI Assistant

This document provides guidelines for Claude when working on the seq-viz project.

## Project Overview

seq-viz is a real-time visualization system for transformer model token predictions during training. It integrates with HuggingFace Transformers to capture and visualize model predictions without recomputing logits.

## Development Guidelines

### Before Making Changes

1. **Understand the existing code**: Read relevant files before modifying
2. **Check for existing patterns**: Follow the codebase's conventions
3. **Test your understanding**: Run existing code before changing it

### Testing Requirements

**IMPORTANT: Always run tests before creating a pull request!**

```bash
# Run all tests (required before PR)
pytest tests/

# If you modify core functionality, run specific tests
pytest tests/test_tensor_to_training_data.py  # For tensor conversion changes
pytest tests/test_data_writer.py              # For data writing changes
pytest tests/test_data_reader.py              # For data reading changes
pytest tests/test_integration.py              # For integration changes

# Quick check - should see "36 passed" or more
pytest tests/ -q
```

### Pull Request Checklist

Before creating a PR, ensure:
- [ ] All tests pass (`pytest tests/`)
- [ ] New features have corresponding tests
- [ ] Code follows existing patterns
- [ ] Commit messages are descriptive
- [ ] No generated files are committed (*.jsonl, model outputs, etc.)

### Code Style

- Use descriptive variable names
- Keep functions focused and single-purpose
- Add type hints where helpful
- Follow existing patterns in the codebase
- Don't add comments unless necessary (code should be self-documenting)

### Common Tasks

#### Adding a new integration
1. Look at existing integrations in `seq_viz/integrations/`
2. Follow the pattern of `create_seq_viz_integration()`
3. Add tests in `test_integration.py`
4. Update examples if needed

#### Modifying the schema
1. Update `seq_viz/core/training_data_schema.json`
2. Update `tensor_to_training_entry()` in `tensor_to_training_data.py`
3. Fix all tests that use the schema
4. Update examples to match

#### Debugging visualization issues
1. Check the JSONL file is valid: `python -m seq_viz.core.validate_training_data <file>`
2. Look at WebSocket messages in browser console
3. Check server output for errors

### Performance Considerations

- The integration should NOT recompute logits (use existing predictions)
- Minimize overhead during training
- Use efficient data structures
- Limit visualization data per evaluation step

### Common Pitfalls to Avoid

1. **Don't recompute model outputs** - Use predictions from Trainer's evaluation
2. **Don't ignore padding tokens** - Handle -100 values properly
3. **Don't assume file paths** - Use absolute paths
4. **Don't commit large files** - Check .gitignore
5. **Don't skip tests** - Always run before pushing

### Useful Commands

```bash
# Install for development
pip install -e ".[dev,examples]"

# Run specific test with output
pytest tests/test_integration.py::test_pipeline_with_real_llama_model -xvs

# Check what would be committed
git status
git diff --staged

# Run the visualization server
python run_server.py --file <jsonl_file>
```

### Architecture Notes

- **Core**: Data processing (`tensor_to_training_entry`, `TrainingDataWriter/Reader`)
- **Integrations**: HuggingFace hooks (`create_seq_viz_integration`)
- **Server**: WebSocket server for streaming updates
- **Web**: JavaScript dashboard for visualization

### When in Doubt

1. Look at existing examples in the codebase
2. Run tests to verify behavior
3. Check the integration tests for usage patterns
4. Keep changes focused and minimal