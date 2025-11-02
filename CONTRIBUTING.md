# Contributing to DELIGHT

Thank you for your interest in contributing to DELIGHT! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, PyTorch version, CUDA version)
- Any relevant error messages or logs

### Suggesting Enhancements

We welcome feature requests and enhancement suggestions! Please open an issue with:
- A clear description of the proposed feature
- Use cases and benefits
- Any implementation ideas you might have

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes**:
   - Follow the existing code style
   - Add docstrings to new functions and classes
   - Include type hints where appropriate
   - Update documentation if needed
3. **Test your changes**:
   - Ensure existing tests pass
   - Add new tests for new features
   - Test on multiple configurations if possible
4. **Commit your changes**:
   - Use clear, descriptive commit messages
   - Reference related issues (e.g., "Fixes #123")
5. **Submit a pull request**:
   - Provide a clear description of the changes
   - Link to any related issues
   - Explain the motivation and context

## Development Setup

### 1. Clone and setup

```bash
git clone https://github.com/alifuatsahin/DELIGHT.git
cd DELIGHT
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install development dependencies

```bash
pip install -r requirements-dev.txt  # If available
```

### 3. Compile third-party extensions

```bash
# ChamferDistance
cd third_party/ChamferDistancePytorch/chamfer3D
python setup.py install
cd ../../..

# EMD
cd third_party/PyTorchEMD
python setup.py install
cd ../..
```

## Code Style Guidelines

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use meaningful variable and function names
- Keep functions focused and modular
- Maximum line length: 100 characters (flexible for readability)

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """Brief description of function.
    
    Longer description if needed, explaining the function's behavior,
    algorithms used, or any important notes.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When input is invalid
    """
    pass
```

### Type Hints

Use type hints for function signatures:

```python
from typing import List, Dict, Optional, Tuple

def process_data(
    data: torch.Tensor,
    config: Dict[str, any],
    normalize: bool = True
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_vae.py

# Run with coverage
python -m pytest --cov=. tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_<module>.py`
- Name test functions as `test_<functionality>()`
- Use fixtures for common setup
- Test edge cases and error conditions

Example:

```python
import pytest
import torch
from models.vae import VAE

def test_vae_forward():
    """Test VAE forward pass."""
    cfg = get_test_config()
    model = VAE(cfg)
    x = torch.randn(2, 2048, 3)
    output = model(x, x)
    assert 'loss' in output
    assert output['loss'].shape == torch.Size([])
```

## Documentation

- Update README.md for significant changes
- Add docstrings to all new classes and functions
- Update configuration documentation for new options
- Include code examples for new features

## Commit Message Guidelines

Use clear, descriptive commit messages:

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat: Add support for multi-scale point cloud encoding

Implement multi-scale encoder that processes point clouds at different
resolutions for better feature extraction.

Closes #123
```

## Review Process

1. All pull requests require review before merging
2. Reviewers will check:
   - Code quality and style
   - Test coverage
   - Documentation completeness
   - Backward compatibility
3. Address review comments by pushing new commits
4. Once approved, maintainers will merge your PR

## Questions?

If you have questions about contributing:
- Open an issue with the "question" label
- Reach out to the maintainers
- Check existing issues and documentation

## License

By contributing to DELIGHT, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to DELIGHT! 🎉
