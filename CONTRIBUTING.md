# Contributing to FastReID

Thank you for your interest in contributing to FastReID! This guide will help you get started.

## Development Setup

1. **Fork the repository** and clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/fast-reid.git
   cd fast-reid
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e .[dev]
   ```

## Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Test your changes**:
   ```bash
   python -m pytest tests/
   python test_model_compatibility.py  # If applicable
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to new functions and classes
- Keep line length under 100 characters

## Testing

- Add tests for new functionality
- Ensure all existing tests pass
- Test with different Python versions (3.8-3.12) if possible

## Submitting Changes

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/examples if applicable

## Types of Contributions

- **Bug fixes**: Fix issues in existing code
- **New features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Performance**: Optimize existing code
- **Tests**: Add or improve test coverage

## Questions?

Feel free to open an issue for questions or join our community discussions.
