# Contributing to AppForge

Thank you for your interest in contributing to AppForge! This document provides guidelines for contributing.

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## Development Setup

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Publishing to PyPI

To publish the package to PyPI:

```bash
pip install build twine
python -m build
twine upload dist/*
```

> **Note:** You need to have PyPI credentials configured. See [PyPI documentation](https://packaging.python.org/en/latest/tutorials/packaging-projects/) for setup instructions.

## Code Style

- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for public functions

## Reporting Issues

Please use the GitHub issue tracker to report bugs or suggest enhancements.
