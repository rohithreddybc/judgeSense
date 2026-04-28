# Contributing to JudgeSense

Thank you for your interest in contributing to JudgeSense! This document explains how to report issues, suggest improvements, and submit code.

## Reporting Issues

- Search [existing issues](https://github.com/rohithreddybc/judgeSense/issues) before opening a new one.
- Include a clear title, steps to reproduce, expected vs. actual behavior, and your Python version.

## Suggesting Improvements

Open a GitHub issue with the label `enhancement`. Describe the motivation and the expected benefit.

## Submitting Pull Requests

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make your changes and add tests in `tests/` if applicable.
4. Run the test suite:
   ```bash
   pytest tests/ -v
   ```
5. Commit with a clear message describing the change.
6. Open a pull request against `main`. Describe what changed and why.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Keep functions small and focused.
- Do not commit API keys or `.env` files.

## Adding a New Judge to the Benchmark

If you want to evaluate a new judge model and contribute its JSS results:

1. Add an API wrapper in `src/models.py`.
2. Run the evaluation using `src/evaluate.py` against the 494 validated pairs.
3. Record results in `data/results/` following the existing JSON schema.
4. Open a pull request with the new results and the model checkpoint identifier.

## License

By submitting a pull request, you agree that your contributions will be licensed under the [MIT License](LICENSE).
