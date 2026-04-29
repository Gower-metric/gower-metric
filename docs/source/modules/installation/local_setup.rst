==================
Local Environment
==================

To set up a local development environment for our project, we suggest to use following tools:

- **Python 3.10+**: Ensure you have Python installed. We recommend using a virtual environment to manage dependencies.
- **Git**: For version control and to clone the repository.
- **Uv**: A tool for managing virtual environments. You can install it via pip:
  
  .. code-block:: bash
  
      pip install uv

- **Tox**: A tool for automating testing in multiple environments. We used it to run tests with all supported Python versions.
- **Pre-commit**: A framework for managing and maintaining multi-language pre-commit hooks. It helps ensure code quality before commits.

In addition, we use ruff and mypy for linting and type checking, respectively.