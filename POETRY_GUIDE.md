# Managing Python Projects with Pyenv and Poetry

## Overview
This guide provides clear instructions for both starting a new project with Poetry and transitioning an existing project to use Poetry, ensuring you can manage dependencies and environments effectively.

This guide provides instructions for two common scenarios:

1. Creating a new project using Poetry.
2. Adding Poetry to an existing project where `pyenv` virtualenv has been used to manage dependencies.

## Prerequisites

- Install [Pyenv](https://github.com/pyenv/pyenv#installation)
- Install [Poetry](https://python-poetry.org/docs/#installation)

## Path 1: Creating a New Project Using Poetry

### Step 1: Set the Global Python Version

Set the global Python version using `pyenv`:

```bash
pyenv install 3.10.13
pyenv global 3.10.13
```

### Step 2: Create a New Poetry Project

Create a new project using Poetry:

```sh
poetry new my-new-project
cd my-new-project
```

### Step 3: Configure the Python Version for the Project

Specify the Python version in your `pyproject.toml`

```toml
[tool.poetry.dependencies]
python = "^3.10"
```

### Step 4: Add Dependencies

Add any dependencies your project needs:

```sh
poetry add requests
```

For development dependencies, use:

```sh
poetry add --dev pytest
```

### Step 5: Install Dependencies

Install the dependencies specified in your `pyproject.toml`:

```sh
poetry install
```

### Step 6: Run Your Project

Run your project using Poetry:

```sh
poetry run python your_script.py
```


## Path 2: Adding Poetry to an Existing Project Managed by `pyenv virtualenv`

### Step 1: Navigate to Your Project Directory

Go to your existing project directory:

```sh
cd path/to/your/existing-project
```

### Step 2: Set the Local Python Version for the Project

Set the local Python version for your project using `pyenv`:

```sh
pyenv install 3.9.7
pyenv local 3.9.7
```

### Step 3: Initialize Poetry in the Project

Initialize Poetry in your project directory:

```sh
poetry init
```

Follow the prompts to set up your `pyproject.toml`. Ensure the Python version is specified correctly:

```sh
[tool.poetry.dependencies]
python = "^3.9"
```

### Step 4: Add Existing Dependencies

Add the dependencies listed in your current requirements.txt file to your `pyproject.toml`:

```sh
poetry add requests
# Repeat for all dependencies
```

For development dependencies:

```sh
poetry add --dev pytest
# Repeat for all dev dependencies
```

### Step 5: Install Dependencies

Install all dependencies using Poetry:

```sh
poetry install
```

### Step 6: Verify the Virtual Environment

Ensure that Poetry is using the correct virtual environment:

```sh
poetry env info
```

### Step 7: Remove Old Virtual Environment (Optional)

If you no longer need the old pyenv virtual environment, deactivate and remove it:

```sh
pyenv deactivate
pyenv uninstall <your-old-virtualenv>
```

### Step 8: Run Your Project

Run your project using Poetry:

```sh
poetry run python your_script.py
```

## Summary

By following these steps, you can effectively manage your Python project’s dependencies and environments using Pyenv and Poetry, whether you’re starting a new project or transitioning an existing one. This setup ensures a clean, isolated environment for your projects, making dependency management and packaging much easier.