# Function to rebuild the virtual environment using Poetry
rebuild_venv() {
    echo "Rebuilding virtual environment..."
    if [ -d ".venv" ]; then
        rm -rf .venv
    fi
    python3.10 -m venv .venv
    source .venv/bin/activate
    poetry env use $(which python)
    poetry install
    echo "Virtual environment rebuilt."
}

rebuild_venv