# Function to rebuild the virtual environment using Poetry
rebuild_venv() {
    echo "Rebuilding virtual environment..."
    if [ -d ".venv" ]; then
        rm -rf .venv
    fi
    poetry env use python3.10
    poetry install
    echo "Virtual environment rebuilt."
}

rebuild_venv