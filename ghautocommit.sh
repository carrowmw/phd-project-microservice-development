# ./ghautocommit.sh
#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Ensure script is run from the git repository root
cd "$(git rev-parse --show-toplevel)" || exit 1

# Function to prompt for commit message and commit changes
commit_changes() {
    local branch=$1
    local message
    read -p "Enter commit message for $branch: " message
    git add .
    git commit -m "$message" || echo "No changes to commit for $branch"
    git push origin "$branch"
}

# Function to update a feature branch with specific files
update_feature_branch() {
    local branch=$1
    shift
    local files=("$@")

    git checkout "$branch"
    git fetch origin
    git reset --hard origin/develop

    # Remove all files except .git
    find . -mindepth 1 -maxdepth 1 -not -name '.git' -exec rm -rf {} +

    # Checkout specific files from develop
    for file in "${files[@]}"; do
        git checkout develop -- "$file" || echo "Warning: Could not checkout $file from develop"
    done

    echo "Changes for $branch:"
    git status

    read -p "Do you want to proceed with these changes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        git commit -m "Update $branch with latest changes"
        git push --force origin "$branch"
    else
        echo "Operation cancelled for $branch"
        git reset --hard
    fi
}

# 1. Commit all files to develop branch
git checkout develop
commit_changes "develop"

# 2. Update feature/api-update branch
api_files=(
    "phd_package/api"
    "phd_package/config/api.json"
    "phd_package/config/query.json"
    "phd_package/utils/config_helper.py"
    "phd_package/utils/data_helper.py"
    "phd_package/config/paths.py"
)
update_feature_branch "feature/api-update" "${api_files[@]}"

# 3. Update feature/dashboard-update branch
dashboard_files=(
    "phd_package/dashboard"
    "phd_package/config/dashboard.json"
    "phd_package/config/query.json"
    "phd_package/utils/config_helper.py"
    "phd_package/utils/data_helper.py"
    "phd_package/config/paths.py"
)
update_feature_branch "feature/dashboard-update" "${dashboard_files[@]}"

# Return to develop branch
git checkout develop

# Run the rebuild the virtual environment script
./rebuild_venv.sh


echo "Script completed. Please verify the changes in each branch."