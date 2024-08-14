# ./ghautocommit.sh

#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Ensure script is run from the git repository root
cd "$(git rev-parse --show-toplevel)" || exit 1

# Prompt for commit messages
read -p "Enter commit message for develop: " DEVELOP_COMMIT_MESSAGE
read -p "Enter commit message for feature/api-update: " API_COMMIT_MESSAGE
read -p "Enter commit message for feature/dashboard-update: " DASHBOARD_COMMIT_MESSAGE

# Function to update branch with specific directories
update_branch_with_specific_dirs() {
    local branch_name=$1
    local commit_message=$2
    shift 2
    local directories=("$@")

    echo "Updating $branch_name..."
    git checkout "$branch_name"

    # Create a temporary branch
    temp_branch="${branch_name}_temp"
    git checkout -b "$temp_branch"

    # Remove everything except .git directory
    find . -mindepth 1 -maxdepth 1 -not -name '.git' -exec rm -rf {} +

    # Checkout specific directories from develop
    for dir in "${directories[@]}"; do
        git checkout develop -- "$dir" || echo "Warning: Could not checkout $dir from develop"
    done

    # Show changes
    echo "Changes for $branch_name:"
    git status

    # Prompt for confirmation
    read -p "Do you want to proceed with these changes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        git add .
        git commit -m "$commit_message" || echo "No changes to commit for $branch_name"
        git checkout "$branch_name"
        git merge --ff-only "$temp_branch"
        git branch -D "$temp_branch"
        git push origin "$branch_name"
    else
        echo "Operation cancelled for $branch_name."
        git checkout "$branch_name"
        git branch -D "$temp_branch"
    fi
}

# Develop branch
git checkout develop
git add .
git commit -m "$DEVELOP_COMMIT_MESSAGE" || echo "No changes to commit for develop"
git push origin develop

# Feature/api-update branch
update_branch_with_specific_dirs "feature/api-update" "$API_COMMIT_MESSAGE" \
    "api" "config/api.json" "config/query.json" "utils/config_helper.py" "utils/data_helper.py" "config/paths.py"

# Feature/dashboard-update branch
update_branch_with_specific_dirs "feature/dashboard-update" "$DASHBOARD_COMMIT_MESSAGE" \
    "dashboard" "config/dashboard.json" "config/query.json" "utils/config_helper.py" "utils/data_helper.py" "config/paths.py"

# Return to develop branch
git checkout develop

echo "Script completed. Please verify the changes in each branch."