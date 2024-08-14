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

    # Instead of removing everything, let's only keep the specified directories
    for dir in "${directories[@]}"; do
        if [ -e "$dir" ]; then
            echo "Keeping $dir"
            mkdir -p "temp_dir/$(dirname "$dir")"
            cp -R "$dir" "temp_dir/$(dirname "$dir")/"
        else
            echo "Warning: $dir does not exist in $branch_name"
        fi
    done

    # Remove everything except .git and temp_dir
    find . -mindepth 1 -maxdepth 1 -not -name '.git' -not -name 'temp_dir' -exec rm -rf {} +

    # Move contents of temp_dir back
    mv temp_dir/* .
    rm -rf temp_dir

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
    "phd_package/api" "phd_package/config/api.json" "phd_package/config/query.json" "phd_package/utils/config_helper.py" "phd_package/utils/data_helper.py" "phd_package/config/paths.py"

# Feature/dashboard-update branch
update_branch_with_specific_dirs "feature/dashboard-update" "$DASHBOARD_COMMIT_MESSAGE" \
    "phd_package/dashboard" "phd_package/config/dashboard.json" "phd_package/config/query.json" "phd_package/utils/config_helper.py" "phd_package/utils/data_helper.py" "phd_package/config/paths.py"

# Return to develop branch
git checkout develop

echo "Script completed. Please verify the changes in each branch."