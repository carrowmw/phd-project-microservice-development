# ./ghautocommit.sh

# Prompt for commit messages
read -p "Enter commit message for develop: " DEVELOP_COMMIT_MESSAGE
read -p "Enter commit message for feature/api-update: " API_COMMIT_MESSAGE
read -p "Enter commit message for feature/dashboard-update: " DASHBOARD_COMMIT_MESSAGE

# Function to update .gitignore and remove ignored directories
update_gitignore_and_remove_ignored() {
    # Stash any uncommitted changes
    git stash

    # Remove everything from the index
    git rm -r --cached .

    # Add all files back
    git add .

    # Show what will be removed
    echo "The following files/directories will be removed from the repository:"
    git ls-files --ignored --exclude-standard

    # Prompt for confirmation
    read -p "Do you want to proceed? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        # Remove ignored files from the repository
        git ls-files --ignored --exclude-standard -z | xargs -0 git rm --cached

        # Commit the changes
        git commit -m "Update .gitignore and remove ignored directories"
    else
        echo "Operation cancelled."
        # Restore the index
        git reset
    fi

    # Restore stashed changes
    git stash pop
}

# Develop branch
git checkout develop
update_gitignore_and_cache
git add .
git commit -m "$DEVELOP_COMMIT_MESSAGE"
git push origin develop

# Feature/api-update branch
git checkout feature/api-update
git merge develop --no-commit --no-ff
update_gitignore_and_cache
git add api/ config/api.json config/query.json utils/config_helper.py utils/data_helper.py config/paths.py
git commit -m "$API_COMMIT_MESSAGE"
git push origin feature/api-update

# Feature/dashboard-update branch
git checkout feature/dashboard-update
git merge develop --no-commit --no-ff
update_gitignore_and_cache
git add dashboard/ config/dashboard.json config/query.json utils/config_helper.py utils/data_helper.py config/paths.py
git commit -m "$DASHBOARD_COMMIT_MESSAGE"
git push origin feature/dashboard-update

# Return to develop branch
git checkout develop