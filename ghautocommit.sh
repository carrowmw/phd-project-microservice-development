# ./ghautocommit.sh

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

    git checkout $branch_name

    # Remove everything from the branch
    git rm -rf .

    # Checkout specific directories from develop
    for dir in "${directories[@]}"; do
        git checkout develop -- $dir
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
        git commit -m "$commit_message"
        git push origin $branch_name
    else
        echo "Operation cancelled for $branch_name."
        git reset --hard
    fi
}

# Develop branch
git checkout develop
git add .
git commit -m "$DEVELOP_COMMIT_MESSAGE"
git push origin develop

# Feature/api-update branch
update_branch_with_specific_dirs "feature/api-update" "$API_COMMIT_MESSAGE" \
    "api" "config/api.json" "config/query.json" "utils/config_helper.py" "utils/data_helper.py" "config/paths.py" "ghautocommit.sh"

# Feature/dashboard-update branch
update_branch_with_specific_dirs "feature/dashboard-update" "$DASHBOARD_COMMIT_MESSAGE" \
    "dashboard" "config/dashboard.json" "config/query.json" "utils/config_helper.py" "utils/data_helper.py" "config/paths.py" "ghautocommit.sh"

# Return to develop branch
git checkout develop