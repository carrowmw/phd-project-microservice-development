

git checkout feature/api-update

# clean up the git cache
git rm -r --cached .

# add the files to feature-api-update branch
git checkout develop
git add api/ config/api.json config/query.json utils/config_helper.py utils/data_helper.py config/paths.py
git commit -m $COMMIT_MESSAGE
git push origin feature/api-update

# add the files to feature/dashboard-update branch
git add dashboard/ config/dashboard.json config/query.json utils/config_helper.py utils/data_helper.py config/paths.py
