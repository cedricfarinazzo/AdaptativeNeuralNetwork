#!/bin/sh

#  ARGS:
# AUR_SSH_PRIVATE_KEY
# AUR_DEPLOY_USER
# AUR_DEPLOY_HOST
# AUR_DEPLOY_PATH

eval $(ssh-agent -s)
echo "$AUR_SSH_PRIVATE_KEY" | tr -d '\r' | ssh-add - > /dev/null
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keyscan "$AUR_DEPLOY_HOST" > ~/.ssh/known_hosts
chmod 644 ~/.ssh/known_hosts

git config --global user.email "$GITLAB_USER_EMAIL"
git config --global user.name "$GITLAB_USER_LOGIN"

git clone -q ssh://"$AUR_DEPLOY_USER@$AUR_DEPLOY_HOST"/"$AUR_DEPLOY_PATH" ~/ann-aur
git checkout aur
cp PKGBUILD ~/ann-aur/
cp .SRCINFO ~/ann-aur/
cd ~/ann-aur
git add .
git commit -m"ANN: $CI_COMMIT_TAG"
git push -q 
