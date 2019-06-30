#!/bin/sh

#  ARGS:
# DOC_SSH_PRIVATE_KEY
# DOC_DEPLOY_USER
# DOC_DEPLOY_HOST
# DOC_DEPLOY_PATH

eval $(ssh-agent -s)
echo "$DOC_SSH_PRIVATE_KEY" | tr -d '\r' | ssh-add - > /dev/null
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keyscan "$DOC_DEPLOY_HOST" > ~/.ssh/known_hosts
chmod 644 ~/.ssh/known_hosts

git config --global user.email "$GITLAB_USER_EMAIL"
git config --global user.name "$GITLAB_USER_LOGIN"

git clone -q "$DOC_DEPLOY_USER@$DOC_DEPLOY_HOST":"$DOC_DEPLOY_PATH" ~/ann-doc
cp -r build/doc/html/* ~/ann-doc/
cd ~/ann-doc
git add .
git commit -m"ANN: doc: $CI_COMMIT_TAG"
git push -q 
