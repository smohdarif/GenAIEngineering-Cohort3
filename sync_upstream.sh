#!/bin/bash

# Script to sync upstream changes while preserving your work
# Usage: ./sync_upstream.sh

set -e  # Exit on error

echo "🔄 Starting upstream sync process..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${YELLOW}⚠️  Warning: You're on branch '$CURRENT_BRANCH', not 'main'${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Fetch latest from upstream
echo -e "${GREEN}📥 Fetching latest changes from upstream...${NC}"
git fetch upstream

# Check if there are any changes
UPSTREAM_CHANGES=$(git rev-list HEAD..upstream/main --count)
if [ "$UPSTREAM_CHANGES" -eq 0 ]; then
    echo -e "${GREEN}✅ Already up to date with upstream!${NC}"
    exit 0
fi

echo -e "${YELLOW}📊 Found $UPSTREAM_CHANGES new commits from upstream${NC}"

# Stash any uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}💾 Stashing uncommitted changes...${NC}"
    git stash push -m "Auto-stash before upstream sync $(date +%Y-%m-%d_%H:%M:%S)"
    STASHED=true
else
    STASHED=false
fi

# Merge upstream changes
echo -e "${GREEN}🔀 Merging upstream/main into current branch...${NC}"
if git merge upstream/main --no-edit; then
    echo -e "${GREEN}✅ Successfully merged upstream changes!${NC}"
else
    echo -e "${RED}❌ Merge conflict detected!${NC}"
    echo -e "${YELLOW}Please resolve conflicts manually and then run:${NC}"
    echo "  git add ."
    echo "  git commit"
    
    # Restore stashed changes if any
    if [ "$STASHED" = true ]; then
        echo -e "${YELLOW}💾 Restoring stashed changes...${NC}"
        git stash pop
    fi
    exit 1
fi

# Restore stashed changes if any
if [ "$STASHED" = true ]; then
    echo -e "${GREEN}💾 Restoring stashed changes...${NC}"
    if ! git stash pop; then
        echo -e "${YELLOW}⚠️  Some stashed changes had conflicts. Resolve them manually.${NC}"
    fi
fi

# Push to your fork
echo -e "${GREEN}📤 Pushing changes to your fork (origin)...${NC}"
read -p "Push to origin? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin main
    echo -e "${GREEN}✅ Sync complete!${NC}"
else
    echo -e "${YELLOW}⏭️  Skipped push. Run 'git push origin main' when ready.${NC}"
fi

