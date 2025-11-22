# Git Workflow Guide: Syncing Upstream While Preserving Your Work

## Current Setup ✅

You have:
- **origin**: Your fork (`smohdarif/GenAIEngineering-Cohort3`)
- **upstream**: Original repo (`outskill-git/GenAIEngineering-Cohort3`)

## Recommended Workflow

### Option 1: Work on a Separate Branch (Recommended) 🌟

This is the safest approach:

```bash
# 1. Create and switch to your personal work branch
git checkout -b my-work

# 2. Do your work on this branch
# ... make your changes ...

# 3. Commit your work
git add .
git commit -m "My personal work"

# 4. When you need to sync upstream:
git checkout main
./sync_upstream.sh  # or manually merge upstream/main

# 5. Merge your work branch into main (if needed)
git merge my-work

# 6. Push everything
git push origin main
git push origin my-work
```

### Option 2: Use the Sync Script

The `sync_upstream.sh` script automates the sync process:

```bash
# Make it executable (first time only)
chmod +x sync_upstream.sh

# Run the sync
./sync_upstream.sh
```

**What the script does:**
1. Fetches latest from upstream
2. Stashes your uncommitted changes
3. Merges upstream/main into your branch
4. Restores your stashed changes
5. Optionally pushes to your fork

### Option 3: Manual Sync

```bash
# 1. Fetch latest from upstream
git fetch upstream

# 2. Check what's new
git log HEAD..upstream/main --oneline

# 3. Merge upstream changes
git merge upstream/main

# 4. If conflicts occur, resolve them:
#    - Edit conflicted files
#    - git add <resolved-files>
#    - git commit

# 5. Push to your fork
git push origin main
```

## Protecting Your Work

### Using .gitattributes

If you have specific files you want to always keep your version:

1. Edit `.gitattributes`:
   ```
   my_personal_file.ipynb merge=ours
   my_folder/** merge=ours
   ```

2. Commit the `.gitattributes` file:
   ```bash
   git add .gitattributes
   git commit -m "Add merge strategy for personal files"
   ```

⚠️ **Warning**: `merge=ours` will always keep your version, even if upstream has important updates. Use carefully!

### Using a Separate Branch

The safest way is to keep your work on a separate branch:
- `main`: Synced with upstream
- `my-work`: Your personal changes

## Best Practices

1. **Commit Often**: Commit your work frequently so it's saved
   ```bash
   git add .
   git commit -m "WIP: My progress"
   ```

2. **Create Branches**: Use branches for different features/experiments
   ```bash
   git checkout -b experiment-1
   git checkout -b homework-week5
   ```

3. **Sync Regularly**: Don't let too many changes accumulate
   ```bash
   # Sync weekly or before starting new work
   ./sync_upstream.sh
   ```

4. **Backup Important Work**: Push to your fork regularly
   ```bash
   git push origin my-work
   ```

## Handling Conflicts

If you get merge conflicts:

1. **See what's conflicted:**
   ```bash
   git status
   ```

2. **Open conflicted files** and look for conflict markers:
   ```
   <<<<<<< HEAD
   Your version
   =======
   Upstream version
   >>>>>>> upstream/main
   ```

3. **Resolve conflicts** by editing the file:
   - Keep your version
   - Keep upstream version
   - Combine both
   - Or create something new

4. **Mark as resolved:**
   ```bash
   git add <resolved-file>
   git commit
   ```

## Quick Reference

```bash
# Check current remotes
git remote -v

# Fetch from upstream (doesn't merge)
git fetch upstream

# See what's new in upstream
git log HEAD..upstream/main --oneline

# Merge upstream into current branch
git merge upstream/main

# Push to your fork
git push origin main

# Create a new branch for your work
git checkout -b my-branch-name

# Switch between branches
git checkout main
git checkout my-branch-name

# See all branches
git branch -a
```

## Troubleshooting

### "I lost my changes after syncing!"
- Check git reflog: `git reflog`
- Look for your commits: `git log --all --oneline`
- Recover: `git checkout <commit-hash>`

### "Too many conflicts!"
- Consider working on a separate branch
- Use `.gitattributes` to protect specific files
- Sync more frequently to avoid large conflicts

### "I want to start fresh but keep my work"
```bash
# Save your work
git checkout -b backup-my-work
git add .
git commit -m "Backup before reset"

# Reset main to upstream
git checkout main
git reset --hard upstream/main

# Your work is safe on backup-my-work branch
```

