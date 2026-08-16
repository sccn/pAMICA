# Git & Version Control Standards

## Commit Messages
- **Format:** `<type>: <description>`
- **Length:** <50 characters
- **No emojis** in commits or PR titles
- **No AI attribution** (no "Co-Authored-By: Claude" or similar)
- **Types:**
  - `feat:` New feature
  - `fix:` Bug fix
  - `docs:` Documentation only
  - `refactor:` Code restructuring
  - `test:` Adding tests (real tests only)
  - `chore:` Maintenance tasks

## Branch Strategy
- **`main` is the stable branch and is protected.** Nothing lands on it except a
  release merge from `dev`. Everything on `main` is released.
- **`dev` is the integration branch.** Feature and fix branches are cut from
  `dev` and merged back into `dev`.
- **Feature branches:** `feature/issue-N-short-description`
- **Bugfix branches:** `fix/issue-N-description`
- **Use `gh issue develop`** to create branches from issues
- **No spaces** in branch names, use hyphens
- **Delete after merge**

## Versioning (automated)
Versions are managed by CI; do not hand-edit `pyproject.toml` to bump.

| Event | Version effect | Workflow |
|---|---|---|
| PR merged into `dev` | `X.Y.Z.devN` -> `X.Y.Z.dev(N+1)` | `auto-bump-dev.yml` |
| `dev` merged into `main` | `.devN` stripped -> `X.Y.Z`, tagged `vX.Y.Z`, release published | `auto-tag.yml` |
| Release published to PyPI | `main` merged back to `dev`, bumped to `X.Y.(Z+1).dev0` | `sync-dev.yml` |

So a release is: merge `dev` into `main`. Everything after that is automatic —
the tag, the GitHub release, the PyPI upload (`publish.yml`), the native binaries
(`release-binaries.yml`), and returning `dev` to a fresh `.dev0`.

`.devN` is PEP 440, so `pip install pamica==0.3.3.dev4` resolves and a dev build
is always identifiable at runtime via `pamica.__version__`.

Notes:
- Add `[skip-bump]` to a commit message to suppress the dev bump for that push.
- The bump workflows commit as `pamica-bot` and skip their own commits; do not
  reuse that identity by hand.
- The automation needs an `AUTO_TAG_PAT` repository secret. The default
  `GITHUB_TOKEN` cannot drive it: pushes and releases it makes do not trigger
  other workflows, so the release chain would go quiet instead of publishing.

### AUTO_TAG_PAT permissions
Fine-grained PAT scoped to this repository:

| Permission | Why |
|---|---|
| Contents: read and write | push bump commits and tags, create releases |
| Workflows: read and write | `sync-dev` merges main into dev; when a release touched `.github/workflows/**` that merge carries workflow files, and GitHub rejects a PAT push that does so without this |

Classic PAT equivalent: `repo` + `workflow`.

The push authenticates as the PAT's owner whatever git identity the workflow
sets, so a machine account keeps the bot attribution honest. Once `main` is
protected, that account also has to be able to push through the protection.
Fine-grained PATs expire; an expired one fails at the push rather than at the
preflight, so the error is less obvious than a missing secret.

## Commit Practice
- **Atomic commits** - One logical change per commit
- **Test before commit** - Ensure code works
- **No broken commits** - Each commit should work independently

## Pull Request Process
1. Create issue first (for significant changes)
2. Use `gh issue develop` to create branch
3. Make atomic commits
4. Push branch
5. Create PR with `gh pr create`:
   - Clear title (<70 chars, no emojis)
   - Description with "Fixes #123"
   - Test results summary
6. Run `/review-pr` and address ALL findings
7. Merge with a regular merge commit to preserve history (squash only for epic sub-phase PRs)

## Merge Strategy
- **Regular merge commits** by default to preserve history; squash only when explicitly requested (e.g. epic sub-phase PRs)
- **Rebase** to update feature branches from base (`git rebase origin/main`)
- **Never force-push** to shared branches (main, develop)
- **Delete branch** after merge

## Git Commands
```bash
# Start feature from issue
gh issue develop 123

# Atomic commits
git add -p  # Stage selectively
git commit -m "feat: add user authentication"

# Update branch
git fetch origin
git rebase origin/main

# Push and create PR
git push -u origin feature/issue-123-auth
gh pr create
```

## .gitignore Essentials
```
__pycache__/     # Python
node_modules/    # JavaScript
.env             # Secrets
*.log            # Logs
.venv/           # Virtual environments
```

---
*Atomic commits, clear messages, clean history. No emojis, no AI attribution.*
