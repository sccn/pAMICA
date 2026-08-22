<!-- PRs target the `dev` branch (the repository default); `main` is release-only. -->

- [ ] Targets `dev`, not `main`
- [ ] Behavior changes are validated against real sample data (see `AGENTS.md` / `.rules/testing.md`); no mocks or synthetic data as the basis for correctness
- [ ] No emojis in commits or PR text
- [ ] `uv run ruff check . && uv run ruff format --check .`, `uv run ty check .`, and `uv run pytest` all pass locally before requesting review
