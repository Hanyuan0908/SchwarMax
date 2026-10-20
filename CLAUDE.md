# SchwarMAX — instructions for coding agents

@docs/DEVELOPMENT.md

- Work on branch `developing-code`. Never commit to `main`.
- Your workstream has its own folder (`SchwarMAX-Nbody/` is the projected scheme). Keep your files
  there; the package at the repository root is shared, and changes to it are add-only.
- Before any commit that touches the package: `cd SchwarMAX-Nbody && python test_regression_projected.py`.
- Stage explicit paths only. Never `git add -A`, `git checkout`, `git stash` or `git reset` here:
  another agent works in this same folder.
