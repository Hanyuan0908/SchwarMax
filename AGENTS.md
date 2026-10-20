# SchwarMAX — instructions for coding agents

Read `docs/DEVELOPMENT.md` before your first change. In short:

- Work on branch `developing-code`. Never commit to `main`.
- Your workstream has its own folder (`SchwarMAX-Nbody/` is the projected scheme). Keep your files
  there; the package at the repository root is shared, and changes to it are add-only: new
  functions and modules, or new keyword arguments whose default keeps the old behaviour.
- Before any commit that touches the package: `cd SchwarMAX-Nbody && python test_regression_projected.py`.
- Stage explicit paths only. Never `git add -A`, `git checkout`, `git stash` or `git reset` here:
  another agent works in this same folder.
- Log your work in `docs/devlog_<your-workstream>.md`.
