# Apply this telemetry patch to the current repository

This bundle is an overlay, not a complete repository.

## 1. Start from the latest integration branch

```bash
git checkout dev
git fetch origin
git pull --ff-only origin dev
git status
git log --oneline -5
git checkout -b feat/telemetry-state
```

Do not continue if `git status` shows unrelated uncommitted changes.

## 2. Extract the patch outside the repository

Example:

```bash
unzip ndt_telemetry_completion_patch.zip -d /tmp/ndt-telemetry-patch
```

## 3. Preview the overlay

From any directory, replace `<REPO>` with the repository path:

```bash
rsync -avnc /tmp/ndt-telemetry-patch/ndt_telemetry_completion_patch/ <REPO>/
```

`-n` is dry-run. Nothing is written yet.

## 4. Apply without deleting unrelated repository files

```bash
rsync -avc /tmp/ndt-telemetry-patch/ndt_telemetry_completion_patch/ <REPO>/
cd <REPO>
git status
git diff --stat
git diff
```

There is intentionally no `--delete`; files that are not part of the telemetry patch are preserved.

## 5. Validate locally

```bash
source ~/ndt-venv/bin/activate
pip install -r requirements-orchestrator.txt
pip install -r requirements-dev.txt
make ci
```

For the controller environment:

```bash
source ~/ryu-venv/bin/activate
pip install -r requirements-controller-extra.txt
python -m py_compile src/controller/telemetry_agent.py src/controller/reactive_controller.py
```

The real `make smoke` test must run on the prepared SDN host with Mininet/Open vSwitch and both virtualenvs available.

## 6. Commit by concern

```bash
git add configs src/schemas
git commit -m "feat(schema): complete telemetry contracts and config"

git add src/controller
git commit -m "fix(telemetry): complete controller and switch telemetry"

git add src/telemetry src/orchestrator
git commit -m "feat(state): collect telemetry and validate coherent snapshots"

git add tests scripts .github Makefile requirements-dev.txt pyproject.toml .gitignore README.md
git commit -m "test(ci): add telemetry tests and SDN integration workflows"
```

Before push, sync latest `dev` on the feature branch:

```bash
git fetch origin
git merge origin/dev
make ci
git push -u origin feat/telemetry-state
```

Open a PR from `feat/telemetry-state` to `dev` and ask the other developer to review it.
