# Upload this project to GitHub

Upload only this repository directory, not its surrounding workspace. It excludes SSH tooling, credentials, local environments, large checkpoints and complete experimental archives.

From this directory, verify the selected release:

```bash
python tools/verify_release.py
python -m pytest -q
```

Create an empty repository on GitHub, then run:

```bash
git init
git add .
git status --short
git commit -m "Release physical semantic codec and measured examples"
git branch -M main
```

Use the remote URL supplied by GitHub:

```bash
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Alternatively, upload the extracted project files using GitHub's web interface.
Upload the file tree rather than a single ZIP if you want GitHub to display the
README and run tests. Dependencies and pretrained weights remain external.

The included GitHub Actions workflow runs CPU checks. CUDA tests run when a
CUDA-capable runner is provided; GitHub-hosted CPU runners do not prove GPU
execution. Local/remote validation results are documented separately.

