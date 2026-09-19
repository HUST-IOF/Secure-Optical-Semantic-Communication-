"""Verify published sample hashes and reject accidentally bundled large/private files."""
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
issues = []
manifest = json.loads((root/'data/measured/manifest.json').read_text(encoding='utf-8'))
for item in manifest['files']:
    p = root/'data/measured'/item['name']
    if hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
        issues.append('Sample checksum mismatch: '+item['name'])
excluded = {'.git', '__pycache__', '.pytest_cache', '.venv', 'runs', 'build', 'dist'}
files = [p for p in root.rglob('*') if p.is_file() and not any(x in excluded or x.endswith('.egg-info') for x in p.relative_to(root).parts)]
for p in files:
    rel = p.relative_to(root).as_posix()
    if p.stat().st_size > 5_000_000:
        issues.append('Unexpected file above 5 MB: '+rel)
    if p.suffix.lower() in {'.pem','.key','.safetensors','.ckpt','.pt','.pth','.prompt','.engine','.onnx'} or p.name.startswith('.env'):
        issues.append('Private/generated/model file: '+rel)
    if p.suffix in {'.py','.md','.txt','.json','.yml','.yaml','.toml','.csv'}:
        content=p.read_text(encoding='utf-8-sig')
        patterns=[r'/root/' + r'autodl-tmp', r'connect\.' + r'[\w.]*seetacloud\.com',
                  r'-----BEGIN ' + r'(?:OPENSSH |RSA |EC )?PRIVATE KEY-----',
                  r'gh[pousr]_' + r'[A-Za-z0-9]{30,}', r'github_pat_' + r'[A-Za-z0-9_]{40,}']
        if any(re.search(pattern,content) for pattern in patterns):
            issues.append('Private path or secret pattern: '+rel)
if issues:
    raise SystemExit('\n'.join(issues))
print(f'Release check passed: {len(files)} files, {sum(p.stat().st_size for p in files)/1e6:.2f} MB uncompressed.')
