"""Verify the source snapshot without importing any research modules."""
import ast
import hashlib
import json
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    python_count = 0
    for relative, entry in manifest['files'].items():
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f'Unsafe manifest path: {relative}')
        data = path.read_bytes()
        normalized = data.replace(b'\r\n', b'\n')
        if sha256(normalized) != entry['sha256_lf']:
            raise ValueError(f'File differs from the snapshot: {relative}')
        if path.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), filename=relative)
            python_count += 1
    print(f'Verified {len(manifest["files"])} files and parsed {python_count} Python scripts.')


if __name__ == '__main__':
    main()
