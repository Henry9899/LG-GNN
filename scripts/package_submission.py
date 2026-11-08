"""Build a source-only submission archive and a per-file integrity manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    excluded = {'__pycache__', '.pytest_cache', '.venv', 'data', 'runs', 'dist', '.git'}
    allowed = {'.py', '.md', '.toml', '.json', '.txt', '.cff'}
    paths = sorted(p for p in root.rglob('*') if p.is_file()
                   and not any(part in excluded or part.endswith('.egg-info') for part in p.relative_to(root).parts)
                   and (p.suffix in allowed or p.name == '.gitignore'))
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            archive.write(p, 'lg-gnn-code/' + p.relative_to(root).as_posix())
        archive.writestr('lg-gnn-code/SHA256SUMS.json', json.dumps(manifest, indent=2) + '\n')
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + '.sha256').write_text(f'{checksum}  {output.name}\n', encoding='utf-8')
    print(f'{len(paths)} source files archived; SHA-256 {checksum}')


if __name__ == '__main__':
    main()
