"""Fetch the exact author detector used by the O³ adapter; no weights or training."""
import hashlib
import urllib.request
from pathlib import Path

REVISION = '839afea7fc09001433ef30bd6670a6d6804a9045'
URL = f'https://raw.githubusercontent.com/GCYZSL/O3-LLM-UNLEARNING/{REVISION}/ScienceQA_expriment/src/ood_model_selector.py'
SHA256 = '08ed1cf87eb3f9d4d8737a2c6d5c6f5da27c94e4aae98d16ae248a7409ce65fd'


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / '.cache/o3-source/ood_model_selector.py'
    data = target.read_bytes() if target.exists() else urllib.request.urlopen(URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError(f'O³ detector checksum mismatch: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f'O³ author detector verified: {target}')


if __name__ == '__main__':
    main()
