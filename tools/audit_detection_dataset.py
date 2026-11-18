"""Audit YOLO detection data and location/session split provenance before training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import math
from PIL import Image
import yaml


def audit(data_yaml, manifest):
    data_yaml = Path(data_yaml).resolve()
    config = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
    names = config['names']
    if isinstance(names, dict):
        assert sorted(names) == list(range(len(names))), 'Class IDs must be contiguous integers'
        names = [names[i] for i in range(len(names))]
    assert names and len(set(n.lower().strip() for n in names)) == len(names), 'Duplicate/empty class names'
    base = Path(config.get('path', data_yaml.parent))
    if not base.is_absolute(): base = data_yaml.parent/base
    with Path(manifest).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    indexed, groups = {}, {}
    official_only = 0
    for row in rows:
        relative = row['image'].replace('\\', '/')
        assert relative not in indexed, f'Duplicate manifest row: {relative}'
        assert row['split'] in ('train', 'val', 'test')
        if row.get('source') == 'rod' and row.get('provenance') == 'official_split':
            official_only += 1
            indexed[relative] = row
            continue  # Explicitly unknown route provenance; preserve official split, still check pixels below.
        for field in ('location_id', 'session_id'):
            assert row[field].strip(), f'Missing {field}'
            key = (field, row[field])
            assert groups.setdefault(key, row['split']) == row['split'], f'Split leakage: {key}'
        indexed[relative] = row
    counts, hashes, used = {}, {}, set()
    for split in ('train', 'val', 'test'):
        directory = base/config[split]
        assert directory.is_dir(), f'Expected image directory: {directory}'
        images = sorted(p for p in directory.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))
        assert images, f'Empty {split}'
        counts[split] = len(images)
        for image in images:
            relative = image.relative_to(base).as_posix()
            assert relative in indexed and indexed[relative]['split'] == split, f'Missing/wrong provenance: {relative}'
            used.add(relative)
            with Image.open(image) as decoded:
                decoded.load()
                rgb = decoded.convert('RGB')
                digest = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            assert digest not in hashes, f'Duplicate pixels: {relative} and {hashes.get(digest)}'
            hashes[digest] = relative
            parts = list(image.parts)
            assert 'images' in parts, f'YOLO images component missing: {image}'
            parts[len(parts)-1-parts[::-1].index('images')] = 'labels'
            label = Path(*parts).with_suffix('.txt')
            assert label.is_file(), f'Missing label: {relative}'
            lines = label.read_text(encoding='utf-8').splitlines()
            if not any(line.strip() for line in lines):
                assert indexed[relative].get('verified_empty') == '1', f'Unreviewed empty image: {relative}'
            for line in lines:
                if not line.strip(): continue
                values = list(map(float, line.split()))
                assert len(values) == 5 and all(math.isfinite(x) for x in values), f'Invalid detection row: {label}'
                cls, x, y, w, h = values
                assert cls.is_integer() and 0 <= cls < len(names), f'Invalid class: {label}'
                assert 0 < w <= 1 and 0 < h <= 1 and w/2-1e-6 <= x <= 1-w/2+1e-6 and h/2-1e-6 <= y <= 1-h/2+1e-6, f'Invalid box: {label}'
    assert used == set(indexed), 'Manifest contains images absent from dataset'
    return dict(images=counts, names=names, exact_pixel_duplicates=0, official_split_only_images=official_only,
                note='Near duplicates and provenance accuracy require human review; no automatic split repair.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--manifest', required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.data, args.manifest), ensure_ascii=False, indent=2))
