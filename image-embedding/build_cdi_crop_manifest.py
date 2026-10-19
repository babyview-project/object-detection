"""
Build a crop manifest from the 2025.2 `object_detections/cdi` pipeline output, for use with
create_image_embeddings.py --image_manifest_csv.

That pipeline never persisted clean per-object crops -- the only pixels it left on disk are
`<video_id>/<frame>_annotated.jpg` (full frames with EVERY box for that frame already drawn on
them) plus `bounding_box_predictions.csv` with the box coordinates in that same image's pixel
space. By default (no --frames_root) "crop" means: crop the box out of the annotated frame. The
drawn rectangle sits ON the box boundary, so a sliver of it (a few px of colored outline) can end
up at the crop edge -- a known limitation of that fallback.

Pass --frames_root to crop out of the clean 1fps-extracted frames instead (e.g.
/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps/): `frame_number` in the CSV is the same
index used by <frames_root>/<video_id>/<frame_number:05d>.jpg, so this needs no annotated.jpg to
exist at all -- it also picks up frames that were detected on but never got an annotated frame
saved.

Usage:
    python build_cdi_crop_manifest.py \
        --cdi_root /ccn2a/dataset/babyview/2025.2/outputs/object_detections/cdi \
        --category_list ../data/included_categories_valid129.txt \
        --min_confidence 0.5 \
        --frames_root /ccn2a/dataset/babyview/2025.2/extracted_frames_1fps \
        --out_csv /ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/babyview_full/cdi_crop_manifest.csv
"""
import argparse
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

ANNOTATED_RE = re.compile(r'^(\d+)_annotated\.jpg$')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cdi_root', type=str,
                    default='/ccn2a/dataset/babyview/2025.2/outputs/object_detections/cdi')
    p.add_argument('--category_list', type=str, required=True,
                    help='One CDI category (class_name) per line, e.g. data/included_categories_valid129.txt')
    p.add_argument('--min_confidence', type=float, default=0.5)
    p.add_argument('--frames_root', type=str, default=None,
                    help='If given, crop out of <frames_root>/<video_id>/<frame_number:05d>.jpg '
                         '(clean 1fps-extracted frames) instead of the annotated fallback frame.')
    p.add_argument('--out_csv', type=str, required=True)
    p.add_argument('--num_workers', type=int, default=16,
                    help='Process-pool workers for the CSV scan (CPU/IO-bound, no GPU needed).')
    p.add_argument('--limit_videos', type=int, default=None,
                    help='For a quick dry run: only scan the first N video dirs.')
    return p.parse_args()


def scan_one_video(video_dir, allowed_categories, min_confidence, frames_root):
    """Returns a list of manifest row dicts for one video directory."""
    csv_path = os.path.join(video_dir, 'bounding_box_predictions.csv')
    if not os.path.exists(csv_path):
        return []

    video_id = os.path.basename(video_dir)

    if frames_root:
        # Clean 1fps-extracted frames: frame_number is the same index used by
        # <frames_root>/<video_id>/<frame_number:05d>.jpg. No annotated.jpg needed.
        frame_path = lambda frame: os.path.join(frames_root, video_id, f'{frame:05d}.jpg')
    else:
        annotated_frames = {}
        for f in glob.glob(os.path.join(video_dir, '*_annotated.jpg')):
            m = ANNOTATED_RE.match(os.path.basename(f))
            if m:
                annotated_frames[int(m.group(1))] = f
        if not annotated_frames:
            return []
        frame_path = lambda frame: annotated_frames.get(frame)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f'[WARN] failed to read {csv_path}: {e}')
        return []

    df = df[df['xmin'].notna()]
    if not frames_root:
        df = df[df['frame_number'].isin(annotated_frames.keys())]
    if df.empty:
        return []
    df = df[df['confidence'] >= min_confidence]
    df['class_name_norm'] = df['class_name'].astype(str).str.strip().str.lower()
    df = df[df['class_name_norm'].isin(allowed_categories)]
    if df.empty:
        return []

    rows = []
    # Disambiguate multiple boxes of the same class in the same frame with a running index.
    dup_counter = {}
    for _, r in df.iterrows():
        frame = int(r['frame_number'])
        path = frame_path(frame)
        if frames_root and not os.path.exists(path):
            continue
        cls = r['class_name_norm']
        key = (frame, cls)
        idx = dup_counter.get(key, 0)
        dup_counter[key] = idx + 1
        image_id = f'{video_id}_{frame:05d}_{cls}_{idx}'
        rows.append({
            'image_id': image_id,
            'path': path,
            'xmin': r['xmin'], 'ymin': r['ymin'], 'xmax': r['xmax'], 'ymax': r['ymax'],
            'class_name': cls,
            'confidence': r['confidence'],
            'video_id': video_id,
            'frame_number': frame,
        })
    return rows


def _worker(args):
    video_dir, allowed_categories, min_confidence, frames_root = args
    return scan_one_video(video_dir, allowed_categories, min_confidence, frames_root)


if __name__ == '__main__':
    args = get_args()
    with open(args.category_list) as fh:
        allowed_categories = {line.strip().lower() for line in fh if line.strip()}
    print(f'Loaded {len(allowed_categories)} allowed categories from {args.category_list}')

    video_dirs = sorted(
        d for d in glob.glob(os.path.join(args.cdi_root, '*')) if os.path.isdir(d)
    )
    if args.limit_videos:
        video_dirs = video_dirs[:args.limit_videos]
    print(f'Scanning {len(video_dirs)} video dirs under {args.cdi_root}')

    all_rows = []
    tasks = [(d, allowed_categories, args.min_confidence, args.frames_root) for d in video_dirs]
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        done = 0
        for fut in as_completed(futures):
            all_rows.extend(fut.result())
            done += 1
            if done % 200 == 0:
                print(f'  scanned {done}/{len(video_dirs)} video dirs, {len(all_rows)} crops so far')

    manifest = pd.DataFrame(all_rows)
    frame_source = f'frame exists under {args.frames_root}' if args.frames_root else 'frame has a saved annotated.jpg'
    print(f'Total crops after filtering (category in list, confidence >= {args.min_confidence}, '
          f'{frame_source}): {len(manifest)}')
    if len(manifest):
        print('Crops per class_name (top 15):')
        print(manifest['class_name'].value_counts().head(15))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    manifest.to_csv(args.out_csv, index=False)
    print(f'Wrote manifest to {args.out_csv}')
