from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
import concurrent.futures
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from .common import CLASSES, ROOT, STL_LABELS, save_json

URL = "https://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
CHECKSUMS = {
    "train_X.bin": "918c2871b30a85fa023e0c44e0bee87f",
    "train_y.bin": "5a34089d4802c674881badbb80307741",
    "test_X.bin": "7f263ba9f9e0b06b93213547f721ac82",
    "test_y.bin": "36f9794fa4beb8a2c72628de14fa638e",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_stl(root=ROOT):
    dest = Path(root) / "data/raw/stl10_binary"
    dest.mkdir(parents=True, exist_ok=True)
    needed = {name for name, digest in CHECKSUMS.items() if not (dest / name).exists() or md5(dest / name) != digest}
    if not needed:
        print("STL-10 supervised files: checksums OK", flush=True)
        return
    print("Streaming official STL-10 archive (skip saving unlabeled images)", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "AnimalKNN/1.0 academic-project"})
    with urllib.request.urlopen(req, timeout=120) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if name not in needed or not member.isfile():
                continue
            print(f"Downloading {name} ({member.size / 2**20:.1f} MiB)", flush=True)
            # Extract only known filenames; never trust archive paths.
            temp = dest / (name + ".part")
            with archive.extractfile(member) as source, temp.open("wb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
            if md5(temp) != CHECKSUMS[name]:
                raise ValueError(f"Checksum mismatch: {name}")
            temp.replace(dest / name)
            needed.remove(name)
            if not needed:
                break
    if needed:
        raise RuntimeError(f"Archive missing: {needed}")
    save_json(Path(root) / "data/raw/source.json", {"url": URL, "md5": CHECKSUMS, "unlabeled_used": False})


def download_ranged(root=ROOT):
    """Download the supervised prefix in bounded HTTP ranges, then verify official files."""
    root = Path(root)
    dest = root / "data/raw/stl10_binary"
    dest.mkdir(parents=True, exist_ok=True)
    needed = {n for n,d in CHECKSUMS.items() if not (dest/n).exists() or md5(dest/n) != d}
    if not needed:
        return
    parts = root / "data/raw/archive_parts"
    parts.mkdir(parents=True, exist_ok=True)
    size = 12 * 1024 * 1024
    def fetch(i):
        path = parts / f"{i:03d}.part"
        if path.exists() and path.stat().st_size == size:
            return path
        start, end = i * size, (i + 1) * size - 1
        subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "180",
                        "--range", f"{start}-{end}", URL, "--output", str(path)], check=True)
        if path.stat().st_size != size:
            raise ValueError(f"Unexpected range length {i}")
        print(f"Stanford range {i+1}/32 complete", flush=True)
        return path
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, range(32)))
    prefix = parts / "supervised_prefix.tar.gz"
    with prefix.open("wb") as output:
        for i in range(32):
            with (parts / f"{i:03d}.part").open("rb") as handle:
                shutil.copyfileobj(handle, output, 1024*1024)
    with tarfile.open(prefix, mode="r|gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if name not in needed or not member.isfile():
                continue
            with archive.extractfile(member) as source, (dest / (name+".range")).open("wb") as output:
                shutil.copyfileobj(source, output, 1024*1024)
            temp = dest / (name+".range")
            if md5(temp) != CHECKSUMS[name]:
                raise ValueError(f"Official checksum mismatch {name}")
            temp.replace(dest / name)
            needed.remove(name)
            print(f"Stanford {name}: official MD5 verified", flush=True)
            if not needed:
                break
    if needed:
        raise ValueError(f"Supervised archive prefix insufficient: {needed}")
    save_json(root / "data/raw/source.json", {"url": URL, "method": "HTTP ranges, supervised prefix only", "md5": CHECKSUMS, "unlabeled_used": False})


def _raw(root, split):
    folder = Path(root) / "data/raw/stl10_binary"
    x = np.memmap(folder / f"{split}_X.bin", dtype=np.uint8, mode="r").reshape(-1, 3, 96, 96)
    x = x.transpose(0, 3, 2, 1)  # Match torchvision's column-major binary decoding.
    y = np.fromfile(folder / f"{split}_y.bin", dtype=np.uint8)
    return x, y


def prepare(root=ROOT, download=False, ranged=False):
    root = Path(root)
    if download:
        if ranged:
            download_ranged(root)
        else:
            download_stl(root)
    for name, digest in CHECKSUMS.items():
        p = root / "data/raw/stl10_binary" / name
        if not p.exists() or md5(p) != digest:
            raise ValueError(f"Missing/corrupt {name}. Run prepare --download.")
    rows, arrays = [], []
    for source_split in ("train", "test"):
        x, y = _raw(root, source_split)
        selected = np.flatnonzero(np.isin(y, STL_LABELS))
        for original in selected:
            pixels = np.asarray(x[original]).copy()
            label = STL_LABELS.index(int(y[original]))
            rows.append({"sample_id": f"stl10_{source_split}_{original:05d}", "source_split": source_split,
                         "source_index": int(original), "label": label, "class_name": CLASSES[label],
                         "sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
            arrays.append(pixels)
    frame = pd.DataFrame(rows)
    frame["split"] = "test"
    train_pool = frame.index[frame.source_split == "train"].to_numpy()
    train, val = train_test_split(train_pool, test_size=.2, random_state=42, stratify=frame.loc[train_pool, "label"])
    frame.loc[train, "split"] = "train"
    frame.loc[val, "split"] = "val"
    duplicates = frame[frame.duplicated("sha256", keep=False)].sort_values("sha256")
    # Holdout precedence prevents exact duplicates leaking into the reference index.
    priority = {"test": 0, "val": 1, "train": 2}
    ordered = frame.assign(priority=frame.split.map(priority)).sort_values(["priority", "sample_id"])
    excluded = ordered[ordered.duplicated("sha256", keep="first")].index
    frame.loc[excluded, "split"] = "excluded_duplicate"
    review_path = root / "data/curation/near_duplicate_review.csv"
    excluded_near = []
    if review_path.exists():
        review = pd.read_csv(review_path)
        excluded_near = review.loc[review.confirmed_duplicate == True, "exclude_id"].dropna().unique().tolist()
        if not set(excluded_near).issubset(set(frame.sample_id)):
            raise ValueError("Curation sample IDs do not belong to this dataset")
        frame.loc[frame.sample_id.isin(excluded_near), "split"] = "excluded_near_duplicate"
    folder = root / "data/processed"
    folder.mkdir(parents=True, exist_ok=True)
    images = np.stack(arrays)
    np.save(folder / "images.npy", images)
    frame.to_csv(folder / "manifest.csv", index=False)
    duplicates.to_csv(folder / "exact_duplicates.csv", index=False)
    save_json(folder / "summary.json", {"classes": CLASSES, "seed": 42,
        "counts": frame.groupby(["split", "class_name"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "excluded_exact_duplicates": len(excluded), "source": URL,
        "excluded_near_duplicates": len(excluded_near),
        "protocol": "Six-class supervised transfer-learning subset; not official STL-10 10-fold protocol"})
    print(frame.groupby(["split", "class_name"]).size().unstack(fill_value=0).to_string(), flush=True)
    return frame


def load(root=ROOT):
    folder = Path(root) / "data/processed"
    return np.load(folder / "images.npy", mmap_mode="r"), pd.read_csv(folder / "manifest.csv")


def audit_near_duplicates(root=ROOT):
    from scipy.fft import dctn
    from sklearn.neighbors import NearestNeighbors
    images, frame = load(root)
    hashes = []
    for pixels in images:
        gray = np.asarray(Image.fromarray(pixels).convert("L").resize((32, 32)), dtype=float)
        low = dctn(gray, norm="ortho")[:8, :8].ravel()[1:]
        hashes.append(low > np.median(low))
    hashes = np.asarray(hashes, dtype=np.uint8)
    rows = []
    for ref_split, hold_splits in [("train", ["val", "test"]), ("val", ["test"])]:
        train_idx = frame.index[frame.split == ref_split].to_numpy()
        hold_idx = frame.index[frame.split.isin(hold_splits)].to_numpy()
        nn = NearestNeighbors(n_neighbors=1, metric="hamming", algorithm="brute", n_jobs=2).fit(hashes[train_idx])
        distances, neighbors = nn.kneighbors(hashes[hold_idx])
        for i, (distance, neighbor) in enumerate(zip(distances[:, 0], neighbors[:, 0])):
            if distance * 63 <= 5:
                a, b = int(hold_idx[i]), int(train_idx[neighbor])
                rows.append({"holdout_id": frame.iloc[a].sample_id, "holdout_index": a,
                             "train_id": frame.iloc[b].sample_id, "train_index": b, "reference_split": ref_split,
                             "hamming_bits": round(distance * 63), "status": "candidate_needs_visual_review"})
    result = pd.DataFrame(rows, columns=["holdout_id", "holdout_index", "train_id", "train_index", "reference_split", "hamming_bits", "status"])
    result.to_csv(Path(root) / "data/processed/near_duplicate_candidates.csv", index=False)
    print(f"Near-duplicate candidates (not automatically treated as duplicates): {len(result)}", flush=True)
    return result
