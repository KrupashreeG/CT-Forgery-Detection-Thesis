# ═══════════════════════════════════════════════════════════════
# CELL 1 — Master Cell SETUP
# ═══════════════════════════════════════════════════════════════

# Mount Drive
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
print("✓ Drive mounted")

# Install packages
import subprocess
subprocess.run(['pip','install','pydicom','numpy',
                'matplotlib','scikit-image','Pillow','-q'],
               capture_output=True)
print("✓ Packages installed")

# Import everything
import os, random, time
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
print("✓ Libraries imported")

# All paths
BASE          = "/content/gdrive/MyDrive/thesis_project"
DATASET_PATH  = f"{BASE}/dataset/LIDC-IDRI"
TARGETED_PATH = f"{BASE}/patches/targeted_authentic"
PROGRESS_FILE = f"{BASE}/progress.txt"

os.makedirs(TARGETED_PATH, exist_ok=True)
print("✓ Paths set")

# Load exactly 200 patients — always same 200
all_patients = sorted([
    f for f in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, f))
])
patients = all_patients[:200]
print(f"✓ {len(patients)} patients loaded")
print(f"  First: {patients[0]}")
print(f"  Last:  {patients[-1]}")

# ── All functions defined here ──────────────────────────────

def find_dicom_files(folder):
    files = []
    for root, dirs, fnames in os.walk(folder):
        for f in fnames:
            if f.endswith('.dcm'):
                files.append(os.path.join(root, f))
    return sorted(files)

def load_ct_volume(folder):
    files = find_dicom_files(folder)
    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
            if not hasattr(dcm, 'ImagePositionPatient'): continue
            if not hasattr(dcm, 'pixel_array'):          continue
            if not hasattr(dcm, 'RescaleSlope'):         continue
            slices.append(dcm)
        except Exception:
            continue
    if not slices:
        raise ValueError(f"No valid slices in {folder}")
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return np.stack([
        s.pixel_array * s.RescaleSlope + s.RescaleIntercept
        for s in slices
    ], axis=0)

def extract_patches(volume, patch_size=128, max_per_slice=15):
    np.random.seed(42)
    patches = []
    for i in range(volume.shape[0]):
        sl = volume[i]
        # Lung parenchyma mask — pure air filled tissue only
        mask = (sl > -950) & (sl < -500)
        # Skip if not enough parenchyma
        if mask.sum() / sl.size < 0.15:
            continue
        # Both lungs must be present
        if mask[:, :256].sum() / mask[:, :256].size < 0.08:
            continue
        if mask[:, 256:].sum() / mask[:, 256:].size < 0.08:
            continue
        # Normalise to 0-255
        sl_norm = np.clip(sl, -1000, 400)
        sl_norm = ((sl_norm + 1000) / 1400 * 255).astype(np.uint8)
        # Find valid patch centres inside parenchyma
        m = patch_size // 2
        coords = np.argwhere(mask)
        coords = coords[
            (coords[:,0] > m) & (coords[:,0] < 512-m) &
            (coords[:,1] > m) & (coords[:,1] < 512-m)
        ]
        if len(coords) < 10:
            continue
        chosen = np.random.choice(
            len(coords), min(max_per_slice, len(coords)), replace=False
        )
        for ci in chosen:
            cy, cx = coords[ci]
            p = sl_norm[cy-m:cy+m, cx-m:cx+m]
            if p.shape != (patch_size, patch_size):
                continue
            # Must be at least 30% dark (lung tissue)
            if (p < 80).sum() / p.size < 0.30:
                continue
            patches.append(p)
    return patches

def save_patch(patch, folder, filename):
    Image.fromarray(patch).convert('RGB').save(
        os.path.join(folder, filename)
    )

def get_done():
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE) as f:
        return set(l.strip() for l in f if l.strip())

def mark_done(name):
    with open(PROGRESS_FILE, 'a') as f:
        f.write(name + '\n')

print("✓ Functions defined")
print()
print("══════════════════════════════════")
print("  SETUP COMPLETE")
print("══════════════════════════════════")


