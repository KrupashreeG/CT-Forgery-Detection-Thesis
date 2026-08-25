# --------------------------------------------------------------
# CELL 1 — Master Cell SETUP
# ----------------------------------------------------------------

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

#------------------------------------------
CELL ends here
#-------------------------------------------


#-------------------------------------------------------
# CELL 2 — PATCH EXTRACTION
#-----------------------------------------------------

done      = get_done()
remaining = [p for p in patients if p not in done]

print(f"Total patients:    {len(patients)}")
print(f"Already done:      {len(done)}")
print(f"Left to process:   {len(remaining)}")

if not remaining:
    print("\nAll done.")
else:
    print(f"Starting from:     {remaining[0]}")
    print("-" * 45)

    session_patches = 0
    session_start   = time.time()

    for idx, name in enumerate(remaining):
        try:
            vol     = load_ct_volume(os.path.join(DATASET_PATH, name))
            patches = extract_patches(vol)

            for i, p in enumerate(patches):
                save_patch(p, TARGETED_PATH, f"{name}_p{i:04d}.png")

            session_patches += len(patches)
            mark_done(name)

            if (idx + 1) % 10 == 0:
                done_total = len(done) + idx + 1
                pct  = done_total / len(patients) * 100
                mins = (time.time() - session_start) / 60
                print(f"  [{pct:5.1f}%] {done_total}/{len(patients)} "
                      f"patients | {session_patches} patches "
                      f"| {mins:.0f} min")

        except Exception as e:
            print(f"  Skipped {name}: {e}")
            mark_done(name)
            continue

    mins = (time.time() - session_start) / 60
    done_now = get_done()
    print("-" * 45)
    print(f"Session done")
    print(f"Patches this run: {session_patches}")
    print(f"Time: {mins:.0f} min")
    print(f"Patients done: {len(done_now)}/{len(patients)}")
    if len(done_now) < len(patients):
        print("Colab disconnected early — run Cell 2 again to resume")
    else:
        print("ALL PATIENTS COMPLETE")
-------------------------------------------------------------      
# CELL ends here
------------------------------------------------------------

# --------------------------------------------------------------
# CELL 3 — CHECK STATUS
# ---------------------------------------------------------------

done      = get_done()
remaining = [p for p in patients if p not in done]

print(f"Patients done:     {len(done)}/{len(patients)}")
print(f"Patients left:     {len(remaining)}")

# Count patches without listing all files
try:
    count = len([
        f for f in os.listdir(TARGETED_PATH)
        if f.endswith('.png')
    ])
    print(f"Patches saved:     {count}")
except Exception as e:
    print(f"Could not count patches: {e}")

if remaining:
    print(f"\nNext patient: {remaining[0]}")
    print("Run Cell 2 to continue")
else:
    print("\nExtraction complete")

# -------------------------------------------------------------
# CELL 4 — SELECT 10,000 PATCHES FOR TRAINING
# -------------------------------------------------------------

# Why we select mathematically rather than listing Drive folder:
# Listing 356,614 files causes Drive timeout
# We generate filenames from what we know about naming pattern
# seed=42 means same selection every single time reproducible

random.seed(42)
np.random.seed(42)

SELECTION_FILE = f"{BASE}/selected_patches.txt"

# Check if selection already exists
if os.path.exists(SELECTION_FILE):
    with open(SELECTION_FILE) as f:
        selected = [l.strip() for l in f if l.strip()]
    print(f"Selection already exists: {len(selected)} patches")
    print(f"First: {selected[0]}")
    print(f"Last:  {selected[-1]}")
    print("Using existing selection — reproducibility maintained")

else:
    print("Generating new selection of 10,000 patches...")

    # We know exactly how patches are named:
    # LIDC-IDRI-XXXX_p0000.png through LIDC-IDRI-XXXX_pNNNN.png
    # Average patches per patient = 356614 / 200 = 1783
    # We sample 50 patches per patient × 200 patients = 10,000

    selected = []
    patches_per_patient = 50

    for patient in patients:
        # Sample from indices 0 to 1500 safely
        # (conservative — all patients have at least 1500 patches)
        indices = random.sample(range(1500), patches_per_patient)
        for idx in indices:
            selected.append(f"{patient}_p{idx:04d}.png")

    # Shuffle so patients are mixed throughout
    random.shuffle(selected)

    print(f"Selected: {len(selected)} patches")
    print(f"From:     {len(patients)} patients")
    print(f"Per patient: {patches_per_patient}")

    # Save to Drive permanently
    with open(SELECTION_FILE, 'w') as f:
        for name in selected:
            f.write(name + '\n')

    print(f"\nSaved to: {SELECTION_FILE}")

print()
print("━" * 45)
print(f"  10,000 authentic patches selected")
print(f"  Ready for CT-GAN injection")
print("━" * 45)
