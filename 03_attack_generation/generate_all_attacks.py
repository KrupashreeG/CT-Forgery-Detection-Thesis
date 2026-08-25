# --------------------------------------
# CELL 1 — MASTER SETUP
#--------------------------------------

# Mount Drive
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
print("✓ Drive mounted")

# Imports
import os, random, time, subprocess, sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
print("✓ Libraries imported")

# Paths
BASE           = "/content/gdrive/MyDrive/thesis_project"
TARGETED_PATH  = f"{BASE}/patches/targeted_authentic"
TAMPERED_PATH  = f"{BASE}/patches/tampered"
SELECTION_FILE = f"{BASE}/selected_patches.txt"
Guassian_PROGRESS = f"{BASE}/guassian_progress.txt"

os.makedirs(TAMPERED_PATH, exist_ok=True)
print("✓ Paths set")

# Load selection list
with open(SELECTION_FILE) as f:
    selected = [l.strip() for l in f if l.strip()]
print(f"✓ Selection loaded: {len(selected)} patches")

# Progress helpers
def get_ctgan_done():
    if not os.path.exists(CTGAN_PROGRESS):
        return set()
    with open(CTGAN_PROGRESS) as f:
        return set(l.strip() for l in f if l.strip())

def mark_ctgan_done(name):
    with open(CTGAN_PROGRESS, 'a') as f:
        f.write(name + '\n')

print("✓ Functions defined")
print()
print("══════════════════════════════════")
print("  SETUP COMPLETE")
print("══════════════════════════════════")

# ---------------------------------------------------------
# CELL 4 Gussian Injection
# Fully resumable if Colab disconnects
# ---------------------------------------------------------

import numpy as np
from PIL import Image
import os
import time

np.random.seed(42)

# Local output folder for tampered patches
LOCAL_TAMPER  = "/content/tampered"
os.makedirs(LOCAL_TAMPER, exist_ok=True)

# Progress tracking — save to Drive so survives restart
INJECT_PROGRESS = f"{BASE}/injection_progress.txt"

def get_injected():
    if not os.path.exists(INJECT_PROGRESS):
        return set()
    with open(INJECT_PROGRESS) as f:
        return set(l.strip() for l in f if l.strip())

def mark_injected(name):
    with open(INJECT_PROGRESS, 'a') as f:
        f.write(name + '\n')

# ── Injection functions ─────────────────────────────────

def create_nodule_mask(size=128):
    radius = np.random.randint(8, 20)
    margin = radius + 5
    cx = np.random.randint(margin, size - margin)
    cy = np.random.randint(margin, size - margin)
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    sigma = radius / 2.5
    mask = np.exp(-dist**2 / (2 * sigma**2))
    return (mask > 0.1).astype(np.float32) * mask

def inject_nodule(patch_array):
    patch = patch_array.astype(np.float32)
    mask  = create_nodule_mask()
    bg    = np.mean(patch[patch < 80]) if (patch < 80).sum() > 0 else 30
    intensity = np.clip(bg + np.random.uniform(80, 140), 100, 200)
    noise = np.random.normal(0, 8, (128, 128)).astype(np.float32)
    tampered = patch * (1 - mask) + (intensity + noise) * mask
    return np.clip(tampered, 0, 255).astype(np.uint8)

def save_rgb(array, folder, filename):
    Image.fromarray(array).convert('RGB').save(
        os.path.join(folder, filename)
    )

# ── Load selection list ──────────────────────────────────

with open(SELECTION_FILE) as f:
    selected = [l.strip() for l in f if l.strip()]

# Find which ones already injected
done      = get_injected()
remaining = [f for f in selected if f not in done]

print(f"Total selected:    {len(selected)}")
print(f"Already injected:  {len(done)}")
print(f"Remaining:         {len(remaining)}")
print()

if not remaining:
    print("All patches already injected. Ready for training.")
else:
    print(f"Starting from: {remaining[0]}")
    print("Reading from Drive one file at a time — no timeout")
    print("-" * 50)

    start   = time.time()
    success = 0
    errors  = 0

    for idx, fname in enumerate(remaining):
        try:
            # Read authentic patch directly from Drive
            src = os.path.join(TARGETED_PATH, fname)
            img = np.array(Image.open(src).convert('L'))

            # Inject nodule
            np.random.seed(idx)  # different nodule per patch
            tampered = inject_nodule(img)

            # Save tampered to local storage
            tamper_name = fname.replace('.png', '_tampered.png')
            save_rgb(tampered, LOCAL_TAMPER, tamper_name)

            # Mark as done in Drive progress file
            mark_injected(fname)
            success += 1

            # Progress every 500 patches
            if (idx + 1) % 500 == 0:
                done_total = len(done) + success
                pct  = done_total / len(selected) * 100
                mins = (time.time() - start) / 60
                rate = success / (time.time() - start) * 60
                print(f"  [{pct:5.1f}%] {done_total}/{len(selected)} "
                      f"| {mins:.0f} min "
                      f"| ~{rate:.0f} patches/min")

        except Exception as e:
            errors += 1
            mark_injected(fname)  # skip broken files
            continue

    mins = (time.time() - start) / 60
    total_done = len(done) + success
    print("-" * 50)
    print(f"Session complete")
    print(f"Injected this run: {success}")
    print(f"Total done: {total_done}/{len(selected)}")
    print(f"Errors skipped: {errors}")
    print(f"Time: {mins:.1f} minutes")

    if total_done < len(selected):
        print()
        print("Colab disconnected — run this cell again to resume")
    else:
        print()
        print("ALL 10,000 PATCHES INJECTED")
        print(f"Tampered patches in: {LOCAL_TAMPER}")
        print("Ready for next step")

# --------------------------------------------------
# CELL 2 — cycleGan
# -------------------------------------------------

import random
import os

random.seed(99)

# ── Domain B — 400 from tampered folder ─────────────────────
# Tampered folder only has 10,000 files — small enough to list
print("Loading tampered patches list...")
all_tampered = sorted([
    f for f in os.listdir(TAMPERED_PATH)
    if f.endswith('.png')
])
print(f"Tampered available: {len(all_tampered)}")
domain_b = random.sample(all_tampered, 400)
print(f"Domain B selected:  {len(domain_b)}")

# ── Domain A — 400 authentic patches ────────────────────────
# Generate filenames mathematically — no folder listing needed
# We know patients 0001-0200 exist and each has 700-1000 patches
# Use patients 0150-0200 which were NOT in our training selection
# (training used patients 0001-0167 range approximately)

print()
print("Generating Domain A filenames from known patient range...")

domain_a = []
# Use patients 0168-0200 — safely outside training range
held_out_patients = [f"LIDC-IDRI-{i:04d}" for i in range(168, 201)]

per_patient = 400 // len(held_out_patients) + 1

for patient in held_out_patients:
    # Each patient has at least 700 patches
    # Sample from indices 0-699 safely
    indices = random.sample(range(700), per_patient)
    for idx in indices:
        fname = f"{patient}_p{idx:04d}.png"
        domain_a.append(fname)
        if len(domain_a) >= 400:
            break
    if len(domain_a) >= 400:
        break

domain_a = domain_a[:400]
print(f"Domain A selected:  {len(domain_a)}")

# ── Save lists to Drive ──────────────────────────────────────
with open(f"{BASE}/cyclegan_domain_a.txt", 'w') as f:
    for name in domain_a:
        f.write(name + '\n')

with open(f"{BASE}/cyclegan_domain_b.txt", 'w') as f:
    for name in domain_b:
        f.write(name + '\n')

print()
print("Saved to Drive:")
print(f"  cyclegan_domain_a.txt — {len(domain_a)} authentic patches")
print(f"  cyclegan_domain_b.txt — {len(domain_b)} tampered patches")
print()
print("Ready for next cell — copying to local storage")

# ---------------------------------------------------------
# CELL 3 — COPY 800 PATCHES TO LOCAL STORAGE
# -------------------------------------------------------------

import shutil
import time
import os

LOCAL_A = "/content/cyclegan_A"
LOCAL_B = "/content/cyclegan_B"
os.makedirs(LOCAL_A, exist_ok=True)
os.makedirs(LOCAL_B, exist_ok=True)

start = time.time()

# ── Copy Domain A — authentic patches ───────────────────────
print("Copying Domain A (400 authentic patches)...")
a_ok = 0
a_fail = 0

for fname in domain_a:
    src = os.path.join(AUTHENTIC_PATH, fname)
    dst = os.path.join(LOCAL_A, fname)
    if os.path.exists(dst):
        a_ok += 1
        continue
    try:
        shutil.copy2(src, dst)
        a_ok += 1
    except Exception as e:
        a_fail += 1

print(f"  Copied: {a_ok}  Failed: {a_fail}")

# ── Copy Domain B — tampered patches ────────────────────────
print("Copying Domain B (400 tampered patches)...")
b_ok = 0
b_fail = 0

for fname in domain_b:
    src = os.path.join(TAMPERED_PATH, fname)
    dst = os.path.join(LOCAL_B, fname)
    if os.path.exists(dst):
        b_ok += 1
        continue
    try:
        shutil.copy2(src, dst)
        b_ok += 1
    except Exception as e:
        b_fail += 1

print(f"  Copied: {b_ok}  Failed: {b_fail}")

# ── Summary ──────────────────────────────────────────────────
mins = (time.time() - start) / 60
local_a_count = len([f for f in os.listdir(LOCAL_A) if f.endswith('.png')])
local_b_count = len([f for f in os.listdir(LOCAL_B) if f.endswith('.png')])

print()
print(f"Time taken:        {mins:.1f} minutes")
print(f"Domain A in local: {local_a_count}")
print(f"Domain B in local: {local_b_count}")
print()

if local_a_count >= 350 and local_b_count >= 350:
    print("✓ Enough patches for CycleGAN training")
    print("✓ Ready for Cell 4 — CycleGAN architecture")
elif local_a_count == 0:
    print("✗ Domain A still empty")
    print("Check that AUTHENTIC_PATH is correct:")
    print(f"  {AUTHENTIC_PATH}")
    print("And that domain_a list was set in Cell 2")
else:
    print(f"Partial copy — {local_a_count} Domain A, {local_b_count} Domain B")
    print("Enough to proceed — CycleGAN works with fewer patches")


# ------------------------------------------------------------
# CELL 4 — CYCLEGAN ARCHITECTURE
# --------------------------------------------------------------

import torch
import torch.nn as nn

# ── Residual Block ───────────────────────────────────────────
# Used inside generators
# Learns what to CHANGE rather than what everything looks like
# Standard building block in image translation networks

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


# ── Generator ────────────────────────────────────────────────
# Takes a 128x128 patch in one domain
# Outputs a 128x128 patch in the other domain
# Architecture: downsample → residual blocks → upsample

class Generator(nn.Module):
    def __init__(self, n_residual=6):
        super().__init__()

        # Initial convolution
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        # 128x128 → 64x64 → 32x32
        in_ch = 64
        for _ in range(2):
            out_ch = in_ch * 2
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch

        # Residual blocks — where the translation learning happens
        for _ in range(n_residual):
            layers.append(ResidualBlock(in_ch))

        # Upsampling
        # 32x32 → 64x64 → 128x128
        for _ in range(2):
            out_ch = in_ch // 2
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 3,
                                   stride=2, padding=1,
                                   output_padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch

        # Output layer — maps to 3 channel image
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()  # Output range -1 to 1
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ── Discriminator ────────────────────────────────────────────
# PatchGAN discriminator
# Does not classify the whole image as real/fake
# Instead classifies overlapping 70x70 patches
# This focuses on local texture artefacts

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_ch, out_ch, normalise=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4,
                                stride=2, padding=1)]
            if normalise:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(3,   64,  normalise=False),
            *block(64,  128),
            *block(128, 256),
            *block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)


# ── Initialise all 4 networks ────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training device: {device}")

G_AB = Generator().to(device)  # Authentic → Tampered
G_BA = Generator().to(device)  # Tampered  → Authentic
D_A  = Discriminator().to(device)  # Judges authentic patches
D_B  = Discriminator().to(device)  # Judges tampered patches

# Count parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters())

print(f"Generator parameters:      {count_params(G_AB):,}")
print(f"Discriminator parameters:  {count_params(D_A):,}")
print()
print("4 networks initialised:")
print("  G_AB — Authentic to Tampered (zero-day generator)")
print("  G_BA — Tampered to Authentic (cycle consistency)")
print("  D_A  — Authentic discriminator")
print("  D_B  — Tampered discriminator")
print()
print(" Architecture ready")

# -------------------------------------------------
# CELL 5 Dataset and training loop
# ----------------------------------------------------

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import time

class CTDomainDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        # Verify every file on load — skip corrupt ones
        all_files = sorted([
            f for f in os.listdir(folder)
            if f.endswith('.png')
        ])
        self.files = []
        skipped = 0
        for f in all_files:
            try:
                path = os.path.join(folder, f)
                img  = Image.open(path).convert('RGB')
                img.verify()  # Check file is valid
                self.files.append(f)
            except Exception:
                skipped += 1
        if skipped > 0:
            print(f"  Skipped {skipped} corrupt files in {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        img  = Image.open(path).convert('RGB')
        return self.transform(img), self.files[idx]


print("Loading datasets with validation...")
dataset_A = CTDomainDataset("/content/cyclegan_A")
dataset_B = CTDomainDataset("/content/cyclegan_B")

print(f"Domain A valid patches: {len(dataset_A)}")
print(f"Domain B valid patches: {len(dataset_B)}")

loader_A = DataLoader(dataset_A, batch_size=4,
                      shuffle=True, drop_last=True)
loader_B = DataLoader(dataset_B, batch_size=4,
                      shuffle=True, drop_last=True)

# Detect label size
with torch.no_grad():
    test_input = torch.randn(1, 3, 128, 128).to(device)
    test_out   = D_A(test_input)
    label_size = test_out.shape[2]
print(f"Label size: {label_size}x{label_size}")

# Loss functions
criterion_GAN   = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_ident = nn.L1Loss()

# Optimisers
opt_G = torch.optim.Adam(
    list(G_AB.parameters()) + list(G_BA.parameters()),
    lr=0.0002, betas=(0.5, 0.999)
)
opt_D_A = torch.optim.Adam(
    D_A.parameters(), lr=0.0002, betas=(0.5, 0.999)
)
opt_D_B = torch.optim.Adam(
    D_B.parameters(), lr=0.0002, betas=(0.5, 0.999)
)

N_EPOCHS     = 50
LAMBDA_CYCLE = 10.0
LAMBDA_IDENT =  5.0

print()
print(f"Training for {N_EPOCHS} epochs")
print(f"Progress every 5 epochs")
print("-" * 50)

train_start = time.time()

for epoch in range(N_EPOCHS):
    epoch_start = time.time()
    g_losses, d_losses = [], []

    for (real_A, _), (real_B, _) in zip(loader_A, loader_B):
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        bs     = real_A.size(0)

        real_label = torch.ones( bs, 1, label_size, label_size).to(device)
        fake_label = torch.zeros(bs, 1, label_size, label_size).to(device)

        # Train Generators
        opt_G.zero_grad()
        fake_B  = G_AB(real_A)
        fake_A  = G_BA(real_B)
        recov_A = G_BA(fake_B)
        recov_B = G_AB(fake_A)

        loss_G = (
            criterion_GAN(D_B(fake_B), real_label) +
            criterion_GAN(D_A(fake_A), real_label) +
            LAMBDA_CYCLE * (
                criterion_cycle(recov_A, real_A) +
                criterion_cycle(recov_B, real_B)
            ) +
            LAMBDA_IDENT * (
                criterion_ident(G_BA(real_A), real_A) +
                criterion_ident(G_AB(real_B), real_B)
            )
        )
        loss_G.backward()
        opt_G.step()

        # Train Discriminator A
        opt_D_A.zero_grad()
        loss_D_A = (
            criterion_GAN(D_A(real_A),          real_label) +
            criterion_GAN(D_A(fake_A.detach()), fake_label)
        ) * 0.5
        loss_D_A.backward()
        opt_D_A.step()

        # Train Discriminator B
        opt_D_B.zero_grad()
        loss_D_B = (
            criterion_GAN(D_B(real_B),          real_label) +
            criterion_GAN(D_B(fake_B.detach()), fake_label)
        ) * 0.5
        loss_D_B.backward()
        opt_D_B.step()

        g_losses.append(loss_G.item())
        d_losses.append((loss_D_A + loss_D_B).item())

    epoch_mins = (time.time() - epoch_start) / 60
    total_mins = (time.time() - train_start) / 60

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{N_EPOCHS} | "
              f"G: {np.mean(g_losses):.3f} | "
              f"D: {np.mean(d_losses):.3f} | "
              f"{epoch_mins:.1f} min/ep | "
              f"Total: {total_mins:.0f} min")

print("-" * 50)
total_mins = (time.time() - train_start) / 60
print(f"Training complete: {total_mins:.0f} minutes")

torch.save(G_AB.state_dict(), f"{MODEL_PATH}/G_AB_final.pth")
torch.save(G_BA.state_dict(), f"{MODEL_PATH}/G_BA_final.pth")
print(" Ready for next Cell ")

# ----------------------------------------------------------
# CELL 6 — GENERATE ZERO-DAY PATCHES FROM SAVED MODEL
# -----------------------------------------------------------
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Rebuild Generator architecture ───────────────────────────
# Must match exactly what was trained

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, n_residual=6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        in_ch = 64
        for _ in range(2):
            out_ch = in_ch * 2
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch
        for _ in range(n_residual):
            layers.append(ResidualBlock(in_ch))
        for _ in range(2):
            out_ch = in_ch // 2
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 3,
                                   stride=2, padding=1,
                                   output_padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ── Load saved model ─────────────────────────────────────────
print("Loading saved G_AB model...")
G_AB = Generator().to(device)
G_AB.load_state_dict(
    torch.load(f"{MODEL_PATH}/G_AB_final.pth",
               map_location=device)
)
G_AB.eval()
print("✓ Model loaded successfully")

# ── Dataset ──────────────────────────────────────────────────
class CTDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files  = []
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        for f in sorted(os.listdir(folder)):
            if not f.endswith('.png'):
                continue
            try:
                img = Image.open(os.path.join(folder, f))
                img.verify()
                self.files.append(f)
            except Exception:
                continue

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        img  = Image.open(path).convert('RGB')
        return self.transform(img), self.files[idx]

dataset_A = CTDataset("/content/cyclegan_A")
print(f"Input patches: {len(dataset_A)}")

loader_A = DataLoader(dataset_A, batch_size=1, shuffle=False)

# ── Generate zero-day patches ────────────────────────────────
ZERODAY_PATH = f"{BASE}/patches/zeroday"
os.makedirs(ZERODAY_PATH, exist_ok=True)

def tensor_to_image(tensor):
    img = tensor.squeeze().cpu().detach().numpy()
    img = img * 0.5 + 0.5
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)

print("Generating zero-day patches...")
print("-" * 40)

generated = 0
with torch.no_grad():
    for batch, fnames in loader_A:
        batch  = batch.to(device)
        output = G_AB(batch)
        img    = tensor_to_image(output)

        save_name = fnames[0].replace('.png', '_zeroday.png')
        img.save(os.path.join(ZERODAY_PATH, save_name))
        generated += 1

        if generated % 50 == 0:
            print(f"  Generated {generated}/{len(dataset_A)}")

print("-" * 40)
print(f"✓ Zero-day patches generated: {generated}")
print(f"✓ Saved to: {ZERODAY_PATH}")
print()
print("Your complete dataset:")
print(f"  Authentic patches:  10,000")
print(f"  Tampered patches:   10,000 (seen attack)")
print(f"  Zero-day patches:   {generated} (unseen attack)")
print()
print("STAGE 3 COMPLETE")

# ═══════════════════════════════════════════════════════════════
# CELL 7 — ATTACK 2: ELLIPTICAL INJECTION
#
# What it does:
#   Same principle as Gaussian injection but uses an ellipse
#   instead of a circle. The nodule is oval shaped.
#
# Why it is different from Gaussian injection:
#   Different shape signature — detector trained only on
#   circular blobs may not recognise elliptical ones
#
# How it works:
#   Ellipse equation: (x-cx)²/rx² + (y-cy)²/ry² <= 1
#   rx and ry are different radii giving the oval shape
#   Gaussian falloff applied for soft edges same as before
# ═══════════════════════════════════════════════════════════════

def elliptical_injection(patch_array, seed):
    np.random.seed(seed)
    patch = patch_array.astype(np.float32)
    size  = 128

    # Random ellipse centre
    cx = np.random.randint(20, size-20)
    cy = np.random.randint(20, size-20)

    # Different radii for x and y — this makes it elliptical
    rx = np.random.randint(8, 22)
    ry = np.random.randint(5, 15)

    # Ellipse distance formula
    y, x = np.ogrid[:size, :size]
    dist = ((x - cx) / rx)**2 + ((y - cy) / ry)**2

    # Gaussian falloff — soft edges
    mask = np.exp(-dist * 2.0)
    mask = (mask > 0.1).astype(np.float32) * mask

    # Nodule intensity
    bg        = np.mean(patch[patch < 80]) \
                if (patch < 80).sum() > 0 else 30
    intensity = np.clip(bg + np.random.uniform(70, 140), 100, 200)
    noise     = np.random.normal(0, 10, (size, size)).astype(np.float32)

    # Alpha blend
    tampered = patch * (1 - mask) + (intensity + noise) * mask
    return np.clip(tampered, 0, 255).astype(np.uint8)


# Check which patches already done
existing = set(os.listdir(ATK2_PATH))
done     = set(f.replace('_elliptical.png', '.png')
               for f in existing)
todo     = [f for f in selected if f not in done]

print(f"Attack 2 — Elliptical Injection")
print(f"  Already done: {len(done)}")
print(f"  Remaining:    {len(todo)}")
print()

if not todo:
    print("All 10,000 already complete!")
else:
    print("Generating...")
    start   = time.time()
    success = 0
    errors  = 0

    for i, fname in enumerate(todo):
        try:
            patch    = load_patch(fname)
            tampered = elliptical_injection(patch, seed=i+1000)
            save_patch(tampered, ATK2_PATH, fname, 'elliptical')
            success += 1
        except Exception as e:
            errors += 1
            continue

        if (i+1) % 1000 == 0:
            mins = (time.time() - start) / 60
            pct  = (len(done) + success) / len(selected) * 100
            print(f"  [{pct:.1f}%] {len(done)+success}/{len(selected)} "
                  f"| {mins:.1f} min")

    mins  = (time.time() - start) / 60
    total = len(os.listdir(ATK2_PATH))
    print(f"\nDone: {success} generated | {errors} errors | {mins:.1f} min")
    print(f"Total in Drive: {total}")
    print("✓ Attack complete")

# ═══════════════════════════════════════════════════════════════
# CELL 8 — ATTACK 3: COPY-MOVE
#
# What it does:
#   Copies a region from one part of the patch and pastes it
#   into a different location in the same patch.
#
# Why it is different:
#   No brightness injection. No external content added.
#   The patch contains only its own pixels rearranged.
#   Creates boundary discontinuity artefacts.
#   Classical image forgery technique used in real cases.
# ═══════════════════════════════════════════════════════════════

def copy_move_attack(patch_array, seed):
    np.random.seed(seed)
    patch = patch_array.astype(np.float32)
    size  = 128

    region_size = np.random.randint(12, 24)
    margin      = region_size + 5

    # Source location — pick a brighter area (vessel or tissue)
    src_y = np.random.randint(margin, size - margin)
    src_x = np.random.randint(margin, size - margin)

    # Destination — must not overlap source
    for _ in range(50):
        dst_y = np.random.randint(margin, size - margin)
        dst_x = np.random.randint(margin, size - margin)
        if (abs(dst_y - src_y) > region_size or
                abs(dst_x - src_x) > region_size):
            break

    # Extract source region
    half = region_size // 2
    source_region = patch[
        max(0, src_y-half):src_y+half,
        max(0, src_x-half):src_x+half
    ].copy()

    if source_region.shape[0] < 4 or source_region.shape[1] < 4:
        return patch.astype(np.uint8)

    # Smooth blending mask — soft edges
    h, w = source_region.shape
    cy_m, cx_m = h//2, w//2
    y_g, x_g   = np.ogrid[:h, :w]
    dist_m      = np.sqrt((x_g-cx_m)**2 + (y_g-cy_m)**2)
    max_r       = min(cy_m, cx_m)
    blend_mask  = np.clip(1 - dist_m / max(max_r, 1), 0, 1)

    # Paste with blending
    tampered = patch.copy()
    y1 = max(0, dst_y - half)
    x1 = max(0, dst_x - half)
    y2 = min(size, y1 + h)
    x2 = min(size, x1 + w)

    h_act = y2 - y1
    w_act = x2 - x1

    if h_act > 0 and w_act > 0:
        m = blend_mask[:h_act, :w_act]
        tampered[y1:y2, x1:x2] = (
            source_region[:h_act, :w_act] * m +
            patch[y1:y2, x1:x2] * (1 - m)
        )

    return np.clip(tampered, 0, 255).astype(np.uint8)


# Check which patches already done
existing = set(os.listdir(ATK3_PATH))
done     = set(f.replace('_copymove.png', '.png')
               for f in existing)
todo     = [f for f in selected if f not in done]

print(f"Attack 3 — Copy-Move")
print(f"  Already done: {len(done)}")
print(f"  Remaining:    {len(todo)}")
print()

if not todo:
    print("All 10,000 already complete!")
else:
    print("Generating...")
    start = time.time()
    success = errors = 0

    for i, fname in enumerate(todo):
        try:
            patch    = load_patch(fname)
            tampered = copy_move_attack(patch, seed=i+2000)
            save_patch(tampered, ATK3_PATH, fname, 'copymove')
            success += 1
        except Exception:
            errors += 1
            continue

        if (i+1) % 1000 == 0:
            mins = (time.time() - start) / 60
            pct  = (len(done)+success) / len(selected) * 100
            print(f"  [{pct:.1f}%] {len(done)+success}/{len(selected)} "
                  f"| {mins:.1f} min")

    mins  = (time.time() - start) / 60
    total = len(os.listdir(ATK3_PATH))
    print(f"\nDone: {success} | Errors: {errors} | {mins:.1f} min")
    print(f"Total in Drive: {total}")
    print("✓ Attack  complete")

# ═══════════════════════════════════════════════════════════════
# CELL 9— ATTACK 4: FREQUENCY DOMAIN
#
# What it does:
#   Converts the patch to the frequency domain using DCT.
#   Modifies specific frequency coefficients.
#   Converts back to the spatial domain.
#
# Why it is different from all others:
#   Does NOT inject any visible object in the spatial domain.
#   The image looks identical to the human eye.
#   But the frequency signature changes in a detectable way.
#   Completely different mathematical basis from GAN attacks.
# ═══════════════════════════════════════════════════════════════

def frequency_domain_attack(patch_array, seed, strength=18):
    np.random.seed(seed)
    patch = patch_array.astype(np.float32)

    # Apply 2D Discrete Cosine Transform
    # This decomposes the image into frequency components
    dct_patch = cv2.dct(patch)

    h, w = dct_patch.shape

    # Modify mid-frequency band
    # Low frequencies (top-left of DCT) = global brightness
    # Mid frequencies = texture and edges
    # High frequencies (bottom-right) = fine detail / noise
    # We modify mid-frequencies — visible to detector, not eye

    tampered_dct = dct_patch.copy()

    # Inject structured pattern into 8x8 blocks
    # This mimics how JPEG compression works but with added artefacts
    for i in range(2, min(h-2, 40), 6):
        for j in range(2, min(w-2, 40), 6):
            # Random signed injection
            injection = strength * np.random.uniform(0.8, 2.2)
            sign      = np.random.choice([-1, 1])
            tampered_dct[i:i+2, j:j+2] += sign * injection

    # Inverse DCT — convert back to spatial domain
    tampered = cv2.idct(tampered_dct)

    return np.clip(tampered, 0, 255).astype(np.uint8)


# Check which already done
existing = set(os.listdir(ATK4_PATH))
done     = set(f.replace('_frequency.png', '.png')
               for f in existing)
todo     = [f for f in selected if f not in done]

print(f"Attack 4 — Frequency Domain")
print(f"  Already done: {len(done)}")
print(f"  Remaining:    {len(todo)}")
print()

if not todo:
    print("All 10,000 already complete!")
else:
    print("Generating...")
    start = time.time()
    success = errors = 0

    for i, fname in enumerate(todo):
        try:
            patch    = load_patch(fname)
            tampered = frequency_domain_attack(patch, seed=i+3000)
            save_patch(tampered, ATK4_PATH, fname, 'frequency')
            success += 1
        except Exception:
            errors += 1
            continue

        if (i+1) % 1000 == 0:
            mins = (time.time() - start) / 60
            pct  = (len(done)+success) / len(selected) * 100
            print(f"  [{pct:.1f}%] {len(done)+success}/{len(selected)} "
                  f"| {mins:.1f} min")

    mins  = (time.time() - start) / 60
    total = len(os.listdir(ATK4_PATH))
    print(f"\nDone: {success} | Errors: {errors} | {mins:.1f} min")
    print(f"Total in Drive: {total}")
    print("✓ Attack complete")

# ═══════════════════════════════════════════════════════════════
# CELL 10 — ATTACK 6: TEXTURE SYNTHESIS
#
# What it does:
#   Samples texture from a bright area of the patch itself.
#   Synthesises a nodule-like region using that texture.
#   Blends it into a dark lung area.
#
# Why it is different:
#   Content comes from within the patch (not external).
#   The nodule is built from real CT tissue texture.
#   More realistic than Gaussian injection.
#   Different boundary characteristics.
# ═══════════════════════════════════════════════════════════════

def texture_synthesis_attack(patch_array, seed):
    np.random.seed(seed)
    patch = patch_array.astype(np.float32)
    size  = 128

    # Target location for synthesised nodule
    radius = np.random.randint(10, 20)
    margin = radius + 8
    cx     = np.random.randint(margin, size - margin)
    cy     = np.random.randint(margin, size - margin)

    # Find bright pixels to sample texture from
    threshold = np.percentile(patch, 65)
    y_bright, x_bright = np.where(patch > threshold)

    if len(y_bright) < 20:
        # Fall back to Gaussian if not enough bright pixels
        y_bright, x_bright = np.where(patch > patch.mean())

    if len(y_bright) == 0:
        return patch.astype(np.uint8)

    # Sample a small texture tile from a bright region
    tile_size = 10
    tries     = 0
    texture   = None

    while tries < 20 and texture is None:
        idx   = np.random.randint(len(y_bright))
        sy, sx = y_bright[idx], x_bright[idx]

        y1 = max(0, sy - tile_size)
        y2 = min(size, sy + tile_size)
        x1 = max(0, sx - tile_size)
        x2 = min(size, sx + tile_size)

        tile = patch[y1:y2, x1:x2]
        if tile.shape[0] >= 6 and tile.shape[1] >= 6:
            texture = tile
        tries += 1

    if texture is None:
        return patch.astype(np.uint8)

    # Tile the texture to fill nodule area
    nodule_diam = radius * 2
    tex_h, tex_w = texture.shape
    repeats_y = (nodule_diam // tex_h) + 2
    repeats_x = (nodule_diam // tex_w) + 2
    tiled     = np.tile(texture, (repeats_y, repeats_x))
    nodule_tex = tiled[:nodule_diam, :nodule_diam]

    if nodule_tex.shape != (nodule_diam, nodule_diam):
        return patch.astype(np.uint8)

    # Add slight brightness boost to make it look like a nodule
    boost      = np.random.uniform(20, 60)
    nodule_tex = np.clip(nodule_tex + boost, 0, 255)

    # Gaussian blending mask
    y_g, x_g = np.ogrid[:size, :size]
    dist_g    = np.sqrt((x_g - cx)**2 + (y_g - cy)**2)
    mask      = np.exp(-dist_g**2 / (2 * (radius/2.5)**2))
    mask      = (mask > 0.1).astype(np.float32) * mask

    # Paste synthesised nodule
    tampered = patch.copy()
    y1 = max(0, cy - radius)
    x1 = max(0, cx - radius)
    y2 = min(size, y1 + nodule_diam)
    x2 = min(size, x1 + nodule_diam)

    h_act = y2 - y1
    w_act = x2 - x1

    if h_act > 0 and w_act > 0:
        m = mask[y1:y2, x1:x2]
        tampered[y1:y2, x1:x2] = (
            nodule_tex[:h_act, :w_act] * m +
            patch[y1:y2, x1:x2] * (1 - m)
        )

    return np.clip(tampered, 0, 255).astype(np.uint8)


# Check which already done
existing = set(os.listdir(ATK6_PATH))
done     = set(f.replace('_texture.png', '.png')
               for f in existing)
todo     = [f for f in selected if f not in done]

print(f"Attack 6 — Texture Synthesis")
print(f"  Already done: {len(done)}")
print(f"  Remaining:    {len(todo)}")
print()

if not todo:
    print("All 10,000 already complete!")
else:
    print("Generating...")
    start = time.time()
    success = errors = 0

    for i, fname in enumerate(todo):
        try:
            patch    = load_patch(fname)
            tampered = texture_synthesis_attack(patch, seed=i+6000)
            save_patch(tampered, ATK6_PATH, fname, 'texture')
            success += 1
        except Exception:
            errors += 1
            continue

        if (i+1) % 1000 == 0:
            mins = (time.time() - start) / 60
            pct  = (len(done)+success) / len(selected) * 100
            print(f"  [{pct:.1f}%] {len(done)+success}/{len(selected)} "
                  f"| {mins:.1f} min")

    mins  = (time.time() - start) / 60
    total = len(os.listdir(ATK6_PATH))
    print(f"\nDone: {success} | Errors: {errors} | {mins:.1f} min")
    print(f"Total in Drive: {total}")
    print("✓ Attack 6 complete")

# ═══════════════════════════════════════════════════════════════
# ATTACK 7: POISSON BLENDING
#
# What it does:
#   Takes a brighter circular region from the patch itself
#   and uses OpenCV seamlessClone to paste it somewhere else.
#   The boundary is mathematically seamless — edges blend
#   perfectly but the region content is statistically wrong.
#
# Why it is different:
#   No visible hard edge at the boundary (unlike copy-move)
#   The forgery is nearly invisible but statistically detectable
#   Different mathematical basis from all other attacks
# ═══════════════════════════════════════════════════════════════

def poisson_blending_attack(patch_array, seed):
    np.random.seed(seed)
    patch = patch_array.astype(np.uint8)
    size  = 128

    # Convert to 3-channel for OpenCV seamlessClone
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

    radius = np.random.randint(10, 20)
    margin = radius + 12

    if margin >= size - margin:
        return patch

    cx = np.random.randint(margin, size - margin)
    cy = np.random.randint(margin, size - margin)

    # Create circular mask
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    # Source: create a brighter version of the patch
    # This simulates a denser tissue region
    bright_val = int(np.mean(patch) +
                     np.random.uniform(50, 120))
    bright_val = min(bright_val, 230)

    source     = patch_rgb.copy()
    noise      = np.random.normal(
        0, 8, source.shape).astype(np.int16)
    source_mod = np.clip(
        source.astype(np.int16) + bright_val//3 + noise,
        0, 255).astype(np.uint8)

    # Poisson blending — seamless edges
    center = (cx, cy)
    try:
        result_rgb = cv2.seamlessClone(
            source_mod, patch_rgb, mask,
            center, cv2.NORMAL_CLONE)
        result = cv2.cvtColor(result_rgb, cv2.COLOR_BGR2GRAY)
        return result
    except Exception:
        return patch


# Generate
existing = set(os.listdir(ATK7))
done     = set(f.replace('_poisson.png', '.png')
               for f in existing)
todo     = [f for f in selected if f not in done]

print(f"Attack 7 — Poisson Blending")
print(f"  Already done: {len(done)}")
print(f"  Remaining:    {len(todo)}")
print()

if not todo:
    print("All 10,000 already complete!")
else:
    print("Generating...")
    start = time.time()
    success = errors = 0

    for i, fname in enumerate(todo):
        try:
            tamp_name = fname.replace('.png', '_tampered.png')
            patch     = np.array(
                Image.open(os.path.join(
                    LOCAL_TAMP, tamp_name)).convert('L'))
            tampered  = poisson_blending_attack(patch, seed=i+7000)
            save_patch(tampered, ATK7, fname, 'poisson')
            success += 1
        except Exception:
            errors += 1

        if (i+1) % 1000 == 0:
            mins = (time.time()-start)/60
            pct  = (len(done)+success)/len(selected)*100
            print(f"  [{pct:.1f}%] {len(done)+success}/"
                  f"{len(selected)} | {mins:.1f} min")

    mins  = (time.time()-start)/60
    total = len(os.listdir(ATK7))
    print(f"\nDone: {success} | Errors: {errors} | {mins:.1f} min")
    print(f"Total in Drive: {total}")
    print("✓ Attack complete")

# ═══════════════════════════════════════════════════════════════
# ATTACK 8: LOCAL CONTRAST ENHANCEMENT
#
# What it does:
#   Applies histogram equalisation or CLAHE to a circular
#   region inside the patch. The region gets enhanced contrast
#   making subtle tissue differences more pronounced.
#   This simulates a region of abnormal tissue density.
#
# Why it is different:
#   Preserves all original pixel content — just redistributes
#   the intensity range within the circular region.
#   No new content added. Different from all injection methods.
#   Creates a distinctive local contrast discontinuity.
# ═══════════════════════════════════════════════════════════════

def contrast_enhancement_attack(patch_array, seed):
    np.random.seed(seed)
    patch = patch_array.astype(np.uint8)
    size  = 128

    radius = np.random.randint(12, 22)
    margin = radius + 5
    cx     = np.random.randint(margin, size-margin)
    cy     = np.random.randint(margin, size-margin)

    # Create circular mask
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x-cx)**2 + (y-cy)**2)
    mask = (dist <= radius).astype(np.float32)

    tampered = patch.copy().astype(np.float32)

    # Extract region
    y1 = max(0, cy-radius)
    y2 = min(size, cy+radius+1)
    x1 = max(0, cx-radius)
    x2 = min(size, cx+radius+1)
    region = patch[y1:y2, x1:x2].copy()

    if region.size == 0:
        return patch

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # to the extracted region
    clahe = cv2.createCLAHE(
        clipLimit=np.random.uniform(2.0, 5.0),
        tileGridSize=(4, 4)
    )
    enhanced = clahe.apply(region)

    # Blend enhanced region back with smooth edges
    blend_mask = mask[y1:y2, x1:x2]
    tampered[y1:y2, x1:x2] = (
        enhanced.astype(np.float32) * blend_mask +
        region.astype(np.float32) * (1 - blend_mask)
    )

    return np.clip(tampered, 0, 255).astype(np.uint8)


# Generate
existing = set(os.listdir(ATK8))
done     = set(f.replace('_contrast.png', '.png')
               for f in existing)
todo     = [f for f in selected if f not in done]

print(f"Attack 8 — Local Contrast Enhancement")
print(f"  Already done: {len(done)}")
print(f"  Remaining:    {len(todo)}")
print()

if not todo:
    print("All 10,000 already complete!")
else:
    print("Generating...")
    start = time.time()
    success = errors = 0

    for i, fname in enumerate(todo):
        try:
            tamp_name = fname.replace('.png', '_tampered.png')
            patch     = np.array(
                Image.open(os.path.join(
                    LOCAL_TAMP, tamp_name)).convert('L'))
            tampered  = contrast_enhancement_attack(
                patch, seed=i+8000)
            save_patch(tampered, ATK8, fname, 'contrast')
            success += 1
        except Exception:
            errors += 1

        if (i+1) % 1000 == 0:
            mins = (time.time()-start)/60
            pct  = (len(done)+success)/len(selected)*100
            print(f"  [{pct:.1f}%] {len(done)+success}/"
                  f"{len(selected)} | {mins:.1f} min")

    mins  = (time.time()-start)/60
    total = len(os.listdir(ATK8))
    print(f"\nDone: {success} | Errors: {errors} | {mins:.1f} min")
    print(f"Total in Drive: {total}")
    print("✓ Attack complete")

# ═══════════════════════════════════════════════════════════════
# ATTACK 9: SMOOTHING INJECTION
#
# What it does:
#   Applies strong Gaussian blur to a circular region.
#   The blurred region loses vessel structure and fine detail
#   making it look like a mass or consolidated region.
#
# Why it is different:
#   No brightness change — only texture smoothing.
#   Real nodules often appear smooth compared to surroundings.
#   Different statistical signature from all injection types.
#   The vessel detail inside the region disappears.
# ═══════════════════════════════════════════════════════════════

def smoothing_attack(patch_array, seed):
    np.random.seed(seed)
    patch = patch_array.astype(np.float32)
    size  = 128

    radius = np.random.randint(12, 24)
    margin = radius + 5
    cx     = np.random.randint(margin, size-margin)
    cy     = np.random.randint(margin, size-margin)

    # Gaussian soft mask
    y, x  = np.ogrid[:size, :size]
    dist  = np.sqrt((x-cx)**2 + (y-cy)**2)
    sigma = radius / 2.0
    mask  = np.exp(-dist**2 / (2*sigma**2))
    mask  = (mask > 0.2).astype(np.float32) * mask

    # Apply strong Gaussian blur to whole patch
    blur_size = np.random.choice([9, 11, 13, 15])
    blurred   = cv2.GaussianBlur(
        patch.astype(np.uint8),
        (blur_size, blur_size), 0
    ).astype(np.float32)

    # Optionally slightly brighten the blurred region
    # to simulate tissue consolidation
    brightness_boost = np.random.uniform(10, 35)
    blurred_boosted  = blurred + brightness_boost * mask

    # Blend: inside circle use blurred+boosted, outside keep original
    tampered = patch * (1 - mask) + blurred_boosted * mask

    return np.clip(tampered, 0, 255).astype(np.uint8)


# Generate
existing = set(os.listdir(ATK9))
done     = set(f.replace('_smoothing.png', '.png')
               for f in existing)
todo     = [f for f in selected if f not in done]

print(f"Attack 9 — Smoothing Injection")
print(f"  Already done: {len(done)}")
print(f"  Remaining:    {len(todo)}")
print()

if not todo:
    print("All 10,000 already complete!")
else:
    print("Generating...")
    start = time.time()
    success = errors = 0

    for i, fname in enumerate(todo):
        try:
            tamp_name = fname.replace('.png', '_tampered.png')
            patch     = np.array(
                Image.open(os.path.join(
                    LOCAL_TAMP, tamp_name)).convert('L'))
            tampered  = smoothing_attack(patch, seed=i+9000)
            save_patch(tampered, ATK9, fname, 'smoothing')
            success += 1
        except Exception:
            errors += 1

        if (i+1) % 1000 == 0:
            mins = (time.time()-start)/60
            pct  = (len(done)+success)/len(selected)*100
            print(f"  [{pct:.1f}%] {len(done)+success}/"
                  f"{len(selected)} | {mins:.1f} min")

    mins  = (time.time()-start)/60
    total = len(os.listdir(ATK9))
    print(f"\nDone: {success} | Errors: {errors} | {mins:.1f} min")
    print(f"Total in Drive: {total}")
    print("✓ Attack complete")

# ═══════════════════════════════════════════════════════════════
# ATTACK 10: SHARPENING INJECTION
#
# What it does:
#   Applies unsharp masking (over-sharpening) to a circular
#   region. Edges and vessel walls become artificially enhanced.
#   Creates a region that looks hyper-detailed and unnatural.
#
# Why it is different:
#   Opposite of smoothing — increases high-frequency content
#   Creates a distinctive ringing artefact at edges
#   No brightness or shape injection — purely texture change
# ═══════════════════════════════════════════════════════════════

def sharpening_attack(patch_array, seed):
    np.random.seed(seed)
    patch = patch_array.astype(np.float32)
    size  = 128

    radius = np.random.randint(12, 24)
    margin = radius + 5
    cx     = np.random.randint(margin, size-margin)
    cy     = np.random.randint(margin, size-margin)

    # Gaussian soft mask
    y, x  = np.ogrid[:size, :size]
    dist  = np.sqrt((x-cx)**2 + (y-cy)**2)
    sigma = radius / 2.0
    mask  = np.exp(-dist**2 / (2*sigma**2))
    mask  = (mask > 0.2).astype(np.float32) * mask

    # Unsharp masking: sharpened = original + strength*(original - blurred)
    blur_size = 5
    blurred   = cv2.GaussianBlur(
        patch.astype(np.uint8),
        (blur_size, blur_size), 0
    ).astype(np.float32)

    strength  = np.random.uniform(1.5, 3.5)
    sharpened = patch + strength * (patch - blurred)

    # Blend: apply sharpening only inside the circular region
    tampered = patch * (1 - mask) + sharpened * mask

    return np.clip(tampered, 0, 255).astype(np.uint8)


# Generate
existing = set(os.listdir(ATK10))
done     = set(f.replace('_sharpening.png', '.png')
               for f in existing)
todo     = [f for f in selected if f not in done]

print(f"Attack 10 — Sharpening Injection")
print(f"  Already done: {len(done)}")
print(f"  Remaining:    {len(todo)}")
print()

if not todo:
    print("All 10,000 already complete!")
else:
    print("Generating...")
    start = time.time()
    success = errors = 0

    for i, fname in enumerate(todo):
        try:
            tamp_name = fname.replace('.png', '_tampered.png')
            patch     = np.array(
                Image.open(os.path.join(
                    LOCAL_TAMP, tamp_name)).convert('L'))
            tampered  = sharpening_attack(patch, seed=i+10000)
            save_patch(tampered, ATK10, fname, 'sharpening')
            success += 1
        except Exception:
            errors += 1

        if (i+1) % 1000 == 0:
            mins = (time.time()-start)/60
            pct  = (len(done)+success)/len(selected)*100
            print(f"  [{pct:.1f}%] {len(done)+success}/"
                  f"{len(selected)} | {mins:.1f} min")

    mins  = (time.time()-start)/60
    total = len(os.listdir(ATK10))
    print(f"\nDone: {success} | Errors: {errors} | {mins:.1f} min")
    print(f"Total in Drive: {total}")
    print("✓ Attack complete")

