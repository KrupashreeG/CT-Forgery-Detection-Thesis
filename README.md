# Forensic Detection of Pathological Forgeries in Clinical CT Imaging

A cybersecurity threat modelling and generalisation gap analysis for a
CT forgery detector, evaluated under two zero-day protocols: Leave-One-
Domain-Out (LODO) and category-holdout (true zero-day).

## Pipeline
1. `01_data_extraction/` — DICOM loading, HU conversion, patch extraction
2. `02_patch_selection/` — fixed 10,000-patch authentic pool selection
3. `03_attack_generation/` — 11 tampering techniques across 6 categories
4. `04_experiments/` — LODO and category-holdout training scripts
5. `05_results/` — final results (JSON) and generated figures
6. `thesis/` — full written thesis document

## Dataset
LIDC-IDRI (Armato et al., 2011) — not included here due to size;
publicly available at https://www.cancerimagingarchive.net/collection/lidc-idri/
