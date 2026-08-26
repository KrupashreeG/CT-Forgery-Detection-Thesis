# Forensic Detection of Pathological Forgeries in Clinical CT Imaging

A cybersecurity threat modelling and generalisation gap analysis for a
CT forgery detector, evaluated under two zero-day protocols.

## The Problem

Forensic detectors trained to catch tampered CT scans are almost always
evaluated only against the exact manipulation technique they were
trained on. This leaves a real gap: what happens when such a detector
meets a manipulation type it has never encountered? This project
investigates that question directly, and asks a more specific one too —
does generalisation failure come from not having *enough* training
data, or from not having *varied enough* training data?

## System Architecture

The full pipeline, from raw CT scans to a measured generalisation gap:
![System Architecture](images/architecture_diagram.png)


## Dataset

All authentic imagery comes from **LIDC-IDRI** (Armato et al., 2011),
a public, de-identified thoracic CT archive spanning 1,081 patients
across multiple scanners and institutions. Full scans were converted to
Hounsfield Units, filtered for lung-tissue content and bilateral
visibility, extracted as 512×512 patches, quality-filtered, and
downsampled to 128×128 pixels — yielding 356,614 candidate patches. A
fixed, seeded subset of 10,000 was selected and used as the authentic
class throughout every experiment.

*Dataset not included in this repo due to size publicly available at
[cancerimagingarchive.net/collection/lidc-idri](https://www.cancerimagingarchive.net/collection/lidc-idri/)*

## Attack Framework

Eleven manipulation techniques were built, grouped into six categories
by underlying mechanism rather than surface appearance:

| Category | Techniques |
|---|---|
| Local pixel injection | Gaussian, elliptical, contrast, smoothing, sharpening, texture |
| Spatial rearrangement | Copy-move |
| Geometric transform | Rotation |
| Gradient-domain blending | Poisson |
| Frequency-domain shift | Frequency (DCT-based) |
| Learned/generative | CycleGAN |

Each technique was applied to the same fixed 10,000-patch authentic
pool, producing ~10,000 tampered patches per technique.

## Detector

A fine-tuned **EfficientNet-B3**, pretrained on ImageNet, with a custom
classification head (dropout → linear(1536→256) → ReLU → dropout →
linear(256→1) → sigmoid). Trained with Adam, cosine annealing over 40
epochs, early stopping (patience 5), batch size 32, seed 42 throughout.

## Evaluation Protocols

Two deliberately different tests were run, with **equal total training
volume (30,000 patches) in both**, so results couldn't be explained by
data quantity alone:

- **LODO (Leave-One-Domain-Out)** — trained on 10 of 11 attacks, tested
  zero-day on the one left out, repeated across all 11
- **Category-holdout** — trained on only 1 category (6 attacks), tested
  zero-day on the other 5 entirely unrelated categories

## Key Finding

Generalisation failure is **not uniform** — it falls into three
distinct patterns:

| Pattern | Categories | Behaviour |
|---|---|---|
| Intrinsically hard | Frequency-domain | Fails badly regardless of training breadth |
| Exposure-sensitive | Geometric, Spatial | Fails badly only when training is narrow |
| Exposure-robust | Learned/generative, Gradient-blending | Stable either way |

Mean generalisation gap rose from **0.0704** (partial exposure) to
**0.2039** (true zero-day)  roughly **3x worse**, despite identical
total training data in both conditions.

## Clinical Significance

False negative rate (share of tampered scans missed) ranged from
**17.6% to 99.3%** depending on attack category showing that strong
accuracy on familiar attacks says very little about performance on
unfamiliar ones.

## Security Context

A generic PACS pipeline was modelled and analysed using the **STRIDE**
threat framework, identifying **Tampering** as the category
structurally identical to the attack studied here, with two concrete
injection points: the scanner-to-gateway network boundary, and the
PACS server itself.

## Repository Structure

01_data_extraction/ DICOM loading, HU conversion, patch extraction
02_patch_selection/ Fixed 10,000-patch authentic pool + manifest
03_attack_generation/ 11 tampering technique scripts
04_experiments/ LODO and category-holdout training scripts
05_results/ Final results (JSON), generated figures, tables


## Author

Krupashree Govindu MSc Cybersecurity, Technological University Dublin
Supervisor: Dr. Jin Xu
