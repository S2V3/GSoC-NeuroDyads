# <p align="center">GSoC-NeuroDyads</p>

<p align="center">
<b>ML4SCI · Google Summer of Code 2026</b><br>
Brain-to-Brain Decoder — Validating Neural Synchrony Patterns in Human Conversation<br><br>
<b>Vaidehi Shivram Hegde</b><br>
B.Tech Engineering Physics · IIT (ISM) Dhanbad · vaidehihegde05@gmail.com
</p>

---

## Project Overview

NeuroDyads studies **brain-to-brain synchrony** during naturalistic conversation using **hyperscanning EEG** — simultaneous EEG recordings from two participants (Listener and Speaker) across two emotional conditions: positive affect and negative affect. This repository contains my complete evaluation task submission for the ML4SCI NeuroDyads GSoC 2026 project.

The pipeline covers raw EEG preprocessing, CEBRA contrastive embedding, KNN decoding, and a three-experiment evaluation that exposes a temporal confound in naive analysis — with direct implications for how the 2026 project should be designed.

---

## Task 1 — EEG Preprocessing

### What Was Done

- Loaded raw EDF files for Listener and Speaker using MNE-Python
- Located DIN1 event markers programmatically (manual reading caused crop errors)
- Cropped into 4 segments: positive affect (marker 1 → marker 2), negative affect (marker 3 → end)
- Resolved a 0.22-second duration mismatch between Listener and Speaker negative segments
- Removed channel 65 (EEG VREF — vertex reference, not a brain source) → 64 channels per participant
- Renamed all 64 channels from generic EEG 1–64 to standard 10-20 system labels
- Applied 60 Hz notch filter (power line interference confirmed from raw PSD) + 1–40 Hz bandpass
- Ran ICA using Picard algorithm (n_components=64, random_state=42)
- Built custom matplotlib + scipy ICA inspection plots — MNE's built-in plotting crashed because the EDF files contained no digitization points
- Manually rejected artifact components: 15 (L+), 9 (L−), 28 (S+), 15 (S−)
- Speaker required more rejections due to jaw and face muscle activity during active speech production
- Verified ICA quality via PSD before/after comparison for all 4 segments — brain signal preserved, artifacts removed
- Saved 4 clean segments as .npy files

### Final Segment Shapes

| Segment | Shape (channels × timepoints) |
|---|---|
| Listener — Positive affect | (64, 36944) |
| Listener — Negative affect | (64, 38487) |
| Speaker — Positive affect | (64, 36944) |
| Speaker — Negative affect | (64, 38487) |

### ICA Quality Verification

<img src="https://github.com/S2V3/GSoC-NeuroDyads/blob/main/assets/PSD%20comparision_listener.png?raw=true" width="900" alt="PSD Before vs After ICA">

*PSD before (blue) and after (red) ICA for Listener segments. Lines are nearly identical at alpha (8–13 Hz) — brain signal preserved. Gaps only at delta (1–4 Hz) where eye-blink and slow drift artifacts were removed.*

---

## Task 2 — CEBRA Embedding

### Notebook Resources

| Resource | Description | Link |
|---|---|---|
| **Main Notebook** | Full pipeline — preprocessing + CEBRA + Parts 3 & 4 | [gsoc-neurodyads-without-split.ipynb](https://github.com/S2V3/GSoC-NeuroDyads/blob/main/gsoc-neurodyads-without-split.ipynb) |
| **Random Split Notebook** | Stratified 80/20 split — proper train/test evaluation | [gsoc-neurodyads-random-split.ipynb](https://github.com/S2V3/GSoC-NeuroDyads/blob/main/gsoc-neurodyads-random-split.ipynb) |
| **Chronological Split Notebook** | Chronological 60/40 split — temporal confound proof | [gsoc-neurodyads-chronological-split.ipynb](https://github.com/S2V3/GSoC-NeuroDyads/blob/main/gsoc-neurodyads-chronological-split.ipynb) |

### CEBRA Model Configuration

```python
model = cebra.CEBRA(
    model_architecture='offset10-model',
    batch_size=512,
    learning_rate=3e-4,
    temperature=1.0,
    output_dimension=3,
    max_iterations=1000,
    distance='cosine',
    conditional='time_delta',
    device='cpu',
    time_offsets=10
)
```

### Main Notebook Results

| Metric | Main Model | Shuffled Control |
|---|---|---|
| KNN Decoding Accuracy (5-fold CV) | **99.53%** | 50.00% |
| Final Training Loss (GoF) | 5.6834 | 6.2390 |

### 3D Embedding Visualization

<img src="https://github.com/S2V3/GSoC-NeuroDyads/blob/main/assets/CEBRA%203D%20Embedding%20plot%20for%20without%20split.png?raw=true" width="900" alt="CEBRA 3D Embeddings">

*Main model (left): two cleanly separated clusters on the unit sphere surface — positive affect (blue) and negative affect (red). Shuffled control (right): structureless blob at 50.00% accuracy, confirming the main model's structure is label-driven, not a geometric artifact.*

---

## Three-Experiment Evaluation

The most important finding of this submission is not the 99.53% — it is what happens when that number is stress-tested across three evaluation designs.

| Evaluation Method | KNN Accuracy | What It Means |
|---|---|---|
| Transductive (main notebook) | 99.53% | CEBRA trained and evaluated on all data |
| Random stratified split | 50.79% | Proper unseen test set — collapses to chance |
| Chronological split | 100.00% | Test set is 100% label 1 — trivially correct |

**One conclusion across all three:** CEBRA learned temporal position, not emotional brain states.

Positive affect always occupies timepoints 1–36,944 and negative affect occupies 36,945–75,431. Condition label and recording time are perfectly correlated. CEBRA's time-delta contrastive objective makes this especially problematic — it constructs positive pairs from timepoints that are close in time, which in this setup almost always means same-label. Remove time as a cue via random stratified splitting and accuracy collapses to 50.79%. Make the temporal boundary explicit via chronological splitting and the test set becomes 100% negative affect — trivially predicted every time.

This confound and its fix — interleaved recording blocks and epoch-level stratified cross-validation — are the core methodological improvements planned for the 2026 project.

---

## Repository Structure

```
GSoC-NeuroDyads/
│
├── gsoc-neurodyads-without-split.ipynb        # Main notebook — full pipeline + Parts 3 & 4
├── gsoc-neurodyads-random-split.ipynb         # Stratified 80/20 split evaluation
├── gsoc-neurodyads-chronological-split.ipynb  # Chronological 60/40 split evaluation
├── assets/
│   ├── CEBRA 3D Embedding plot for without split.png
│   └── PSD comparision_listener.png
└── README.md
```

---

## Key Numbers at a Glance

```
Data shapes:
  L_positive: (64, 36944)    S_positive: (64, 36944)
  L_negative: (64, 38487)    S_negative: (64, 38487)
  Full matrix: (75431, 128)
  Labels: 36944 zeros (positive) + 38487 ones (negative)

Results:
  Transductive KNN:        99.53% main / 50.00% shuffled
  Random split KNN:        50.79% main / 49.92% shuffled
  Chronological split KNN: 100.00% main / 100.00% shuffled

CEBRA: offset10-model · output_dim=3 · cosine · time_delta · 1000 iterations
```

---

## References

- Schneider et al. (2023). Learnable latent embeddings for joint behavioural and neural analysis. *Nature.*
- Hasson et al. (2012). Brain-to-brain coupling: a mechanism for creating and sharing a social world. *Trends in Cognitive Sciences.*
- Roca et al. (2023). Cross-entropy metrics for evaluating CEBRA embedding distributions.
- Crompton et al. (2025). Neurotype-dependent communication breakdowns. *Nature Human Behaviour.*

---

## About

| | |
|---|---|
| **Organization** | ML4SCI (Machine Learning for Science) |
| **Project** | NeuroDyads — Brain-to-Brain Decoder |
| **Mentors** | Dr. Evie Malaia (University of Alabama) · Dr. Brendan Ames (University of Southampton) |
| **Applicant** | Vaidehi Shivram Hegde |
| **Institution** | IIT (ISM) Dhanbad |
| **Contact** | ml4-sci@cern.ch |
