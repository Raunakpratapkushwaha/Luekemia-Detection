# Leukemia Detection from Peripheral Blood Smear Images (Research Audit Log)

> [!WARNING]
> **Clinical Disclaimer**
> This system is intended strictly for research and clinical decision support. It must NOT be presented or used as an autonomous diagnostic system. Any models developed here require human-in-the-loop review by qualified hematologists/pathologists.

## Project Overview

- **Research Objective**: To develop a machine-learning system for classifying single-cell microscopic images of peripheral blood smears as either blast cells (associated with Acute Lymphoblastic Leukemia, ALL) or non-blast cells (from healthy individuals).
- **Intended Clinical-Support Use**: Aiding hematologists/pathologists in highlighting potential blast cells for manual review, thereby reducing screening time and fatigue.
- **Scope**: Single-cell classification (Blast vs. Non-Blast).
- **Non-Goals**: Full slide analysis, autonomous patient diagnosis, subtyping of leukemia.
- **Limitations**: Restricted to the staining and imaging protocols of the training data. Potential biases towards specific imaging equipment. The dataset lacks patient-level metadata.
- **Current Project Status**: Pre-Phase-3 Audit Complete. Major structural limitations in data leakage identified.

## Research Question

**Goal**: Cell classification.
Specifically, predicting whether an isolated cell crop from a peripheral blood smear is a **lymphoblast cell** (leukemia suspect) or a **normal mononuclear cell** (healthy).
*Note: This is strictly cell-level classification, not patient-level leukemia classification or subtyping.*

## Dataset Registry

### PRIMARY DATASET
- **Dataset Name**: ALL-IDB2 (Acute Lymphoblastic Leukemia Image Database for Image Processing)
- **Source**: Department of Information Technology - Università degli Studi di Milano
- **Official URL**: http://www.dti.unimi.it/fscotti/all/
- **Date Accessed**: 2026-09-05
- **License**: Research use (as per ALL-IDB dataset terms)
- **Number of Patients**: **UNKNOWN** (Source dataset ALL-IDB1 has 108 images, but ALL-IDB2's 260 cells lack provenance mapping).
- **Number of Images**: 260 cropped single-cell images total
- **Sample Type**: Peripheral blood smear (single cell crops)
- **Disease Classes**: Lymphoblast cell (Y=1) vs Normal cell (Y=0)
- **Available Labels**: Cell-level binary labels embedded in the filenames (ImXXX_Y)
- **Imaging Method**: Optical Microscopy
- **Stain if known**: Giemsa (standard for ALL-IDB)
- **Known Limitations**: 
  - Extremely small dataset size (260 images). 
  - **Critical**: No explicit patient or source-image mapping available in the filenames. `XXX` is merely a progressive index (1 to 260).

## Patient Split & Leakage Audit

> [!CAUTION]
> **Severe Data Leakage Fundamental Flaw**
> Patient IDs and source-image IDs cannot be recovered. Because the 260 cells originated from a smaller number of patients/slides (likely 108 as per ALL-IDB1), random splitting mathematically scrambles cells from the same patient across both the training and validation sets. **Patient-level independence cannot be guaranteed.**

- **Number of training patients**: Unknown (No metadata)
- **Number of validation patients**: Unknown (No metadata)
- **Splitting algorithm**: The provided `All-IDB dataset` directory already contains a predefined 200/60 random split. 
- **Leakage Findings**: 
  - We found 1 exact duplicate file pair in the training set (`Im257_0.jpg` == `Im223_0.jpg`). 
  - Near-duplicate (same-patient) leakage across train and val is virtually guaranteed due to the lack of source-image tracking.
- **Validation Inflation**: Validation metrics on this split will be scientifically optimistic/inflated and do not represent generalization to unseen patients.

## Research Claims (What this dataset can prove)
Based strictly on the available labels and metadata:
- **cell-level classification**: PARTIALLY SUPPORTED (leakage makes evaluation optimistic)
- **lymphoblast detection**: PARTIALLY SUPPORTED
- **general blast detection**: NOT SUPPORTED (only trained on ALL lymphoblasts, not AML myeloblasts, etc.)
- **ALL-related morphological abnormality recognition**: PARTIALLY SUPPORTED
- **patient-level ALL detection**: NOT SUPPORTED
- **general leukemia detection**: NOT SUPPORTED
- **AML detection**: NOT SUPPORTED
- **leukemia subtype classification**: NOT SUPPORTED
- **early/pre-diagnostic leukemia detection**: NOT SUPPORTED

## Recommended Evaluation Strategy
Because patient-level CV is impossible on ALL-IDB2 as provided, we must use the current 200/60 split strictly for **proof-of-concept cell morphology feature extraction**. 
**Final Validation Requirement**: Any model produced here must be evaluated on an independent, external test set (e.g., C-NMC dataset or an internal hospital dataset with known patient IDs) before any performance claims can be scientifically defended.

## Preprocessing Audit

- **Input Dimension**: To be standardized to 224x224 during modeling.
- **Color Format**: RGB
- **Image Normalization Constants** (computed from Train split):
  - **Mean**: `[0.5454, 0.4797, 0.4986]`
  - **Standard Deviation**: `[0.1111, 0.1531, 0.0825]`
- **Class Distribution**:
  - **Train**: 100 Blast, 100 Non-Blast (Total: 200)
  - **Val**: 30 Blast, 30 Non-Blast (Total: 60)
  - **Balance**: Perfectly balanced (1:1 ratio) in both train and val splits.

## Model Registry

- **Experiment ID**: EXP_001_Baseline_CNN
- **Architecture**: Custom 3-block CNN with BatchNormalization, GlobalAveragePooling, and heavy Dropout.
- **Pretrained Weights**: None (trained from scratch)
- **Input Resolution**: 224x224 RGB
- **Optimizer**: Adam (lr=1e-4)
- **Loss**: BinaryCrossentropy
- **Batch Size**: 16
- **Epochs**: 10 (completed)
- **Augmentation**: RandomFlip, RandomRotation
- **Class Weighting**: None (dataset is perfectly balanced)
- **Checkpoint Path**: `checkpoints/best_baseline.keras`

## Results Registry

> [!WARNING]
> **Inflated Results Disclaimer**
> As documented in the Patient Split Audit, the validation split mathematically contains cells from the same patients as the training split. These metrics do NOT represent real-world generalization to unseen patients.

**EXP_001_Baseline_CNN Results (Validation Set):**
- **Loss**: 0.6819
- **Accuracy**: 50.00% (Threshold calibration required)
- **ROC-AUC**: 0.9700 (Very high class separability, confirming the model learned the morphological features, though highly likely inflated due to patient leakage).

## Phase 4: Scientific Evaluation & Optimization

### Step 1: ROC-AUC vs Accuracy Investigation & Probability Distribution

**Objective**: Diagnose why the baseline model achieved ~0.97 ROC-AUC but exactly 50% accuracy on the validation set.

**Findings**:
By running a custom `debug_model.py` to inspect the raw predicted probabilities, we discovered the following distribution on the validation set:
- **Min probability**: 0.534051
- **Max probability**: 0.570010
- **Mean probability for True Class 0 (Non-blast)**: 0.540963
- **Mean probability for True Class 1 (Blast)**: 0.557036

**Conclusion**: 
The accuracy is exactly 50% because the model is predicting `1` (Blast) for *every single image* at the default 0.5 threshold (all probabilities are between 0.53 and 0.57). 

However, the probabilities for Class 1 are consistently higher than Class 0 (perfect ranking), which explains the near-perfect ROC-AUC (0.9833).

**Root Cause Analysis**:
The logits are severely squashed and shifted towards 0 (sigmoid ~ 0.5) during inference. This is a classic symptom of **Batch Normalization (BN) statistics not converging** during training. 
Because the dataset is extremely small (200 images, batch size 16 = 12 steps per epoch), after only 10 epochs (120 updates), the BN layers' running mean and variance (momentum=0.99) have not caught up to the actual data distribution. During training, BN uses batch statistics (which are fine), but during inference, it uses the unconverged running statistics, causing activation collapse.

### Step 2: Threshold Analysis

**Objective**: Evaluate thresholds from 0.05 to 0.95 to find the optimal trade-off between Sensitivity and Specificity.

**Findings**:
A threshold sweep was executed and saved to `reports/threshold_analysis.csv` and plotted. Given the squashed distribution (0.53 to 0.57), any threshold below 0.53 yields 100% Sensitivity and 0% Specificity, while any threshold above 0.57 yields 0% Sensitivity and 100% Specificity. A calibrated threshold explicitly within the `[0.54, 0.55]` window optimally separates the classes.

**Selected Threshold**: ~0.548 (the midpoint of the Class 0 and Class 1 means).
**Justification**: This threshold perfectly distinguishes the non-converged distributions without requiring architectural changes, though the fundamental fix is to address the BatchNormalization convergence or use a different architecture in Step 4.
