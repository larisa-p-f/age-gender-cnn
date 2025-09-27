# Age Estimation and Gender Classification with CNNs

This project applies **Convolutional Neural Networks (CNNs)** to predict **age** (regression) and **gender** (binary classification) from face images.  
The models were trained and validated on **5,000 cropped face images (128×128)** from the **UTKFace dataset**.

---

## Project Overview

You will find two models implemented in the notebook:

- **Model A**: Custom CNN built from scratch, following architectural constraints (multi-output network with shared convolutional layers and two branches).  
- **Model B**: Transfer learning using **VGG16**, fine-tuned on the dataset for age and gender prediction.

Both models take images of size **128×128×3** directly (no resizing), as required.

---

## Dataset

- Dataset source: UTKFace (cropped & aligned faces, 128×128 resolution).  
- Training & validation split: provided in `train_val/` (not included in this repo).  
- Test set: unseen images for final evaluation.  

---

## Key Requirements (Assignment Constraints)

For **Model A**:
1. Input size must be **128×128×3** (no resizing allowed).  
2. Gender branch = **1 output unit with sigmoid** (not 2).  
3. Feature map entering the first fully connected layer must be **< 10×10**.  
4. Include overfitting prevention (BatchNorm, Dropout, EarlyStopping, etc.).  

For **Model B**:
- Use a **pre-trained CNN** (e.g., VGG16 on ImageNet).  
- Fine-tune on the dataset, keeping the same input size and 1-unit gender output branch.  

---

## Results

| Model   | Age MAE (Validation) | Gender Accuracy (Validation) | Notes |
|---------|-----------------------|------------------------------|-------|
| Model A (Custom CNN) | ~6.4 years | ~88% | Faster, smoother convergence |
| Model B (VGG16 Fine-tuned) | ~6.6 years | ~88% | Slightly slower, more fluctuations |

- Both models generalised well with minimal overfitting.  
- Model A trained more efficiently (~25 min vs ~30 min).

The full coursework report, including methodology, training details, and analysis, can be found in [report.pdf](./report.pdf).

---

## Features & Techniques

- **Multi-output CNN**: shared backbone with regression + classification heads  
- **Data Augmentation**: horizontal flips, small rotations  
- **Training strategies**:
  - Optimiser: Adam  
  - EarlyStopping (patience=20)  
  - ReduceLROnPlateau (patience=3)  
- **Transfer Learning**: fine-tuned VGG16 with frozen/unfrozen layers  

---

## Ethical Considerations
Age and gender prediction has potential applications (smartphones, security, advertising) but also raises concerns around privacy and bias.
Models trained on limited datasets (like UTKFace) may not generalise fairly across different ethnicities, age groups, or real-world conditions.

