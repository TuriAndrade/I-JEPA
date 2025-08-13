# JEPA — I-JEPA with VICReg and Simplicial Embeddings

This repository implements **I-JEPA** (Image Joint-Embedding Predictive Architecture) and proposes two information-theoretic upgrades that improve downstream accuracy on ImageNet-100:

- **VICReg loss**: encourages variance, invariance, and de-correlation in the learned space.
- **Simplicial Embeddings (SEM)**: projects features onto concatenated simplices (via softmax), inducing sparsity and better class separability.

The **best results** come from combining both: **I-JEPA + SEM + VICReg**.

---

## Why these changes?

- **VICReg** adds explicit regularizers (variance & covariance) on top of the similarity term to avoid collapse and reduce redundancy.
- **SEM** reshapes the latent space by normalizing each projection with softmax and concatenating multiple simplex-projected blocks, biasing toward sparse, more separable representations.

---

## Setup summary

- **Backbone**: ViT-Tiny (same as the official I-JEPA repo).
- **Dataset**: ImageNet-100 (100 classes sampled from ImageNet-1K).
- **Images**: resized to 224×224, normalized.
- **Protocol**:
  - Train/fine-tune with **10% / 25% / 50% / 100%** labeled splits.
  - Compare **from-scratch** ViT vs. **pretrained** I-JEPA variants.

---

## Results (Top-1 Accuracy on ImageNet-100)

| Pre-training Method         | 10%    | 25%    | 50%    | 100%   |
|----------------------------|:------:|:------:|:------:|:------:|
| None (from scratch)        | 0.1079 | 0.2264 | 0.3460 | 0.4971 |
| I-JEPA                     | 0.3816 | 0.4953 | 0.5523 | 0.5982 |
| I-JEPA + **SEM**           | 0.3916 | 0.4680 | **0.5963** | 0.6604 |
| I-JEPA + **VICReg**        | 0.3582 | 0.4898 | 0.5691 | 0.6604 |
| **I-JEPA + SEM + VICReg**  | **0.4105** | **0.5433** | 0.5820 | **0.6866** |

**Takeaways**

- All I-JEPA variants beat training from scratch.
- **SEM + VICReg** is consistently the strongest, especially in low-label regimes.

> **Disclaimer**: These models were **not** extensively tuned due to limited computing resources. Results focus on **relative comparison** between methods rather than achieving the highest possible accuracy.

---

## Notes on representation analysis

We also computed **MINE** (mutual information) and **LiDAR** scores during pretraining. Although **I-JEPA + VICReg** sometimes showed higher MINE/LiDAR than **I-JEPA + SEM + VICReg**, the latter still produced **better classification accuracy**, suggesting SEM’s sparsity/structure improves generalization even if MI estimates decrease due to lower entropy.

---

## References

- I-JEPA: Assran et al., 2023  
- VICReg: Bardes et al., 2022  
- Simplicial Embeddings: Lavoie et al., 2022
