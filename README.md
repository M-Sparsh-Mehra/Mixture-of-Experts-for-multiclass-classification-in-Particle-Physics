# Mixture of Experts for Multiclass Classification in Particle Physics

A dynamic two-stage Mixture-of-Experts (MoE) inference pipeline for High-Energy Physics (LHC) event classification.

---

## Pipeline Overview




```mermaid
flowchart TD
    A[Raw Data] --> B[Stage I: FROCC Sorter]
    B --> C[Event Filtering & Routing]
    C --> D[Stage II: Expert Pool]
    D --> E[Aggregator]
    E --> F[Final Class Probabilities]
```


