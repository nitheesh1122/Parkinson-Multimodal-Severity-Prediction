# Research Gap

## Identified Gaps From Existing Studies

- Most models are limited to binary detection (PD vs non-PD) rather than decision support.
- Explainability is often restricted to feature importance plots and not translated to clinical summaries.
- Predictions are rarely accompanied by confidence-oriented interpretation for practitioners.
- Existing pipelines do not consistently generate structured reports combining risk score, key biomarkers, and interpretation.
- Clinical usability is limited because outputs are model-centric rather than clinician-centric.

## Gap Addressed in This Project

This project targets an explainable decision support layer that connects model prediction with a structured report containing:

- prediction outcome and confidence,
- top contributing speech biomarkers,
- SHAP-based interpretation,
- concise clinical recommendation notes.
