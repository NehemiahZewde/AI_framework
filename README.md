# Clinical ML for Patient Enrichment and Stratification

A patient-level machine learning framework for improving clinical-trial design through enrichment, stratification, and prospective screening.

This framework helps clinical and translational teams evaluate whether patient-level data can be used to identify higher-yield subgroups, define clinically meaningful strata, and support more efficient study planning across disease areas.

It is designed for the reality of clinical-trial data: datasets may be small, heterogeneous, noisy, and expensive to collect, but teams still need to make high-impact decisions about enrollment, subgroup analysis, validation, and future trial design.

<img width="1568" height="814" alt="image" src="https://github.com/user-attachments/assets/ba2dc360-c5d4-49ca-a84a-feb4c2a20b50" />
<p align="center"><em>Modality- and disease-area-agnostic workflow for translating clinical data into patient enrichment and stratification strategies.</em></p>


## The Challenge

Clinical trials are expensive, time-consuming, and often affected by heterogeneous patient response. When patients vary widely in disease biology, progression, or likelihood of response, broad enrollment can dilute treatment effects, increase sample-size requirements, and make trial results harder to interpret.

At the same time, clinical-trial datasets are often small, noisy, and difficult to reuse for future study design. Even when patient-level features contain useful signal, it can be difficult to determine whether that signal is stable enough and actionable enough to support patient selection.

The practical challenge is not only whether a model can predict an outcome, but whether patient-level predictions can help identify enriched subgroups, define meaningful patient strata, reduce screening inefficiency, and support more focused clinical-trial decisions.


## The Solution

This framework helps clinical teams translate patient-level data into enrichment and stratification strategies that can support more efficient clinical-trial design.

It uses machine learning to generate patient-level risk, diagnostic, or response scores, then converts those scores into trial-planning outputs that help teams:

- identify higher-yield patient subgroups for focused enrollment
- define probability-based strata for subgroup analysis and trial planning
- estimate screening burden, sample size, power, and validation feasibility

In practice, this helps teams assess whether enrichment or stratification could improve trial efficiency by focusing enrollment, increasing the expected response or event rate, reducing uninformative screening or enrollment, and supporting faster go/no-go decisions.

## Patient Selection Framework

The same patient-level score can support both **enrichment** and **stratification**.

- **Enrichment** applies a fixed threshold to select a subgroup.
- **Stratification** divides patients into ordered probability groups, such as low, medium, and high likelihood groups.

| Patient selection framework | Core question | Score used | Enrichment use | Stratification use | Status |
|---|---|---|---|---|---|
| Diagnostic | Who likely has the disease or diagnostic target? | `P(disease given patient features)` | Select patients above a diagnostic-likelihood threshold | Group patients into diagnostic-likelihood strata | Supported |
| Prognostic | Who likely experiences the future outcome of interest? | `P(outcome given patient features)` | Select patients above an outcome-probability threshold | Group patients into outcome-likelihood strata | Supported |
| Treatment-benefit / treatment-effect | Who likely benefits more from treatment than control or no treatment? | Estimated treatment benefit | Select patients above a treatment-benefit threshold | Group patients into treatment-benefit strata | Future work |

## What the Framework Produces

Typical outputs include:

- calibrated patient-level scores and ranked screening views
- enrichment and stratification summaries, including subgroup rates and enrichment factors
- screening burden, number needed to screen, sample-size, and power estimates
- patient-level uncertainty and stability summaries

---

## Use Cases

This framework is designed for clinical and translational teams working with retrospective or early-stage clinical-trial data.

Example use cases include:

- enrichment feasibility analysis for diagnostic or prognostic endpoints
- patient stratification for subgroup analysis and trial planning
- biomarker or feature-panel evaluation
- screening, recruitment, and prospective validation planning

## Technical Foundation

The framework is built around clinical-trial data realities: limited sample size, heterogeneous patients, repeated measurements, patient-level grouping, calibration needs, and validation leakage risk.

Key technical features include:

- patient-level and group-aware validation
- nested cross-validation support
- calibrated probability estimation
- repeated prediction pooling to patient-level summaries
- diagnostic and prognostic enrichment analysis
- probability-based patient stratification
- sample-size and power calculations
- modality-agnostic inputs, including tabular, biomarker, time-series, imaging-derived, and multimodal features


## Requirements

- Python 3.9+


## Installation

### Option 1: Install directly from GitHub

```bash
pip install git+https://github.com/NehemiahZewde/AI_framework.git
````
### Option 2: Clone and install
```bash
git clone https://github.com/NehemiahZewde/AI_framework.git
cd AI_framework
pip install -e .
````

## Project layout

* `ai_framework/` — core pipeline building blocks (data prep, training, evaluation, calibration, reporting)
* `tutorial/` — end-to-end examples

---
