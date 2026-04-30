# Clinical ML for Patient Enrichment and Stratification

A patient-level machine learning framework for improving clinical-trial design through enrichment, stratification, and prospective screening.

This framework helps clinical and translational teams evaluate whether patient-level data can be used to identify higher-yield subgroups, define clinically meaningful strata, and support more efficient study planning across disease areas.

It is designed for the reality of clinical-trial data: datasets may be small, heterogeneous, noisy, and expensive to collect, but teams still need to make high-impact decisions about enrollment, subgroup analysis, validation, and future trial design.

<img width="1568" height="814" alt="image" src="https://github.com/user-attachments/assets/ba2dc360-c5d4-49ca-a84a-feb4c2a20b50" />
<p align="center"><em>Modality- and disease-area-agnostic workflow for translating clinical data into patient enrichment and stratification strategies.</em></p>

---

## The Challenge

Clinical trials are expensive, time-consuming, and often affected by heterogeneous patient response. When patients vary widely in disease biology, progression, or likelihood of response, broad enrollment can dilute treatment effects, increase sample-size requirements, and make trial results harder to interpret.

Clinical-trial datasets often contain useful patient-level signal, but translating that signal into trial-design decisions is difficult.

Key challenges include:

- Clinical outcomes can vary substantially across patients, making response, progression, or diagnostic status difficult to predict.
- Patient-level signals may exist, but limited sample size can make it hard to know whether a subgroup pattern is reliable.
- A model may rank patients well, but the practical value depends on whether the subgroup is enriched, recruitable, and efficient to screen.

This creates practical questions:

- Can patient features identify individuals or subgroups more likely to respond, progress, or experience the target outcome?
- Can patients be grouped into clinically meaningful low-, medium-, and high-likelihood strata?
- Can enrichment or stratification make a future trial more focused, efficient, and decision-ready?

---

## The Solution

This framework helps clinical teams translate patient-level data into enrichment and stratification strategies that can support more efficient clinical-trial design.

It uses machine learning to generate patient-level risk, diagnostic, or response scores, then converts those scores into decision-oriented trial-planning outputs.

The framework helps teams:

- identify higher-yield patient subgroups for focused enrollment
- define probability-based strata for subgroup analysis and trial planning
- estimate screening burden, sample size, and power
- evaluate whether a patient-selection rule is stable enough for validation
- translate retrospective signals into prospective study-design decisions

In practice, this helps teams assess whether enrichment or stratification could improve trial efficiency by focusing enrollment, increasing the expected response or event rate, reducing uninformative screening or enrollment, and supporting faster go/no-go decisions.

---

## Patient Selection Framework

The same patient-level score can support both **enrichment** and **stratification**.

- **Enrichment** applies a fixed threshold to select a subgroup.
- **Stratification** divides patients into ordered probability groups, such as low, medium, and high likelihood groups.

| Patient selection framework | Core question | Score used | Enrichment use | Stratification use | Status |
|---|---|---|---|---|---|
| Diagnostic | Who likely has the disease or diagnostic target? | `P(disease given patient features)` | Select patients above a diagnostic-likelihood threshold | Group patients into diagnostic-likelihood strata | Supported |
| Prognostic | Who likely experiences the future outcome of interest? | `P(outcome given patient features)` | Select patients above an outcome-probability threshold | Group patients into outcome-likelihood strata | Supported |
| Treatment-benefit / treatment-effect | Who likely benefits more from treatment than control or no treatment? | Estimated treatment benefit | Select patients above a treatment-benefit threshold | Group patients into treatment-benefit strata | Future work |

---

## What the Framework Produces

Typical outputs include:

- calibrated patient-level risk, diagnostic, or response scores
- ranked patient screening views
- enrichment threshold summaries
- low / medium / high probability strata
- subgroup response, event, or diagnostic-positive rates
- enrichment factors relative to the full study population
- screening burden and number needed to screen
- sample-size and power estimates for validation
- patient-level uncertainty and stability summaries

---

## Use Cases

This framework is designed for clinical and translational teams working with retrospective or early-stage clinical-trial data.

Example use cases include:

- diagnostic case-enrichment analysis
- prognostic response-enrichment assessment
- patient stratification for subgroup analysis
- biomarker or feature-panel evaluation
- screening and recruitment planning
- prospective validation planning

---

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

---

## Requirements

- Python 3.9+

---

## Installation

### Option 1: Install directly from GitHub

```bash
pip install git+https://github.com/NehemiahZewde/AI_framework.git

### Option 2: Clone and install
git clone https://github.com/NehemiahZewde/AI_framework.git
cd AI_framework
pip install -e .
````

---

## Quickstart

> Coming next: a minimal end-to-end example (data → train → report) runnable in a few commands.

---

## Project layout

* `ai_framework/` — core pipeline building blocks (data prep, training, evaluation, calibration, reporting)
* `tutorial/` — end-to-end examples

---
