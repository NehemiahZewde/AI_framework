# Clinical ML for Patient Enrichment and Stratification

A patient-level machine learning framework for translating clinical-trial data into enrichment, stratification, and prospective screening strategies.

This framework is designed for the reality of clinical-trial data: datasets may be small, heterogeneous, noisy, and expensive to collect, but teams still need to make decisions about patient selection, subgroup analysis, and future study design.

<img width="1568" height="814" alt="image" src="https://github.com/user-attachments/assets/ba2dc360-c5d4-49ca-a84a-feb4c2a20b50" />
<p align="center"><em>Modality- and disease-area-agnostic workflow for translating clinical data into patient enrichment and stratification strategies.</em></p>

---

## The Challenge
Clinical-trial datasets often contain valuable patient-level signal, but translating that signal into trial-design decisions is difficult.

Key challenges include:

- Clinical outcomes can vary substantially across patients, making response, progression, or diagnostic status difficult to predict.
- Patient-level signals may exist, but limited sample size can make it hard to know whether a subgroup pattern is reliable.
- A model may rank patients well, but the practical value depends on whether the subgroup is enriched, recruitable, and efficient to screen.

This creates practical questions:

- Can patient features identify individuals or subgroups more likely to respond, progress, or experience the target outcome?
- Can patients be grouped into clinically meaningful low-, medium-, and high-likelihood strata?
- Is the signal strong and stable enough to support enrichment, stratification, or future validation?

---

## The Solution
This framework helps clinical teams evaluate whether patient-level features can support more efficient clinical-trial design.

It uses machine learning to generate patient-level risk or response scores, then translates those scores into practical enrichment and stratification strategies.

The framework helps teams:

- identify higher-yield patient subgroups
- define probability-based patient strata
- estimate screening burden, sample size, and power
- evaluate whether a patient-selection rule is stable enough for validation
- translate retrospective signals into future trial-planning decisions

In practice, this helps answer whether enrichment or stratification is feasible, which subgroup or stratum to prioritize, and whether the signal is strong enough to justify further validation.

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

## Requirements

- Python 3.9+

---

## Installation

### Option 1: Install directly from GitHub

```bash
pip install git+https://github.com/NehemiahZewde/AI_framework.git
### Option 2: Clone and install
```bash
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
