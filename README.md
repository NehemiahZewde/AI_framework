# Clinical ML for Patient Enrichment and Stratification

A patient-level machine learning framework for translating clinical-trial data into enrichment, stratification, and prospective screening strategies.

<img width="1568" height="814" alt="image" src="https://github.com/user-attachments/assets/ba2dc360-c5d4-49ca-a84a-feb4c2a20b50" />
<p align="center"><em>Modality- and disease-area-agnostic workflow for translating clinical data into patient enrichment and stratification strategies.</em></p>

## The Challenge

Clinical-trial datasets are often small, heterogeneous, noisy, and expensive to collect. Even when useful signals exist, clinical development teams still need to decide whether those signals can support patient selection, subgroup analysis, enrichment, stratification, or future trial design.

Common challenges include:

- Patient response and disease progression can be heterogeneous across a trial population.
- Clinical or patient-level features may contain signal, but the signal may be weak, unstable, or difficult to translate into action.
- Small sample sizes can make it difficult to know whether an observed subgroup pattern is reliable.
- A model may rank patients well, but the practical value depends on whether the selected subgroup is large enough and sufficiently enriched.
- Trial teams need to understand screening burden, sample-size implications, and the potential cost-efficiency of a patient-selection strategy.
- A patient-selection rule must be interpretable enough to support prospective validation or future study planning.

Key questions include:

- Can patient features identify individuals more likely to respond, progress, or experience the target outcome?
- Is there a subgroup that is sufficiently enriched to justify prospective validation?
- Can patients be stratified into clinically meaningful low-, medium-, and high-likelihood groups?
- How many patients would need to be screened to enroll the target subgroup?
- Could enrichment or stratification improve trial efficiency by reducing unnecessary enrollment, focusing resources, or increasing the expected event/response rate?
- Is the signal stable enough to support future study planning?

## The Solution

This framework helps clinical teams evaluate whether patient-level features can support more efficient clinical-trial design.

It uses machine learning to generate patient-level risk or response scores, then translates those scores into practical enrichment and stratification strategies.

The goal is to help answer:

**Can we use patient-level predictions to enroll a more informative study population, reduce screening inefficiency, and make better trial-planning decisions?**

The framework supports this by:

- identifying higher-yield patient subgroups for enrichment
- defining clinically meaningful probability strata
- estimating screening burden, sample size, and power
- evaluating whether a patient-selection rule is stable enough for validation
- translating retrospective signals into future trial-planning decisions

In practice, this helps teams decide whether enrichment or stratification is feasible, which subgroup or stratum to prioritize, and whether the signal is strong enough to justify further validation.

## Requirements
- Python 3.9+

---

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

---

## Quickstart

> Coming next: a minimal end-to-end example (data → train → report) runnable in a few commands.

---

## Project layout

* `ai_framework/` — core pipeline building blocks (data prep, training, evaluation, calibration, reporting)
* `tutorial/` — end-to-end examples

---
