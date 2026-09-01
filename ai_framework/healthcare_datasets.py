"""
Clinical healthcare dataset loading utilities.

This module centralizes access to curated clinical tabular datasets and returns
all datasets in a standardized format for downstream machine-learning workflows.

Supported dataset sources include:
- sklearn-native datasets
- OpenML datasets
- UCI datasets accessed with ucimlrepo
- local clinical CSV datasets, including MS INNI modeling views
- local pharma-focused CSV and Excel datasets

Public API:
- get_healthcare_dataset_catalog(): view available datasets and metadata
- load_healthcare_dataset(): load a selected dataset by dataset_name

All loaders return a dictionary with a consistent structure:
- df: full modeling DataFrame, including the target column
- X: feature matrix
- y: target variable
- feature_names: input feature names
- metadata: clinical and modeling metadata
- raw_data: original source object, raw source DataFrame, or source workbook view

Patient-selection metadata
--------------------------
The catalog separates two ideas that are often combined in informal language:

1. clinical_score_type: what the model score represents clinically.
   Active values include diagnostic likelihood, prognostic risk, recurrence
   risk, and treatment response.

2. selection_use: how the score is used operationally. Current workflows use
   the score for enrichment or stratification.

Treatment response is represented as prediction of response after a specified
therapy, for example P(response after treatment | baseline features). This is
not the same as treatment-effect or causal benefit estimation, which would
require comparing outcomes under treatment versus control, no treatment, or an
alternative treatment strategy.
"""


# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:  # pragma: no cover - only used when optional dependency is absent.
    fetch_ucirepo = None


# =============================================================================
# Type aliases
# =============================================================================

# task_type combines the clinical score interpretation with the operational
# patient-selection use. Only task types currently represented by the catalog
# are listed here. Treatment-effect task types are intentionally not included
# because the current datasets do not estimate causal treatment benefit.
TaskType = Literal[
    "diagnostic_enrichment",
    "prognostic_enrichment",
    "diagnostic_stratification",
    "prognostic_stratification",
    "treatment_response_enrichment",
    "recurrence_risk_enrichment",
]

ClinicalScoreType = Literal[
    "diagnostic_likelihood",
    "prognostic_risk",
    "recurrence_risk",
    "treatment_response",
]

SelectionUse = Literal[
    "enrichment",
    "stratification",
]

DatasetGroup = Literal[
    "standard_reference",
    "pharma_core",
]

MLTask = Literal[
    "binary_classification",
    "regression",
]


# =============================================================================
# Constants: MS INNI dataset configuration
# =============================================================================

MS_INNI_FOLDER = "local_data/ML_MS_2026"
MS_INNI_FILENAME = "Valsasina_Front_Artif_Intell_2026.csv"

MS_INNI_ID_COL = "Patient: Hash"
MS_INNI_GROUP_COL = "GROUP-CODE"
MS_INNI_PHENOTYPE_COL = "PHENOTYPE-CODE"
MS_INNI_EDSS_COL = "EDSS"
MS_INNI_DMT_COL = "Treatment-code"

MS_INNI_DISEASE_DURATION_COL = "Disease duration"

MS_INNI_DEMOGRAPHIC_SITE_FEATURES = [
    "Age",
    "SEX-CODE",
    "INSTITUTION-CODE",
]

MS_INNI_MRI_FEATURES = [
    "T2 LV (cubic mm)",
    "ncorticalGREY [ml]",
    "nWHITE [ml]",
    "nThalamus [ml]",
    "nHippocampus [ml]",
    "nOtherDGM [ml]",
    "nBrainstem [ml]",
    "nGM_Cerebellum [ml]",
    "nWM_Cerebellum [ml]",
]

# Paper-aligned feature set for MS-vs-HC classification.
# MS-only clinical variables are excluded because they are unavailable/not
# applicable for healthy controls and would leak disease status.
MS_INNI_DIAGNOSIS_FEATURES = [
    *MS_INNI_DEMOGRAPHIC_SITE_FEATURES,
    *MS_INNI_MRI_FEATURES,
]

# Paper-aligned feature set for relapsing-vs-progressive MS classification.
# This is an MS-only task, so clinical severity, disease-duration, and treatment
# variables available in the released CSV are included.
MS_INNI_PHENOTYPE_FEATURES = [
    *MS_INNI_DEMOGRAPHIC_SITE_FEATURES,
    "T2 LV (cubic mm)",
    MS_INNI_DISEASE_DURATION_COL,
    MS_INNI_EDSS_COL,
    MS_INNI_DMT_COL,
    "ncorticalGREY [ml]",
    "nWHITE [ml]",
    "nThalamus [ml]",
    "nHippocampus [ml]",
    "nOtherDGM [ml]",
    "nBrainstem [ml]",
    "nGM_Cerebellum [ml]",
    "nWM_Cerebellum [ml]",
]

# Paper-aligned feature set for cross-sectional EDSS regression.
# EDSS is the target, so it is intentionally excluded from X.
MS_INNI_EDSS_REGRESSION_FEATURES = [
    *MS_INNI_DEMOGRAPHIC_SITE_FEATURES,
    "T2 LV (cubic mm)",
    MS_INNI_DISEASE_DURATION_COL,
    MS_INNI_DMT_COL,
    "ncorticalGREY [ml]",
    "nWHITE [ml]",
    "nThalamus [ml]",
    "nHippocampus [ml]",
    "nOtherDGM [ml]",
    "nBrainstem [ml]",
    "nGM_Cerebellum [ml]",
    "nWM_Cerebellum [ml]",
]

MS_INNI_ALL_FEATURES = sorted(
    set(
        MS_INNI_DIAGNOSIS_FEATURES
        + MS_INNI_PHENOTYPE_FEATURES
        + MS_INNI_EDSS_REGRESSION_FEATURES
    )
)


# =============================================================================
# Constants: pharma-focused local dataset configuration
# =============================================================================

# These datasets are stored locally, following the same pattern as MS INNI.
# The first filename in each list is the preferred year-tagged filename. The
# second filename is accepted as a convenience fallback when the original
# downloaded filename is kept unchanged.

RA_BDMARD_FOLDER = "local_data/RA_BDMARD_RESPONSE_2024"
RA_BDMARD_FILENAMES = [
    "data_bdmards_2024.csv",
    "data_bdmards.csv",
]

MELANOMA_PD1_PROTEOMICS_FOLDER = "local_data/MELANOMA_PD1_PROTEOMICS_RESPONSE_2022"
MELANOMA_PD1_PROTEOMICS_FILENAMES = [
    "melanoma_pd1_proteomics_2022_baseline_combined_processed.csv",
    "processed/melanoma_pd1_proteomics_2022_baseline_combined_processed.csv",
]

PROSTATE_CANCER_FOLLOWUP_FOLDER = "local_data/PROSTATE_CANCER_FOLLOWUP_2025"
PROSTATE_CANCER_FOLLOWUP_FILENAMES = [
    "prostat_ca_veri_seti_duzeltilmis_v2.csv",
]

NSCLC_ICI_RESPONSE_FOLDER = "local_data/NSCLC_ICI_RESPONSE_2023"
NSCLC_ICI_RESPONSE_FILENAMES = [
    "41588_2023_1355_MOESM3_ESM.xlsx",
]

RA_BDMARD_TARGET_COL = "remission"
MELANOMA_PD1_PROTEOMICS_TARGET_COL = "target_response"
PROSTATE_BCR_TARGET_COL = "BCR_Durum"
NSCLC_ICI_TARGET_COL = "target_response"

# Final pharma-focused portfolio. These are the local datasets currently used
# as the core therapy-response and recurrence-risk examples.
CORE_PHARMA_DATASET_NAMES = {
    "ra_bdmard_remission_2024",
    "melanoma_pd1_proteomics_response_2022",
    "prostate_bcr_prediction_2025",
    "nsclc_ici_response_2023",
}


# =============================================================================
# Catalog framework metadata helpers
# =============================================================================

def _infer_clinical_score_type(task_type: str) -> ClinicalScoreType:
    """
    Infer the clinical meaning of the patient-level model score.

    This value describes what the score represents clinically. It is separate
    from selection_use, which describes whether the score is used for threshold-
    based enrichment or ordered stratification.
    """

    if task_type.startswith("diagnostic_"):
        return "diagnostic_likelihood"

    if task_type.startswith("treatment_response_"):
        return "treatment_response"

    if task_type.startswith("recurrence_risk_"):
        return "recurrence_risk"

    if task_type.startswith("prognostic_"):
        return "prognostic_risk"

    raise ValueError(f"Unsupported task_type for score interpretation: {task_type}")


def _infer_selection_use(task_type: str) -> SelectionUse:
    """Infer whether the score is used for enrichment or stratification."""

    if task_type.endswith("_enrichment"):
        return "enrichment"

    if task_type.endswith("_stratification"):
        return "stratification"

    raise ValueError(f"Unsupported task_type for selection use: {task_type}")


def _add_catalog_framework_metadata(
    catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Add patient-selection framework fields to each catalog row.

    Added fields
    ------------
    clinical_score_type:
        Clinical interpretation of the model score, such as diagnostic
        likelihood, prognostic risk, recurrence risk, or treatment response.

    selection_use:
        Operational use of the score: enrichment or stratification.

    dataset_group:
        Whether the row belongs to the standard reference datasets or the final
        pharma-focused core portfolio.

    is_core_pharma:
        Boolean convenience flag for filtering the final pharma portfolio.
    """

    catalog_with_framework_metadata: list[dict[str, Any]] = []

    for row in catalog:
        updated_row = dict(row)
        task_type = str(updated_row["task_type"])
        dataset_name = str(updated_row["dataset_name"])
        is_core_pharma = dataset_name in CORE_PHARMA_DATASET_NAMES

        updated_row["clinical_score_type"] = _infer_clinical_score_type(task_type)
        updated_row["selection_use"] = _infer_selection_use(task_type)
        updated_row["dataset_group"] = (
            "pharma_core" if is_core_pharma else "standard_reference"
        )
        updated_row["is_core_pharma"] = is_core_pharma

        catalog_with_framework_metadata.append(updated_row)

    return catalog_with_framework_metadata


# =============================================================================
# Public API
# =============================================================================

def get_healthcare_dataset_catalog() -> pd.DataFrame:
    """
    Return a catalog of available healthcare datasets.

    Returns
    -------
    catalog_df:
        DataFrame where each row describes one available healthcare dataset.
    """

    # Each row defines one dataset option exposed to notebook workflows.
    catalog: list[dict[str, Any]] = [
        {
            "dataset_name": "load_breast_cancer",
            "display_name": "Breast Cancer Wisconsin Diagnostic",
            "source": "sklearn",
            "loader": "load_breast_cancer",
            "data_id": None,
            "disease_area": "oncology",
            "task_type": "diagnostic_enrichment",
            "ml_task": "binary_classification",
            "target_goal": "malignant_vs_benign",
            "notes": "Clean sklearn-native binary classification dataset.",
        },
        {
            "dataset_name": "pima_indians_diabetes",
            "display_name": "Pima Indians Diabetes",
            "source": "openml",
            "loader": "fetch_openml",
            "data_id": 37,
            "disease_area": "metabolic",
            "task_type": "diagnostic_enrichment",
            "ml_task": "binary_classification",
            "target_goal": "diabetes_vs_no_diabetes",
            "notes": "Classic diabetes prediction dataset from OpenML.",
        },
        {
            "dataset_name": "breast_cancer_coimbra",
            "display_name": "Breast Cancer Coimbra",
            "source": "openml",
            "loader": "fetch_openml",
            "data_id": 42900,
            "disease_area": "oncology",
            "task_type": "diagnostic_enrichment",
            "ml_task": "binary_classification",
            "target_goal": "breast_cancer_vs_healthy_control",
            "notes": (
                "Biomarker-style dataset using anthropometric and "
                "blood-analysis predictors."
            ),
        },
        {
            "dataset_name": "indian_liver_patient",
            "display_name": "Indian Liver Patient Dataset",
            "source": "openml",
            "loader": "fetch_openml",
            "data_id": 1480,
            "disease_area": "hepatology",
            "task_type": "diagnostic_enrichment",
            "ml_task": "binary_classification",
            "target_goal": "liver_patient_vs_non_liver_patient",
            "notes": "Biochemical-marker dataset for liver disease classification.",
        },
        {
            "dataset_name": "heart_disease_comprehensive",
            "display_name": "Heart Disease Comprehensive",
            "source": "openml",
            "loader": "fetch_openml",
            "data_id": 43672,
            "disease_area": "cardiology",
            "task_type": "diagnostic_enrichment",
            "ml_task": "binary_classification",
            "target_goal": "heart_disease_vs_no_heart_disease",
            "notes": "Heart disease dataset useful for cardiovascular risk modeling.",
        },
        {
            "dataset_name": "parkinsons_disease",
            "display_name": "Parkinson's Disease",
            "source": "openml",
            "loader": "fetch_openml",
            "data_id": 1488,
            "disease_area": "neurology",
            "task_type": "diagnostic_enrichment",
            "ml_task": "binary_classification",
            "target_goal": "parkinsons_disease_vs_healthy_control",
            "notes": (
                "Voice-measurement dataset for distinguishing Parkinson's "
                "disease from healthy controls."
            ),
        },
        {
            "dataset_name": "diabetic_retinopathy_debrecen",
            "display_name": "Diabetic Retinopathy Debrecen",
            "source": "uci",
            "loader": "fetch_ucirepo",
            "data_id": 329,
            "disease_area": "ophthalmology",
            "task_type": "diagnostic_stratification",
            "ml_task": "binary_classification",
            "target_goal": "diabetic_retinopathy_signs_vs_no_signs",
            "notes": (
                "Tabular features extracted from retinal images to predict "
                "signs of diabetic retinopathy."
            ),
        },
        {
            "dataset_name": "thoracic_surgery",
            "display_name": "Thoracic Surgery Data",
            "source": "uci",
            "loader": "fetch_ucirepo",
            "data_id": 277,
            "disease_area": "respiratory",
            "task_type": "prognostic_enrichment",
            "ml_task": "binary_classification",
            "target_goal": "post_surgical_survival_or_risk_outcome",
            "notes": (
                "Pulmonary/thoracic surgery dataset from patients who underwent "
                "major lung resections for primary lung cancer."
            ),
        },
        {
            "dataset_name": "ms_inni_diagnosis",
            "display_name": "INNI Multiple Sclerosis: MS vs Healthy Control",
            "source": "local_csv",
            "loader": "load_ms_inni",
            "data_id": "ORDR_hnsppf3k2p_v1",
            "disease_area": "neurology",
            "task_type": "diagnostic_enrichment",
            "ml_task": "binary_classification",
            "target_goal": "multiple_sclerosis_vs_healthy_control",
            "notes": (
                "Paper-aligned classification task distinguishing MS patients "
                "from healthy controls using demographic/site and MRI-derived "
                "features available for both groups."
            ),
        },
        {
            "dataset_name": "ms_inni_phenotype",
            "display_name": "INNI Multiple Sclerosis: Relapsing vs Progressive MS",
            "source": "local_csv",
            "loader": "load_ms_inni",
            "data_id": "ORDR_hnsppf3k2p_v1",
            "disease_area": "neurology",
            "task_type": "diagnostic_stratification",
            "ml_task": "binary_classification",
            "target_goal": "progressive_vs_relapsing_ms",
            "notes": (
                "Paper-aligned MS-only phenotype classification task. The "
                "target is derived from PHENOTYPE-CODE: relapsing MS=0 and "
                "progressive MS=1."
            ),
        },
        {
            "dataset_name": "ms_inni_edss_regression",
            "display_name": "INNI Multiple Sclerosis: Cross-sectional EDSS Prediction",
            "source": "local_csv",
            "loader": "load_ms_inni",
            "data_id": "ORDR_hnsppf3k2p_v1",
            "disease_area": "neurology",
            "task_type": "prognostic_enrichment",
            "ml_task": "regression",
            "target_goal": "predict_observed_cross_sectional_edss_score",
            "notes": (
                "Paper-aligned MS-only regression task predicting the observed "
                "cross-sectional EDSS score from concurrently available "
                "demographic, clinical, and MRI-derived features."
            ),
        },

        {
            "dataset_name": "ra_bdmard_remission_2024",
            "display_name": "Rheumatoid Arthritis bDMARD: 6-Month Remission",
            "source": "local_csv",
            "loader": "load_ra_bdmard",
            "data_id": "zenodo_12506988",
            "disease_area": "rheumatology_autoimmune",
            "task_type": "treatment_response_enrichment",
            "ml_task": "binary_classification",
            "target_goal": "remission_after_bdmard_therapy",
            "notes": (
                "Baseline clinical variables from RA patients treated with "
                "biological DMARDs. The default target is remission after "
                "6-month follow-up."
            ),
        },
        {
            "dataset_name": "melanoma_pd1_proteomics_response_2022",
            "display_name": "Metastatic Melanoma PD-1 Proteomics Response",
            "source": "local_csv",
            "loader": "load_melanoma_pd1_proteomics",
            "data_id": "mendeley_h2fr3nwzc6_v1_processed",
            "disease_area": "oncology_melanoma",
            "task_type": "treatment_response_enrichment",
            "ml_task": "binary_classification",
            "target_goal": "pd1_responder_vs_non_responder",
            "notes": (
                "Baseline plasma proteomics features from metastatic melanoma "
                "patients treated with PD-1 immune checkpoint blockade. The "
                "target is responder vs non-responder."
            ),
        },
        {
            "dataset_name": "prostate_bcr_prediction_2025",
            "display_name": "Prostate Cancer Biochemical Recurrence Prediction",
            "source": "local_csv",
            "loader": "load_prostate_cancer_followup",
            "data_id": "zenodo_15007105",
            "disease_area": "oncology_prostate_cancer",
            "task_type": "recurrence_risk_enrichment",
            "ml_task": "binary_classification",
            "target_goal": "biochemical_recurrence_vs_no_recurrence",
            "notes": (
                "Clinical, pathological, laboratory, treatment, and follow-up "
                "data from prostate cancer patients. The default target is "
                "biochemical recurrence status."
            ),
        },
        {
            "dataset_name": "nsclc_ici_response_2023",
            "display_name": "Advanced NSCLC Checkpoint Blockade Response",
            "source": "local_excel",
            "loader": "load_nsclc_ici_response",
            "data_id": "nature_genetics_2023_41588_2023_1355_moesm3",
            "disease_area": "oncology_nsclc",
            "task_type": "treatment_response_enrichment",
            "ml_task": "binary_classification",
            "target_goal": "checkpoint_blockade_responder_vs_non_responder",
            "notes": (
                "Advanced NSCLC cohort treated with immune checkpoint blockade. "
                "The default target maps confirmed CR/PR to responder and "
                "SD/PD to non-responder. Features include baseline clinical "
                "variables plus available genomic and immune-signature summaries."
            ),
        },
    ]

    # Add framework-level metadata after defining the compact catalog rows.
    # This keeps task_type, clinical_score_type, and selection_use aligned.
    catalog = _add_catalog_framework_metadata(catalog)

    # Return as a DataFrame so users can filter, display, and inspect it easily.
    return pd.DataFrame(catalog)


def load_healthcare_dataset(
    dataset_name: str = "load_breast_cancer",
) -> dict[str, Any]:
    """
    Load a selected healthcare dataset.

    Parameters
    ----------
    dataset_name:
        Name of the dataset to load.

    Returns
    -------
    dataset_dict:
        Dictionary containing df, X, y, feature_names, metadata, data_keys,
        and raw_data.
    """

    # Look up the requested dataset in the registered catalog.
    catalog_df = get_healthcare_dataset_catalog()

    # Fail early with a helpful message if the dataset name is not registered.
    if dataset_name not in catalog_df["dataset_name"].values:
        available = catalog_df["dataset_name"].tolist()
        raise ValueError(
            f"Dataset '{dataset_name}' was not found. "
            f"Available datasets are: {available}"
        )

    # Convert the selected catalog row into metadata that travels with the dataset.
    metadata_row = catalog_df.loc[
        catalog_df["dataset_name"] == dataset_name
    ].iloc[0]

    metadata: dict[str, Any] = metadata_row.to_dict()

    # Route to the correct backend loader based on source and loader metadata.
    if metadata["source"] == "sklearn":
        return _load_sklearn_dataset(metadata)

    if metadata["source"] == "openml":
        return _load_openml_dataset(metadata)

    if metadata["source"] == "uci":
        return _load_uci_dataset(metadata)

    if metadata["source"] == "local_csv" and metadata["loader"] == "load_ms_inni":
        return _load_ms_inni_dataset(
            dataset_name=dataset_name,
            metadata=metadata,
        )


    if metadata["source"] == "local_csv" and metadata["loader"] == "load_ra_bdmard":
        return _load_ra_bdmard_dataset(metadata=metadata)

    if metadata["source"] == "local_csv" and metadata["loader"] == "load_melanoma_pd1_proteomics":
        return _load_melanoma_pd1_proteomics_dataset(metadata=metadata)

    if metadata["source"] == "local_csv" and metadata["loader"] == "load_prostate_cancer_followup":
        return _load_prostate_cancer_followup_dataset(metadata=metadata)

    if metadata["source"] == "local_excel" and metadata["loader"] == "load_nsclc_ici_response":
        return _load_nsclc_ici_response_dataset(metadata=metadata)

    # Any unrecognized source/loader combination is treated as a catalog error.
    raise ValueError(
        f"Unsupported source/loader combination: "
        f"source={metadata['source']}, loader={metadata['loader']}"
    )


# =============================================================================
# Generic source-specific loaders
# =============================================================================

def _load_sklearn_dataset(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Load a sklearn-native healthcare dataset.

    Parameters
    ----------
    metadata:
        Metadata row from the healthcare dataset catalog.

    Returns
    -------
    dataset_dict:
        Standardized dataset dictionary.
    """

    # Keep this helper explicit: currently only the sklearn breast-cancer
    # dataset is registered as a sklearn-native source.
    if metadata["loader"] != "load_breast_cancer":
        raise ValueError(f"Unsupported sklearn loader: {metadata['loader']}")

    # Request the sklearn Bunch with pandas-backed data objects.
    data = load_breast_cancer(as_frame=True)

    # Standardize sklearn's output into the common dataset dictionary format.
    X = data.data.copy()
    y = data.target.copy()
    feature_names = [str(name) for name in list(data.feature_names)]

    # Keep the feature matrix column labels aligned with standardized names.
    X.columns = feature_names

    # Rebuild the full modeling DataFrame from standardized X and y.
    df = pd.concat([X, y.rename("target")], axis=1)

    # Preserve sklearn's original Bunch fields without allowing them to
    # overwrite the standardized top-level API keys below.
    source_data = dict(data)
    source_data.pop("feature_names", None)

    dataset_dict: dict[str, Any] = {
        **source_data,
        "df": df,
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "metadata": metadata,
        "data_keys": list(data.keys()),
        "raw_data": data,
    }

    return dataset_dict

def _load_openml_dataset(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Load an OpenML healthcare dataset.

    Parameters
    ----------
    metadata:
        Metadata row from the healthcare dataset catalog.

    Returns
    -------
    dataset_dict:
        Standardized dataset dictionary.
    """

    # Fetch the OpenML dataset by the numeric data_id stored in the catalog.
    data = fetch_openml(
        data_id=int(metadata["data_id"]),
        as_frame=True,
        parser="auto",
    )

    # Standardize OpenML's returned Bunch into the common dataset format.
    df = data.frame.copy()
    X = data.data
    y = data.target
    feature_names = list(data.data.columns)

    # Standardize feature names to plain Python strings and align X columns.
    X, feature_names = _standardize_feature_names(
        X=X,
        feature_names=feature_names,
    )

    # Keep the full modeling DataFrame aligned with standardized feature names.
    df = pd.concat([X, y.rename(data.target.name)], axis=1)

    dataset_dict: dict[str, Any] = {
        "df": df,
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "metadata": metadata,
        "data_keys": list(data.keys()),
        "raw_data": data,
    }

    # Preserve OpenML's original Bunch keys for users who need source details.
    dataset_dict.update(dict(data))
    return dataset_dict


def _load_uci_dataset(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Load a UCI healthcare dataset using ucimlrepo.

    Parameters
    ----------
    metadata:
        Metadata row from the healthcare dataset catalog.

    Returns
    -------
    dataset_dict:
        Standardized dataset dictionary.
    """

    # Fetch the UCI dataset by the numeric id stored in the catalog. Keep the
    # dependency optional so local pharma datasets can still be loaded in
    # environments where ucimlrepo is not installed.
    if fetch_ucirepo is None:
        raise ImportError(
            "The ucimlrepo package is required to load UCI datasets. "
            "Install ucimlrepo or select a non-UCI dataset."
        )

    data = fetch_ucirepo(id=int(metadata["data_id"]))

    # UCI separates features and targets, so combine them only for df.
    X = data.data.features
    y = data.data.targets
    feature_names = list(X.columns)

    # Standardize feature names to plain Python strings and align X columns.
    X, feature_names = _standardize_feature_names(
        X=X,
        feature_names=feature_names,
    )

    # Keep the full modeling DataFrame aligned with standardized feature names.
    df = pd.concat([X, y], axis=1)

    # Return both standardized fields and UCI-specific source objects.
    return {
        "df": df,
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "metadata": metadata,
        "data_keys": [
            "data",
            "metadata",
            "variables",
            "data.features",
            "data.targets",
        ],
        "raw_data": data,
        "uci_data": data.data,
        "uci_metadata": data.metadata,
        "uci_variables": data.variables,
    }



# =============================================================================
# Pharma-focused local dataset utilities and loaders
# =============================================================================

def _get_local_dataset_file_path(
    *,
    folder: str,
    filenames: list[str],
) -> Path:
    """
    Return the first existing local dataset file from a list of candidates.

    Parameters
    ----------
    folder:
        Folder path relative to this module.

    filenames:
        Candidate filenames in priority order.

    Returns
    -------
    file_path:
        Existing path for the first matching local file.

    Raises
    ------
    FileNotFoundError
        If none of the candidate files are found.
    """

    module_dir = Path(__file__).resolve().parent
    folder_path = module_dir / folder

    for filename in filenames:
        file_path = folder_path / filename
        if file_path.exists():
            return file_path

    expected = [str(folder_path / filename) for filename in filenames]
    raise FileNotFoundError(
        "Local dataset file was not found. Expected one of:\n"
        + "\n".join(expected)
    )


def _drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with one unique value, including all-missing columns."""

    keep_cols = [
        col for col in df.columns
        if df[col].nunique(dropna=False) > 1
    ]
    return df[keep_cols].copy()


def _augment_binary_classification_metadata(
    *,
    metadata: dict[str, Any],
    target_column: str,
    target_definition: str,
    positive_class: str,
    negative_class: str,
    class_mapping: dict[int, str],
    y: pd.Series,
    feature_names: list[str],
    population_filter: str,
    feature_groups: dict[str, list[str]] | None = None,
    target_handling: str | None = None,
    prediction_time_horizon: str | None = None,
    longitudinal_prediction: bool | None = None,
) -> dict[str, Any]:
    """
    Add standard metadata for binary local pharma-focused datasets.

    Catalog-level fields such as task_type, clinical_score_type, selection_use,
    dataset_group, and is_core_pharma are preserved from the selected catalog
    row. This helper adds target-specific details after the source file has been
    loaded and the final modeling view has been constructed.
    """

    updated_metadata = dict(metadata)
    updated_metadata.update(
        {
            "target_column": target_column,
            "target_definition": target_definition,
            "positive_class": positive_class,
            "negative_class": negative_class,
            "class_mapping": class_mapping,
            "class_distribution": y.value_counts().sort_index().to_dict(),
            "feature_names": feature_names,
            "population_filter": population_filter,
            "feature_groups": feature_groups or {},
            "target_handling": target_handling,
            "prediction_time_horizon": prediction_time_horizon,
            "longitudinal_prediction": longitudinal_prediction,
        }
    )
    return updated_metadata


def _load_ra_bdmard_dataset(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Load the RA bDMARD treatment-response dataset.

    Default modeling view
    ---------------------
    Predict 6-month remission after bDMARD therapy from baseline clinical
    features. Other outcome columns in the source file are excluded from X.
    """

    csv_path = _get_local_dataset_file_path(
        folder=RA_BDMARD_FOLDER,
        filenames=RA_BDMARD_FILENAMES,
    )
    raw_df = pd.read_csv(csv_path)
    df = raw_df.copy()

    target_col = RA_BDMARD_TARGET_COL
    if target_col not in df.columns:
        raise ValueError(f"RA bDMARD target column is missing: {target_col}")

    exclude_cols = [
        "index",
        "tptID",
        "level_0",
        "remission",
        "effectiveness",
        "sustained",
    ]

    model_df = df.dropna(subset=[target_col]).copy()
    y = model_df[target_col].astype(int).rename(target_col)

    feature_cols = [
        col for col in model_df.columns
        if col not in exclude_cols
    ]
    X = _drop_constant_columns(model_df[feature_cols])
    feature_names = list(X.columns)

    metadata = _augment_binary_classification_metadata(
        metadata=metadata,
        target_column=target_col,
        target_definition=(
            "Remission after 6-month follow-up among rheumatoid arthritis "
            "patients treated with biological DMARDs."
        ),
        positive_class="remission",
        negative_class="no_remission",
        class_mapping={0: "no_remission", 1: "remission"},
        y=y,
        feature_names=feature_names,
        population_filter="rheumatoid_arthritis_patients_treated_with_bdmards",
        feature_groups={
            "clinical_baseline": feature_names,
        },
        target_handling=(
            "The source outcome columns remission, effectiveness, and sustained "
            "are excluded from the feature matrix. The default view uses "
            "remission as the binary target."
        ),
        prediction_time_horizon="6_month_follow_up",
        longitudinal_prediction=True,
    )

    return _build_dataset_dict(
        df=pd.concat([X, y], axis=1),
        X=X,
        y=y,
        feature_names=feature_names,
        metadata=metadata,
        raw_data=raw_df,
    )


def _load_melanoma_pd1_proteomics_dataset(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Load the metastatic melanoma PD-1 plasma-proteomics response dataset.

    The loader expects the preprocessed baseline-wide CSV created from the
    released training and validation files. Rows represent baseline samples;
    protein features are already pivoted into columns.
    """

    csv_path = _get_local_dataset_file_path(
        folder=MELANOMA_PD1_PROTEOMICS_FOLDER,
        filenames=MELANOMA_PD1_PROTEOMICS_FILENAMES,
    )
    raw_df = pd.read_csv(csv_path)
    df = raw_df.copy()

    target_col = MELANOMA_PD1_PROTEOMICS_TARGET_COL
    if target_col not in df.columns:
        raise ValueError(
            f"Melanoma proteomics target column is missing: {target_col}"
        )

    exclude_cols = [
        "split",
        "SampleId",
        "patient_id",
        "response_simple",
        target_col,
        "pfs",
        "censure_pfs",
        "os",
        "censure_os",
        "pre_day",
        "days_from_baseline",
    ]

    model_df = df.dropna(subset=[target_col]).copy()
    y = model_df[target_col].astype(int).rename(target_col)

    feature_cols = [
        col for col in model_df.columns
        if col not in exclude_cols
    ]
    X = _drop_constant_columns(model_df[feature_cols])
    feature_names = list(X.columns)

    protein_features = [col for col in feature_names if col.startswith("prot_")]
    clinical_features = [col for col in feature_names if col not in protein_features]

    metadata = _augment_binary_classification_metadata(
        metadata=metadata,
        target_column=target_col,
        target_definition=(
            "Responder vs non-responder status after PD-1 immune checkpoint "
            "blockade in metastatic melanoma."
        ),
        positive_class="responder",
        negative_class="non_responder",
        class_mapping={0: "non_responder", 1: "responder"},
        y=y,
        feature_names=feature_names,
        population_filter="baseline_metastatic_melanoma_pd1_treated_samples",
        feature_groups={
            "clinical_baseline": clinical_features,
            "plasma_proteomics": protein_features,
        },
        target_handling=(
            "Progression-free survival and overall survival columns are excluded "
            "from the feature matrix to avoid post-outcome leakage."
        ),
        prediction_time_horizon="post_treatment_response",
        longitudinal_prediction=True,
    )

    return _build_dataset_dict(
        df=pd.concat([X, y], axis=1),
        X=X,
        y=y,
        feature_names=feature_names,
        metadata=metadata,
        raw_data=raw_df,
    )


def _load_prostate_cancer_followup_dataset(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Load the prostate cancer follow-up dataset.

    Default modeling view
    ---------------------
    Predict biochemical recurrence status from clinical, laboratory,
    treatment, and pathology variables. This is a recurrence-risk enrichment
    task rather than a treatment-response task. Follow-up and post-recurrence
    columns are excluded from X.
    """

    csv_path = _get_local_dataset_file_path(
        folder=PROSTATE_CANCER_FOLLOWUP_FOLDER,
        filenames=PROSTATE_CANCER_FOLLOWUP_FILENAMES,
    )
    raw_df = pd.read_csv(csv_path)
    df = raw_df.copy()

    target_col = PROSTATE_BCR_TARGET_COL
    if target_col not in df.columns:
        raise ValueError(f"Prostate BCR target column is missing: {target_col}")

    preferred_feature_cols = [
        "Yas",
        "PSA_Tani",
        "Klinik_Evre",
        "Biyopsi_Gleason",
        "Risk_Grubu",
        "Albumin",
        "Lenfosit",
        "CRP",
        "NLR",
        "CALLY_Index",
        "Komorbidite_Skor",
        "Tedavi_Tipi",
        "RT_Dozu",
        "ADT_Tipi",
        "ADT_Suresi",
        "Patolojik_Evre",
        "Cerrahi_Sinir",
        "Final_Gleason",
    ]

    model_df = df.dropna(subset=[target_col]).copy()
    y = model_df[target_col].astype(int).rename(target_col)

    feature_cols = [
        col for col in preferred_feature_cols
        if col in model_df.columns
    ]
    X = _drop_constant_columns(model_df[feature_cols])
    feature_names = list(X.columns)

    metadata = _augment_binary_classification_metadata(
        metadata=metadata,
        target_column=target_col,
        target_definition="Biochemical recurrence status in prostate cancer follow-up.",
        positive_class="biochemical_recurrence",
        negative_class="no_biochemical_recurrence",
        class_mapping={0: "no_biochemical_recurrence", 1: "biochemical_recurrence"},
        y=y,
        feature_names=feature_names,
        population_filter="prostate_cancer_patients_with_bcr_status",
        feature_groups={
            "clinical_laboratory_treatment_pathology": feature_names,
        },
        target_handling=(
            "BCR date, metastasis, survival, follow-up, PSA nadir, and serial "
            "post-treatment PSA follow-up columns are excluded from X to reduce "
            "outcome leakage."
        ),
        prediction_time_horizon="follow_up_biochemical_recurrence",
        longitudinal_prediction=True,
    )

    return _build_dataset_dict(
        df=pd.concat([X, y], axis=1),
        X=X,
        y=y,
        feature_names=feature_names,
        metadata=metadata,
        raw_data=raw_df,
    )


def _prefix_feature_columns(
    df: pd.DataFrame,
    *,
    key_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Prefix non-key columns in a feature table before merging."""

    renamed = {
        col: f"{prefix}{col}"
        for col in df.columns
        if col != key_col
    }
    return df.rename(columns=renamed)


def _read_excel_feature_sheet(
    excel_path: Path,
    *,
    sheet_name: str,
    key_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Read and prefix an Excel feature sheet used by the NSCLC loader."""

    feature_df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if key_col not in feature_df.columns:
        raise ValueError(
            f"Expected key column '{key_col}' in sheet '{sheet_name}'."
        )
    feature_df = feature_df.drop_duplicates(subset=[key_col]).copy()
    return _prefix_feature_columns(
        feature_df,
        key_col=key_col,
        prefix=prefix,
    )


def _load_nsclc_ici_response_dataset(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Load the advanced NSCLC checkpoint-blockade response workbook.

    Default modeling view
    ---------------------
    Predict confirmed best overall response after checkpoint blockade. CR/PR
    are mapped to responder=1; SD/PD are mapped to non-responder=0. NE and
    missing response labels are excluded.
    """

    excel_path = _get_local_dataset_file_path(
        folder=NSCLC_ICI_RESPONSE_FOLDER,
        filenames=NSCLC_ICI_RESPONSE_FILENAMES,
    )

    # The first two rows are workbook annotations; row 3 contains real headers.
    clinical = pd.read_excel(
        excel_path,
        sheet_name="Table_S1_Clinical_Annotations",
        header=2,
    )

    response_col = "Harmonized_Confirmed_BOR"
    if response_col not in clinical.columns:
        raise ValueError(f"NSCLC response column is missing: {response_col}")

    response_mapping = {
        "CR": 1,
        "PR": 1,
        "SD": 0,
        "PD": 0,
    }
    clinical[NSCLC_ICI_TARGET_COL] = clinical[response_col].map(response_mapping)
    model_df = clinical.dropna(subset=[NSCLC_ICI_TARGET_COL]).copy()
    model_df[NSCLC_ICI_TARGET_COL] = model_df[NSCLC_ICI_TARGET_COL].astype(int)

    wes_key = "Harmonized_SU2C_WES_Tumor_Sample_ID_v2"
    rna_key = "Harmonized_SU2C_RNA_Tumor_Sample_ID_v2"

    feature_tables = [
        _read_excel_feature_sheet(
            excel_path,
            sheet_name="Table_S5_Mutation_Burden",
            key_col=wes_key,
            prefix="wes_",
        ),
        _read_excel_feature_sheet(
            excel_path,
            sheet_name="Table_S10_Antigen_Presentation",
            key_col=wes_key,
            prefix="antigen_",
        ),
        _read_excel_feature_sheet(
            excel_path,
            sheet_name="Table_S11_Mixcr_Results",
            key_col=wes_key,
            prefix="mixcr_",
        ),
        _read_excel_feature_sheet(
            excel_path,
            sheet_name="Table_S18_Immune_Signatures",
            key_col=rna_key,
            prefix="immune_",
        ),
        _read_excel_feature_sheet(
            excel_path,
            sheet_name="Table_S19_Myeloid_Signatures",
            key_col=rna_key,
            prefix="myeloid_",
        ),
        _read_excel_feature_sheet(
            excel_path,
            sheet_name="Table_S20_Curated_Signatures",
            key_col=rna_key,
            prefix="curated_",
        ),
    ]

    for feature_df in feature_tables:
        merge_key = wes_key if wes_key in feature_df.columns else rna_key
        model_df = model_df.merge(feature_df, on=merge_key, how="left")

    clinical_feature_cols = [
        "Institution",
        "Patient_Age_at_Diagnosis",
        "Patient_Sex",
        "Patient_Race",
        "Patient_Smoking_Status",
        "Patient_Smoking_Pack_Years_Harmonized",
        "Histology_Harmonized",
        "Initial_Stage",
        "Initial_Stage_Substage",
        "PDL1_TPS",
        "Clinical_Driver",
        "Line_of_Therapy",
        "Agent_PD1_Category",
        "Prior_Platinum",
        "Prior_TKI",
    ]

    derived_feature_cols = [
        col for col in model_df.columns
        if col.startswith((
            "wes_",
            "antigen_",
            "mixcr_",
            "immune_",
            "myeloid_",
            "curated_",
        ))
    ]

    feature_cols = [
        col for col in clinical_feature_cols + derived_feature_cols
        if col in model_df.columns
    ]

    X = _drop_constant_columns(model_df[feature_cols])
    feature_names = list(X.columns)
    y = model_df[NSCLC_ICI_TARGET_COL].astype(int).rename(NSCLC_ICI_TARGET_COL)

    feature_groups = {
        "clinical_baseline": [
            col for col in feature_names
            if col in clinical_feature_cols
        ],
        "wes_summary": [
            col for col in feature_names
            if col.startswith("wes_")
        ],
        "antigen_presentation": [
            col for col in feature_names
            if col.startswith("antigen_")
        ],
        "immune_signatures": [
            col for col in feature_names
            if col.startswith("immune_")
        ],
        "myeloid_signatures": [
            col for col in feature_names
            if col.startswith("myeloid_")
        ],
        "curated_signatures": [
            col for col in feature_names
            if col.startswith("curated_")
        ],
    }

    metadata = _augment_binary_classification_metadata(
        metadata=metadata,
        target_column=NSCLC_ICI_TARGET_COL,
        target_definition=(
            "Confirmed best overall response to immune checkpoint blockade in "
            "advanced non-small cell lung cancer. CR/PR are responders; SD/PD "
            "are non-responders."
        ),
        positive_class="responder_cr_or_pr",
        negative_class="non_responder_sd_or_pd",
        class_mapping={0: "non_responder_sd_or_pd", 1: "responder_cr_or_pr"},
        y=y,
        feature_names=feature_names,
        population_filter="advanced_nsclc_patients_with_confirmed_bor_cr_pr_sd_or_pd",
        feature_groups=feature_groups,
        target_handling=(
            "Harmonized_Confirmed_BOR is mapped as CR/PR=1 and SD/PD=0. "
            "NE and missing labels are excluded. PFS, OS, and RECIST outcome "
            "columns are excluded from X."
        ),
        prediction_time_horizon="post_checkpoint_blockade_response",
        longitudinal_prediction=True,
    )

    return _build_dataset_dict(
        df=pd.concat([X, y], axis=1),
        X=X,
        y=y,
        feature_names=feature_names,
        metadata=metadata,
        raw_data=clinical,
    )


# =============================================================================
# MS INNI file access and validation
# =============================================================================

def _get_ms_inni_csv_path() -> Path:
    """
    Return the expected local path for the MS INNI CSV file.

    Expected folder structure:

    healthcare_datasets_ms_updated.py
    local_data/
        ML_MS_2026/
            Valsasina_Front _Artif Intell_2026.csv

    Returns
    -------
    csv_path:
        Path to the MS INNI CSV file.

    Raises
    ------
    FileNotFoundError
        If the expected CSV file is not present.
    """

    # Resolve the CSV relative to this module so notebooks can run from any CWD.
    module_dir = Path(__file__).resolve().parent
    csv_path = module_dir / MS_INNI_FOLDER / MS_INNI_FILENAME

    # Fail early if the local clinical CSV has not been placed in the expected path.
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MS INNI CSV was not found at: {csv_path}\n\n"
            f"Expected folder structure:\n"
            f"{module_dir / MS_INNI_FOLDER / MS_INNI_FILENAME}"
        )

    return csv_path


def _read_ms_inni_csv() -> pd.DataFrame:
    """
    Read the MS INNI CSV file.

    The ORDR-released MS INNI CSV is comma-delimited. Delimiter inference is
    used so the loader can also handle semicolon-delimited copies if needed.

    Returns
    -------
    df:
        Raw MS INNI DataFrame.
    """

    # Locate the local source file before reading it.
    csv_path = _get_ms_inni_csv_path()

    # Try UTF-8 first, then fall back to latin1 for compatibility with exported
    # clinical CSV files. Delimiter inference avoids hard-coding comma vs semicolon.
    try:
        return pd.read_csv(
            csv_path,
            sep=None,
            engine="python",
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            csv_path,
            sep=None,
            engine="python",
            encoding="latin1",
        )


def _validate_ms_inni_columns(df: pd.DataFrame) -> None:
    """
    Validate that the expected MS INNI columns are present.

    Parameters
    ----------
    df:
        Raw MS INNI DataFrame.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """

    # Include identifiers, target/source columns, and all paper-aligned features.
    required_columns = {
        MS_INNI_ID_COL,
        MS_INNI_GROUP_COL,
        MS_INNI_PHENOTYPE_COL,
        MS_INNI_EDSS_COL,
        MS_INNI_DMT_COL,
        MS_INNI_DISEASE_DURATION_COL,
        *MS_INNI_ALL_FEATURES,
    }

    # Report all missing columns at once so the CSV can be fixed in one pass.
    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise ValueError(
            "The MS INNI CSV is missing required columns: "
            f"{missing}"
        )


# =============================================================================
# MS INNI dispatcher
# =============================================================================

def _load_ms_inni_dataset(
    *,
    dataset_name: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Load one paper-aligned modeling view from the MS INNI dataset.

    Parameters
    ----------
    dataset_name:
        MS INNI dataset view to load. Supported views correspond to the three
        primary modeling tasks in the associated paper.

    metadata:
        Metadata row from the healthcare dataset catalog.

    Returns
    -------
    dataset_dict:
        Standardized dataset dictionary.
    """

    # Read and validate the shared raw source before creating a modeling view.
    raw_df = _read_ms_inni_csv()
    _validate_ms_inni_columns(raw_df)

    # Dispatch to the requested paper-aligned task-specific modeling view.
    if dataset_name == "ms_inni_diagnosis":
        return _prepare_ms_inni_diagnosis(
            raw_df=raw_df,
            metadata=metadata,
        )

    if dataset_name == "ms_inni_phenotype":
        return _prepare_ms_inni_phenotype(
            raw_df=raw_df,
            metadata=metadata,
        )

    if dataset_name == "ms_inni_edss_regression":
        return _prepare_ms_inni_edss_regression(
            raw_df=raw_df,
            metadata=metadata,
        )

    raise ValueError(f"Unsupported MS INNI dataset view: {dataset_name}")


# =============================================================================
# MS INNI modeling-view builders
# =============================================================================

def _prepare_ms_inni_diagnosis(
    *,
    raw_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Prepare the paper-aligned MS vs healthy-control classification view.

    Target
    ------
    GROUP-CODE:

    - 0: healthy control
    - 1: multiple sclerosis

    Notes
    -----
    MS-only clinical variables such as disease duration, EDSS, and treatment are
    excluded because they are unavailable or not applicable for healthy controls.

    Parameters
    ----------
    raw_df:
        Raw MS INNI DataFrame.

    metadata:
        Dataset metadata.

    Returns
    -------
    dataset_dict:
        Standardized dataset dictionary.
    """

    # Work on a copy so the returned raw_data remains unchanged.
    df = raw_df.copy()

    # Use all participants and predict MS diagnosis status from GROUP-CODE.
    feature_names = list(MS_INNI_DIAGNOSIS_FEATURES)
    target_col = MS_INNI_GROUP_COL

    # Coerce model inputs and target to numeric values before dropping missing rows.
    df = _coerce_ms_inni_numeric_columns(df, feature_names + [target_col])

    # Keep only complete rows for this modeling view.
    model_df = df[feature_names + [target_col]].dropna().copy()

    # Split the modeling DataFrame into features and target.
    X = model_df[feature_names]
    y = model_df[target_col].astype(int)

    # Add task-specific clinical metadata to support downstream reporting.
    metadata = _augment_ms_inni_metadata(
        metadata=metadata,
        target_column=target_col,
        target_definition="GROUP-CODE: 0=healthy_control, 1=multiple_sclerosis",
        positive_class="multiple_sclerosis",
        negative_class="healthy_control",
        population_filter="all_participants",
        class_mapping={
            0: "healthy_control",
            1: "multiple_sclerosis",
        },
        class_distribution=y.value_counts().sort_index().to_dict(),
        feature_names=feature_names,
        target_handling=(
            "EDSS, disease duration, and treatment are excluded because they "
            "are MS-only variables and are unavailable or not applicable for "
            "healthy controls."
        ),
        prediction_time_horizon=None,
        longitudinal_prediction=None,
    )

    return _build_dataset_dict(
        df=model_df,
        X=X,
        y=y,
        feature_names=feature_names,
        metadata=metadata,
        raw_data=raw_df,
    )


def _prepare_ms_inni_phenotype(
    *,
    raw_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Prepare the paper-aligned relapsing vs progressive MS classification view.

    Target
    ------
    Derived from PHENOTYPE-CODE among MS patients only:

    - 0: relapsing MS
    - 1: progressive MS

    Parameters
    ----------
    raw_df:
        Raw MS INNI DataFrame.

    metadata:
        Dataset metadata.

    Returns
    -------
    dataset_dict:
        Standardized dataset dictionary.
    """

    # Work on a copy so the returned raw_data remains unchanged.
    df = raw_df.copy()

    # Restrict to MS patients with relapsing or progressive phenotype labels.
    df = df[
        (df[MS_INNI_GROUP_COL] == 1)
        & (df[MS_INNI_PHENOTYPE_COL].isin([1, 2]))
    ].copy()

    # Predict progressive vs relapsing MS using a derived binary target.
    feature_names = list(MS_INNI_PHENOTYPE_FEATURES)
    target_col = "progressive_ms"

    # Code progressive MS as 1 and relapsing MS as 0.
    df[target_col] = (df[MS_INNI_PHENOTYPE_COL] == 2).astype(int)

    # Coerce clinical, MRI, treatment, and target columns to numeric values.
    df = _coerce_ms_inni_numeric_columns(df, feature_names + [target_col])

    # Keep all MS phenotype cases. Missing feature values remain in X so the
    # downstream preprocessing pipeline can handle them consistently.
    model_df = df[feature_names + [target_col]].copy()

    X = model_df[feature_names]
    y = model_df[target_col].astype(int)

    metadata = _augment_ms_inni_metadata(
        metadata=metadata,
        target_column=target_col,
        target_definition=(
            "Derived from PHENOTYPE-CODE among MS patients only: "
            "0=relapsing_ms, 1=progressive_ms"
        ),
        positive_class="progressive_ms",
        negative_class="relapsing_ms",
        population_filter="multiple_sclerosis_patients_only",
        class_mapping={
            0: "relapsing_ms",
            1: "progressive_ms",
        },
        class_distribution=y.value_counts().sort_index().to_dict(),
        feature_names=feature_names,
        target_handling=(
            "PHENOTYPE-CODE is used only to derive the binary target. The "
            "feature set includes concurrently available MS-only clinical "
            "variables and MRI-derived variables represented in the released CSV."
        ),
        prediction_time_horizon=None,
        longitudinal_prediction=None,
    )

    return _build_dataset_dict(
        df=model_df,
        X=X,
        y=y,
        feature_names=feature_names,
        metadata=metadata,
        raw_data=raw_df,
    )


def _prepare_ms_inni_edss_regression(
    *,
    raw_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Prepare the paper-aligned cross-sectional EDSS regression view.

    Target
    ------
    Observed EDSS score among MS patients with non-missing EDSS.

    Notes
    -----
    This is a cross-sectional prediction task. The target is the observed EDSS
    score in the released dataset, not a future follow-up EDSS score.

    Parameters
    ----------
    raw_df:
        Raw MS INNI DataFrame.

    metadata:
        Dataset metadata.

    Returns
    -------
    dataset_dict:
        Standardized dataset dictionary.
    """

    # Work on MS patients only because EDSS is not applicable to healthy controls.
    df = raw_df.copy()
    df = df[df[MS_INNI_GROUP_COL] == 1].copy()

    # Use continuous EDSS as the regression target. EDSS is not included in X.
    feature_names = list(MS_INNI_EDSS_REGRESSION_FEATURES)
    target_col = MS_INNI_EDSS_COL

    # Coerce model inputs and the continuous target to numeric values.
    df = _coerce_ms_inni_numeric_columns(df, feature_names + [target_col])

    # EDSS must be observed because it is the target. Missing feature values
    # remain in X so the downstream preprocessing pipeline can handle them.
    model_df = df[feature_names + [target_col]].dropna(subset=[target_col]).copy()

    X = model_df[feature_names]
    y = model_df[target_col].astype(float)

    metadata = _augment_ms_inni_metadata(
        metadata=metadata,
        target_column=target_col,
        target_definition="Observed cross-sectional EDSS score",
        positive_class=None,
        negative_class=None,
        population_filter="multiple_sclerosis_patients_only_with_non_missing_edss",
        class_mapping=None,
        class_distribution=None,
        feature_names=feature_names,
        target_handling=(
            "EDSS is used as the continuous regression target and is excluded "
            "from the feature matrix. Rows with missing EDSS are excluded."
        ),
        prediction_time_horizon="cross_sectional",
        longitudinal_prediction=False,
    )

    # Add a compact numeric summary for the continuous regression target.
    metadata["target_summary"] = {
        "count": int(y.shape[0]),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "max": float(y.max()),
    }

    return _build_dataset_dict(
        df=model_df,
        X=X,
        y=y,
        feature_names=feature_names,
        metadata=metadata,
        raw_data=raw_df,
    )


# =============================================================================
# MS INNI helper functions
# =============================================================================

def _coerce_ms_inni_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Coerce selected MS INNI columns to numeric values.

    Parameters
    ----------
    df:
        Input DataFrame.

    columns:
        Columns to convert to numeric.

    Returns
    -------
    converted_df:
        DataFrame with selected columns converted to numeric when present.
    """

    # Work on a copy so caller-owned DataFrames are not modified in place.
    converted_df = df.copy()

    # Convert only columns that are present; missing required columns are handled earlier.
    for col in columns:
        if col in converted_df.columns:
            converted_df[col] = pd.to_numeric(converted_df[col], errors="coerce")

    return converted_df


def _augment_ms_inni_metadata(
    *,
    metadata: dict[str, Any],
    target_column: str,
    target_definition: str,
    positive_class: str | None,
    negative_class: str | None,
    population_filter: str,
    class_mapping: dict[int, str] | None,
    class_distribution: dict[int, int] | None,
    feature_names: list[str],
    target_handling: str,
    prediction_time_horizon: str | None,
    longitudinal_prediction: bool | None,
) -> dict[str, Any]:
    """
    Add MS-specific modeling metadata.

    Parameters
    ----------
    metadata:
        Base metadata dictionary from the catalog.

    target_column:
        Name of the target column returned in the modeling DataFrame.

    target_definition:
        Human-readable target definition.

    positive_class:
        Name of the positive class for binary classification tasks.

    negative_class:
        Name of the negative class for binary classification tasks.

    population_filter:
        Description of which rows are included in the modeling view.

    class_mapping:
        Mapping from numeric class values to class labels.

    class_distribution:
        Class counts for binary classification tasks.

    feature_names:
        Task-specific feature columns included in the returned feature matrix.

    target_handling:
        Human-readable description of how source target columns are handled.

    prediction_time_horizon:
        Time horizon represented by the target, if applicable.

    longitudinal_prediction:
        Whether the target represents a future longitudinal outcome, if applicable.

    Returns
    -------
    updated_metadata:
        Metadata dictionary with MS-specific task information.
    """

    # Copy the catalog metadata so task-specific updates do not mutate the input.
    updated_metadata = dict(metadata)

    # Add task, target, class, feature-group, and interpretation details.
    updated_metadata.update(
        {
            "target_column": target_column,
            "target_definition": target_definition,
            "positive_class": positive_class,
            "negative_class": negative_class,
            "population_filter": population_filter,
            "class_mapping": class_mapping,
            "class_distribution": class_distribution,
            "feature_names": feature_names,
            "target_handling": target_handling,
            "prediction_time_horizon": prediction_time_horizon,
            "longitudinal_prediction": longitudinal_prediction,
            "source_repository": "San Raffaele Open Research Data Repository",
            "source_record": "hnsppf3k2p",
            "source_version": 1,
            "feature_groups": {
                "demographic_site": [
                    col for col in MS_INNI_DEMOGRAPHIC_SITE_FEATURES
                    if col in feature_names
                ],
                "clinical": [
                    col
                    for col in [
                        MS_INNI_DISEASE_DURATION_COL,
                        MS_INNI_EDSS_COL,
                        MS_INNI_DMT_COL,
                    ]
                    if col in feature_names
                ],
                "mri_derived": [
                    col for col in MS_INNI_MRI_FEATURES
                    if col in feature_names
                ],
            },
            "dmt_mapping": {
                0: "no_dmt",
                1: "medium_efficacy_dmt",
                2: "high_efficacy_dmt",
            },
            "phenotype_mapping": {
                0: "healthy_control_or_not_applicable",
                1: "relapsing_ms",
                2: "progressive_ms",
            },
            "group_mapping": {
                0: "healthy_control",
                1: "multiple_sclerosis",
            },
        }
    )

    return updated_metadata


# =============================================================================
# Standardized output builder
# =============================================================================

def _standardize_feature_names(
    *,
    X: pd.DataFrame,
    feature_names: list[Any],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Convert feature names to plain Python strings and keep X columns aligned.

    This prevents downstream sklearn ColumnTransformer errors caused by numpy
    string objects, pandas extension string types, or mixed column-label types.
    """

    # Convert all feature names to plain Python strings.
    feature_names_str = [str(name) for name in feature_names]

    # Keep the feature matrix column labels aligned with feature_names.
    X = X.copy()
    X.columns = feature_names_str

    return X, feature_names_str


def _build_dataset_dict(
    *,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    feature_names: list[str],
    metadata: dict[str, Any],
    raw_data: Any,
) -> dict[str, Any]:
    """
    Build a standardized dataset dictionary.

    Parameters
    ----------
    df:
        Full modeling DataFrame, including the target column.

    X:
        Feature matrix.

    y:
        Target vector or target DataFrame.

    feature_names:
        Feature column names.

    metadata:
        Dataset metadata.

    raw_data:
        Original raw dataset object.

    Returns
    -------
    dataset_dict:
        Standardized dataset dictionary.
    """

    # Standardize feature names to plain Python strings and align X columns.
    X, feature_names = _standardize_feature_names(
        X=X,
        feature_names=feature_names,
    )

    # Rebuild df so its feature columns are aligned with X and feature_names.
    df = pd.concat([X, y], axis=1)

    # Also keep metadata feature names standardized when present.
    metadata = dict(metadata)
    if "feature_names" in metadata:
        metadata["feature_names"] = [str(name) for name in metadata["feature_names"]]

    return {
        "df": df,
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "metadata": metadata,
        "data_keys": [
            "df",
            "X",
            "y",
            "feature_names",
            "metadata",
            "raw_data",
        ],
        "raw_data": raw_data,
    }