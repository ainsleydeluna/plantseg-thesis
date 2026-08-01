"""Statistical analysis layer — deterministic paired inference (A3a) + bootstrap/BCa (A3b).

Implements docs/STATISTICAL_ANALYSIS_CONTRACT.md sections 1-10 and 12: A3a covers sections 1-7 and
section 10 ingestion/alignment; A3b adds sections 8-9, 8.7, 12.2, 12.3 and 12.4 — the deterministic
paired bootstrap and BCa, the pooled dataset-mIoU estimand, the E3->E6 non-inferiority check, the
E6-KD trigger and the three-file statistics artifact.

Import-time behaviour is side-effect free: importing `src.stats` does not inspect Git, read an
artifact, invoke a SciPy test, create a file, initialise an RNG, hash a contract, or write a report.
"""

from .align import (METRIC_ALL_CLASS, METRIC_DISEASE_ONLY, METRIC_MIOU_C, AlignmentError,
                    PairedVector, align_runs, align_vectors)
from .artifact import (ARTIFACT_STATUSES, INPUT_ARTIFACT_FIELDS, INTEGRITY_CHECKS, NPZ_SUFFIXES,
                       OFFICIALITY_WARNINGS, PINNED_ENVIRONMENT, RUN_ID_PATTERN,
                       TIMESTAMP_FORMAT, TOP_LEVEL_KEYS, ArtifactError, ArtifactInputs,
                       build_family, contract_sha256, derive_artifact_status,
                       officiality_warnings, serialize_a3a, software_environment_block,
                       verify_statistics_artifact, write_statistics_artifact)
from .bootstrap import (ANALYSIS_ID_OFFICIAL, ANALYSIS_ID_SMOKE, FROZEN_A3A_FIELDS,
                        FROZEN_TASK_IDS, FROZEN_TASK_MATRIX, OBSERVED_SOURCES, ONE_SIDED_LOWER,
                        PRODUCTION_B, ROOT_SEED, STATISTIC_NAMES, TASK_WARNINGS, TWO_SIDED,
                        BcaOutcome, BootstrapError, BootstrapTaskResult, DescriptiveScalars,
                        SeedMaterial, TaskSpec, assert_a3a_source_schema, bca_interval,
                        canonical_seed_bytes, comparison_observed, derive_seed,
                        descriptive_scalars, draw_replicate_indices, expected_comparison_ids,
                        make_generator, reconstruct_seed_material, run_bootstrap_task,
                        scalar_callables, scalar_statistic)
from .noninferiority import (E6KD_TRIGGER_THRESHOLD, PRIMARY_MARGIN, SENSITIVITY_MARGINS,
                             E6KDResult, NonInferiorityResult, PooledStages, SensitivityDecision,
                             build_e6_kd, build_non_inferiority, densify,
                             pooled_miou_from_totals, relative_retention)
from .corruption_protocol import (CorruptionEntry, CorruptionProtocol, CorruptionProtocolError,
                                  build_official_corruption_grid, load_corruption_protocol,
                                  official_corruption_grid)
from .ingest import (EvaluationRun, EvaluationRunIdentity, IngestError, PerImageRecord,
                     PerImageSufficientStats, Policy, load_run, verify_run_manifest)
from .robustness import CorruptionGrid, MiouCVector, RobustnessError, align_miou_c, assemble_miou_c
from .tests import (ALPHA, CANONICAL_COMPARISON_IDS, CohensDzResult, ComparisonResult, HolmFamily,
                    HolmMember, RankBiserialResult, StatsError, TTestResult, WilcoxonResult,
                    cohens_dz, engineering_shifts, hodges_lehmann, holm_audit, holm_family,
                    paired_t_test, rank_biserial, run_comparison, wilcoxon_test)

__all__ = [
    # ingestion + policy
    "Policy", "IngestError", "EvaluationRun", "EvaluationRunIdentity", "PerImageRecord",
    "PerImageSufficientStats", "load_run", "verify_run_manifest",
    # alignment
    "AlignmentError", "PairedVector", "align_runs", "align_vectors",
    "METRIC_DISEASE_ONLY", "METRIC_ALL_CLASS", "METRIC_MIOU_C",
    # robustness / mIoU-C
    "RobustnessError", "CorruptionGrid", "MiouCVector", "assemble_miou_c", "align_miou_c",
    # frozen corruption vocabulary (A3b-0)
    "CorruptionProtocol", "CorruptionEntry", "CorruptionProtocolError",
    "load_corruption_protocol", "build_official_corruption_grid", "official_corruption_grid",
    # tests + effect sizes
    "StatsError", "ALPHA", "CANONICAL_COMPARISON_IDS", "WilcoxonResult", "TTestResult",
    "RankBiserialResult", "CohensDzResult", "ComparisonResult", "HolmMember", "HolmFamily",
    "wilcoxon_test", "paired_t_test", "rank_biserial", "cohens_dz", "hodges_lehmann",
    "engineering_shifts", "run_comparison", "holm_audit", "holm_family",
    # bootstrap + BCa (A3b)
    "BootstrapError", "TaskSpec", "FROZEN_TASK_MATRIX", "FROZEN_TASK_IDS", "STATISTIC_NAMES",
    "TASK_WARNINGS", "OBSERVED_SOURCES", "TWO_SIDED", "ONE_SIDED_LOWER", "ROOT_SEED",
    "PRODUCTION_B", "ANALYSIS_ID_OFFICIAL", "ANALYSIS_ID_SMOKE", "SeedMaterial",
    "canonical_seed_bytes", "derive_seed", "make_generator", "reconstruct_seed_material",
    "draw_replicate_indices", "BcaOutcome", "bca_interval", "BootstrapTaskResult",
    "run_bootstrap_task", "scalar_statistic", "scalar_callables", "DescriptiveScalars",
    "descriptive_scalars", "comparison_observed", "expected_comparison_ids",
    "FROZEN_A3A_FIELDS", "assert_a3a_source_schema",
    # pooled estimand, non-inferiority, E6-KD (A3b)
    "PRIMARY_MARGIN", "SENSITIVITY_MARGINS", "E6KD_TRIGGER_THRESHOLD", "densify",
    "pooled_miou_from_totals", "PooledStages", "SensitivityDecision", "NonInferiorityResult",
    "E6KDResult", "relative_retention", "build_non_inferiority", "build_e6_kd",
    # statistics artifact (A3b)
    "ArtifactError", "ArtifactInputs", "TOP_LEVEL_KEYS", "INPUT_ARTIFACT_FIELDS",
    "INTEGRITY_CHECKS", "NPZ_SUFFIXES", "OFFICIALITY_WARNINGS", "ARTIFACT_STATUSES",
    "PINNED_ENVIRONMENT", "RUN_ID_PATTERN", "TIMESTAMP_FORMAT", "serialize_a3a",
    "software_environment_block", "officiality_warnings", "derive_artifact_status",
    "contract_sha256", "build_family", "write_statistics_artifact",
    "verify_statistics_artifact",
]
