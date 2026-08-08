"""Terminal-Bench 2 task definitions — 89 terminal tasks."""

from enum import Enum
from typing import Dict, List, Tuple


class TB2Category(str, Enum):
    """Task categories for TB2."""
    SYSTEM_ADMIN = "system_admin"
    SOFTWARE_ENG = "software_eng"
    DATA_SCIENCE = "data_science"
    SECURITY = "security"
    ML_AI = "ml_ai"
    DEBUGGING = "debugging"
    SCIENTIFIC = "scientific"


# (task_id, category, description_keyword)
_TASKS: List[Tuple[str, str, str]] = [
    ("adaptive-rejection-sampler", "scientific", "Adaptive rejection sampling"),
    ("bn-fit-modify", "data_science", "Bayesian network fitting"),
    ("break-filter-js-from-html", "software_eng", "Extract JS from HTML"),
    ("build-cython-ext", "software_eng", "Build Cython extension"),
    ("build-pmars", "software_eng", "Build pMARS Core Wars"),
    ("build-pov-ray", "software_eng", "Build POV-Ray renderer"),
    ("caffe-cifar-10", "ml_ai", "Caffe CIFAR-10 training"),
    ("cancel-async-tasks", "debugging", "Cancel async tasks"),
    ("chess-best-move", "software_eng", "Chess best move"),
    ("circuit-fibsqrt", "scientific", "Circuit Fibonacci sqrt"),
    ("cobol-modernization", "software_eng", "COBOL modernization"),
    ("code-from-image", "ml_ai", "Code generation from image"),
    ("compile-compcert", "software_eng", "Compile CompCert compiler"),
    ("configure-git-webserver", "system_admin", "Configure git web server"),
    ("constraints-scheduling", "data_science", "Constraint scheduling"),
    ("count-dataset-tokens", "data_science", "Count dataset tokens"),
    ("crack-7z-hash", "security", "Crack 7z hash"),
    ("custom-memory-heap-crash", "debugging", "Custom memory heap crash"),
    ("db-wal-recovery", "system_admin", "Database WAL recovery"),
    ("distribution-search", "scientific", "Distribution search"),
    ("dna-assembly", "scientific", "DNA assembly"),
    ("dna-insert", "scientific", "DNA insert"),
    ("extract-elf", "system_admin", "Extract ELF binary"),
    ("extract-moves-from-video", "ml_ai", "Extract moves from video"),
    ("feal-differential-cryptanalysis", "security", "FEAL differential cryptanalysis"),
    ("feal-linear-cryptanalysis", "security", "FEAL linear cryptanalysis"),
    ("filter-js-from-html", "software_eng", "Filter JS from HTML"),
    ("financial-document-processor", "data_science", "Financial document processing"),
    ("fix-code-vulnerability", "security", "Fix code vulnerability"),
    ("fix-git", "debugging", "Fix git repository"),
    ("fix-ocaml-gc", "debugging", "Fix OCaml GC"),
    ("gcode-to-text", "software_eng", "G-code to text"),
    ("git-leak-recovery", "security", "Git leak recovery"),
    ("git-multibranch", "system_admin", "Git multibranch"),
    ("gpt2-codegolf", "ml_ai", "GPT-2 code golf"),
    ("headless-terminal", "system_admin", "Headless terminal"),
    ("hf-model-inference", "ml_ai", "HuggingFace model inference"),
    ("install-windows-3.11", "system_admin", "Install Windows 3.11"),
    ("kv-store-grpc", "software_eng", "KV store gRPC"),
    ("large-scale-text-editing", "software_eng", "Large-scale text editing"),
    ("largest-eigenval", "scientific", "Largest eigenvalue"),
    ("llm-inference-batching-scheduler", "ml_ai", "LLM batch scheduler"),
    ("log-summary-date-ranges", "data_science", "Log summary date ranges"),
    ("mailman", "system_admin", "Mailman mailing list"),
    ("make-doom-for-mips", "software_eng", "Build DOOM for MIPS"),
    ("make-mips-interpreter", "software_eng", "MIPS interpreter"),
    ("mcmc-sampling-stan", "scientific", "MCMC sampling Stan"),
    ("merge-diff-arc-agi-task", "data_science", "Merge diff ARC AGI"),
    ("model-extraction-relu-logits", "security", "Model extraction attack"),
    ("modernize-scientific-stack", "software_eng", "Modernize scientific stack"),
    ("mteb-leaderboard", "ml_ai", "MTEB leaderboard"),
    ("mteb-retrieve", "ml_ai", "MTEB retrieval"),
    ("multi-source-data-merger", "data_science", "Multi-source data merger"),
    ("nginx-request-logging", "system_admin", "Nginx request logging"),
    ("openssl-selfsigned-cert", "security", "OpenSSL self-signed cert"),
    ("overfull-hbox", "software_eng", "Fix overfull hbox LaTeX"),
    ("password-recovery", "security", "Password recovery"),
    ("path-tracing", "scientific", "Path tracing"),
    ("path-tracing-reverse", "scientific", "Path tracing reverse"),
    ("polyglot-c-py", "software_eng", "Polyglot C/Python"),
    ("polyglot-rust-c", "software_eng", "Polyglot Rust/C"),
    ("portfolio-optimization", "data_science", "Portfolio optimization"),
    ("protein-assembly", "scientific", "Protein assembly"),
    ("prove-plus-comm", "scientific", "Prove plus commutativity"),
    ("pypi-server", "system_admin", "PyPI server setup"),
    ("pytorch-model-cli", "ml_ai", "PyTorch model CLI"),
    ("pytorch-model-recovery", "debugging", "PyTorch model recovery"),
    ("qemu-alpine-ssh", "system_admin", "QEMU Alpine SSH"),
    ("qemu-startup", "system_admin", "QEMU startup"),
    ("query-optimize", "data_science", "Query optimization"),
    ("raman-fitting", "scientific", "Raman fitting"),
    ("regex-chess", "software_eng", "Regex chess"),
    ("regex-log", "data_science", "Regex log parsing"),
    ("reshard-c4-data", "data_science", "Reshard C4 data"),
    ("rstan-to-pystan", "scientific", "RStan to PyStan"),
    ("sam-cell-seg", "ml_ai", "SAM cell segmentation"),
    ("sanitize-git-repo", "security", "Sanitize git repo"),
    ("schemelike-metacircular-eval", "software_eng", "Metacircular evaluator"),
    ("sparql-university", "data_science", "SPARQL university query"),
    ("sqlite-db-truncate", "data_science", "SQLite DB truncate"),
    ("sqlite-with-gcov", "system_admin", "SQLite with gcov"),
    ("torch-pipeline-parallelism", "ml_ai", "Torch pipeline parallelism"),
    ("torch-tensor-parallelism", "ml_ai", "Torch tensor parallelism"),
    ("train-fasttext", "ml_ai", "Train fastText model"),
    ("tune-mjcf", "ml_ai", "Tune MJCF model"),
    ("video-processing", "software_eng", "Video processing"),
    ("vulnerable-secret", "security", "Vulnerable secret"),
    ("winning-avg-corewars", "software_eng", "Winning Core Wars avg"),
    ("write-compressor", "software_eng", "Write compressor"),
]

# Build lookup dicts
TASK_IDS: List[str] = [t[0] for t in _TASKS]
TASK_CATEGORY_MAP: Dict[str, str] = {t[0]: t[1] for t in _TASKS}
TASK_DESC_MAP: Dict[str, str] = {t[0]: t[2] for t in _TASKS}

CATEGORY_NAMES: Dict[str, str] = {
    "system_admin": "System Administration",
    "software_eng": "Software Engineering",
    "data_science": "Data Science",
    "security": "Security",
    "ml_ai": "Machine Learning & AI",
    "debugging": "Debugging",
    "scientific": "Scientific Computing",
}


def get_all_task_ids() -> List[str]:
    """Return all 89 TB2 task IDs."""
    return list(TASK_IDS)


def get_task_ids_by_category(category: str) -> List[str]:
    """Return task IDs filtered by category."""
    return [tid for tid, cat in TASK_CATEGORY_MAP.items() if cat == category]


def get_categories() -> List[str]:
    """Return all unique categories."""
    return sorted(set(TASK_CATEGORY_MAP.values()))
