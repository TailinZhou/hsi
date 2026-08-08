# Balrog Benchmark — Running Guide

## Overview

Balrog is a multi-game environment benchmark containing 6 text-game environments that evaluate agents in sequential decision-making scenarios. Each task runs multiple episodes (observe -> act -> reward -> repeat), and the average progression score is used as the reward.

**Supported environments:**

| Environment | Description | Default episodes |
|-------------|-------------|------------------|
| `babyai` | BabyAI grid-world navigation and item manipulation | 10 |
| `nle` | NetHack dungeon exploration | 5 |
| `minihack` | MiniHack puzzle rooms | 5 |
| `crafter` | Crafter survival and synthesis game | 10 |
| `textworld` | TextWorld interactive text adventure | 10 |
| `babaisai` | BabaIsAI rule-based reasoning puzzle | 3 |

---

## Option 1: Docker (Recommended)

Build everything in one step on WSL + Docker Desktop, avoiding the hassle of installing dependencies manually.

### 1. Build the Docker image

From the project root directory:

```bash
docker build -f Dockerfile.balrog -t wolverine-balrog .
```

The build takes about 10-15 minutes (NLE compiles NetHack source, which is slow). The build includes:
- System dependencies (build-essential, cmake, ncurses, etc.)
- Python packages for all 6 game environments
- Boxoban level data (for MiniHack)
- TextWorld game data

### 2. Run the container

```bash
# Enter the container interactively
docker run -it --rm \
    -v $(pwd):/workspace \
    -w /workspace \
    --env-file .env \
    wolverine-balrog

# Run evolution inside the container
python main.py
```

**Parameter notes:**
- `-v $(pwd):/workspace` — mount the project directory so evolution results are written to the host
- `--env-file .env` — inject environment variables such as API keys
- `--rm` — automatically remove the container on exit

### 3. Run in the background

```bash
# Start in the background, log output to a file
docker run -d --name balrog-run \
    -v $(pwd):/workspace \
    -w /workspace \
    --env-file .env \
    wolverine-balrog \
    python main.py

# View logs
docker logs -f balrog-run

# Stop
docker stop balrog-run && docker rm balrog-run
```

### 4. Install only a subset of environments

If you do not need all 6 environments, you can customize the build. Edit `Dockerfile.balrog` and comment out the environments you do not need:

```dockerfile
# Only run BabyAI, comment out the others
RUN pip install git+https://github.com/BartekCupial/Minigrid.git@...
# RUN pip install nle
# RUN pip install git+https://github.com/balrog-ai/minihack.git@...
# RUN pip install crafter
# RUN pip install textworld
# RUN pip install git+https://github.com/nacloos/baba-is-ai
```

Then set the corresponding `suite` in `config.yaml`.

---

## Option 2: Local installation

### 1. Install core dependencies

```bash
pip install numpy gym==0.23.0 gymnasium>=1.2.0 setuptools wheel
```

### 2. Install environments on demand

You do not need to install all environments — only the ones you intend to test. Uninstalled environments raise `ImportError` at runtime and do not affect the others.

**BabyAI** (recommended as a starting point):
```bash
pip install git+https://github.com/BartekCupial/Minigrid.git@cf73dd148e51276bd675a37e3bfb0bf2b42329b2
```

**NLE (NetHack)**:
```bash
# Requires system dependencies: build-essential cmake libncurses5-dev zlib1g-dev libbz2-dev
sudo apt-get install build-essential cmake libncurses5-dev zlib1g-dev libbz2-dev
pip install nle
```

> **Note**: If you plan to run MiniHack, do not install nle separately — just install MiniHack directly (see below). The two packages conflict.

**MiniHack**:
```bash
sudo apt-get install -y build-essential cmake autoconf libtool bison flex zlib1g-dev libbz2-dev libncurses5-dev libncursesw5-dev
pip install git+https://github.com/balrog-ai/minihack.git@3ecb6da4eadcac7a4dbc5f8e801ccf636e34d7ec
```

> **Note: nle / balrog-nle package conflict**
>
> minihack depends on `balrog-nle` (0.9.0), which installs files into the `nle/` directory and registers environments with **old gym**. If you previously installed `pip install nle` (1.2.0, registered with gymnasium) separately, both packages write to the same `nle/` directory, and the later `balrog-nle` install overwrites the files, so `import nle` actually runs the 0.9.0 code.
>
> **Correct approach**: do not install `nle` separately — just install minihack (it pulls in `balrog-nle` automatically). At the code level, `nle_env.py` already uses try/except to handle both registration styles (tries gymnasium first, falls back to gym).

**Crafter**:
```bash
pip install crafter
```

**TextWorld**:
```bash
sudo apt-get install libffi-dev
pip install textworld
# You also need to download the game data:
python scripts/download_balrog_data.py
```

**BabaIsAI**:
```bash
pip install git+https://github.com/nacloos/baba-is-ai
```

---

## Configuration

Copy `benchmark_config_goal/balrog/config.yaml` to the project root:

```bash
cp benchmark_config_goal/balrog/config.yaml config.yaml
```

Key configuration options:

```yaml
llm:
  model: "GLM-4.7"
  temperature: 1

benchmark:
  type: "balrog"
  suite: "babyai"        # Select the environment(s) to test
  dev_ratio: 1.0         # 1.0 = use all for dev, do not split a test set
```

**Available `suite` values:**

| suite value | Included environments |
|-------------|-----------------------|
| `"babyai"` | BabyAI only |
| `"nle"` | NetHack only |
| `"babyai-crafter"` | BabyAI + Crafter |
| `"all"` | All 6 environments |

Separate multiple environment names with hyphens `-`.

---

## Running

```bash
python main.py
```

### Verify the installation

Before a real run, quickly verify with Python:

```bash
cd src/

# Verification 1: import and registration
python -c "
from benchmark.balrog.evaluator import BalrogEvaluator
from benchmark.evaluators.registry import get_evaluator
e = get_evaluator('balrog')
tasks = e.load_tasks(suite='babyai')
print(f'OK: {len(tasks)} babyai tasks loaded')
for t in tasks:
    print(f'  {t.task_id}')
"

# Verification 2: whether environment packages are installed
python -c "
envs = {
    'babyai': 'minigrid',
    'nle': 'nle',
    'minihack': 'minihack',
    'crafter': 'crafter',
    'textworld': 'textworld',
    'babaisai': 'baba',
}
for name, mod in envs.items():
    try:
        __import__(mod)
        print(f'  {name}: installed')
    except ImportError:
        print(f'  {name}: NOT installed')
"

# Verification 3: create an environment and run one step (using babyai as an example)
python -c "
from benchmark.balrog.config import BalrogConfig
from benchmark.balrog.environments import make_env

cfg = BalrogConfig(env_names=['babyai'])
config_dict = cfg.to_hyperagents_config()
env = make_env('babyai', 'BabyAI-MixedTrainLocal-v0/goto', config_dict)
obs, info = env.reset(seed=42)
print('Observation keys:', list(obs.keys()))
print('Text preview:', obs.get('text', {}).get('long_term_context', '')[:200])
"
```

---

## Output structure

```
evolution_results/balrog/
└── run_{timestamp}/
    ├── repo/              # Git repository (one commit per iteration)
    ├── best_agent/        # Best agent version
    ├── context.json       # Evolution context (resumable)
    └── eval_logs/         # Evaluation logs
        ├── iter_000/      # Evaluation results for iteration 0
        │   └── eval_000_*.json
        └── iter_001/
            └── eval_000_*.json
```

---

## Reward calculation

- Each `(env_name, task_name)` runs `num_episodes` episodes.
- Each episode computes a `progression` (0.0 ~ 1.0, representing completion progress).
- A single task's score = the average progression across all episodes.
- The overall `reward = sum(task_progressions) / num_tasks`.

---

## FAQ

**Q: NLE compilation fails during Docker build**
A: NLE compiles NetHack from source. Ensure `build-essential cmake libncurses5-dev zlib1g-dev libbz2-dev` is installed. Under the Docker option these are already included in the Dockerfile.

**Q: I only want to test BabyAI — do I need to install all dependencies?**
A: No. Under Docker, edit the Dockerfile to comment out unneeded environments; locally, only install `minigrid`. Set `suite: "babyai"` in `config.yaml`.

**Q: TextWorld reports that game files cannot be found**
A: You need to download the TextWorld game data: `python scripts/download_balrog_data.py`. Under Docker this is downloaded automatically. If you do not need TextWorld, simply remove it from the suite.

**Q: gym and gymnasium version conflict**
A: Balrog's environment wrapper `GymV21CompatibilityV0` handles gym v21/v26 compatibility. Installing `gym==0.23.0` + `gymnasium>=1.2.0` is sufficient.

**Q: BabaIsAI's pygame reports errors on a headless server**
A: Set the environment variable `SDL_VIDEODRIVER=dummy` or `SDL_VIDEODRIVER=offscreen`. Under Docker there is no GUI and this does not affect evaluation (only text observations are used).
