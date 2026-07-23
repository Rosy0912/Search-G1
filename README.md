# Search-G1: Representation-Based Intrinsic Rewards for Grounded Search Agents

<p align="center">
  <img src="assets/pipeline.png" width="100%" alt="Search-G1 pipeline">
</p>

Search-G1 augments correctness-first reinforcement learning with intrinsic shaping signals
read from policy representations. It combines two lightweight, periodically refreshed
readouts:

- **Evidence sensitivity** estimates whether the answer changes when retrieved evidence is
  deleted, conditional on the retained reasoning prefix.
- **Retrieval necessity** is the complement of the policy's predicted closed-book
  sufficiency at the initial prompt state.

Their product gates the searched-answer grounding bonus, while a bounded repeated-marker
cost discourages unnecessarily long search trajectories. Training uses masked GRPO so that
gradients are applied only to policy-generated reasoning, search markers, and answers;
retrieved text is context, not an optimization target. As policy representations change,
Search-G1 refits both readouts on fresh trajectories, forming a
measurement--credit--adaptation loop.

## Method

For a rollout, the answer-commit readout returns evidence-deletion sensitivity `d`, and the
prompt-state readout returns closed-book sufficiency `b`. Retrieval necessity is `1 - b`,
giving the shaping score

```text
grounding = d * (1 - b)
search_cost = min(1, max(parsed_response_search_markers - 1, 0))
```

Here `d * (1 - b)` is a multiplicative shaping score, not a calibrated joint probability or
a causal certificate of evidence use. Correct searched answers receive an additive
grounding bonus and repeated-marker cost.
Correct direct answers receive a closed-book-sufficiency bonus. Wrong and invalid outputs
remain below correct outputs, preserving task-outcome ordering while creating useful
within-group credit when EM is otherwise constant.

The paper configuration uses the hidden state at the final generated answer token for the
evidence-sensitivity readout and the hidden state at the end of the initial agent prompt for
the closed-book-sufficiency readout. Evidence-sensitivity labels are obtained by deleting
retrieved evidence, retaining the rollout's reasoning and search history, and
deterministically re-decoding the answer. The label therefore measures answer-stage
sensitivity under this intervention, not the total causal effect of retrieval.

## Paper-Reported Results

The following headline values are transcribed from the manuscript. Task columns report EM
on each benchmark. TC, D-Inv, and Search/Q in these tables are evaluated on NQ. Search/Q is
the mean number of response-side, parser-observed `<search>` markers—not a count of physical
retrieval calls or a compute unit. For trained methods, entries are means over four
independent runs; Base and Prompted Search are untrained point estimates.
Methods whose original pipelines differ from this environment are adapted under the shared
model, retriever, update budget, and validation rule; these rows are controlled
reimplementations rather than claims about untouched official pipelines.

### Qwen2.5-3B-Instruct

| Method | NQ EM ↑ | HotpotQA EM ↑ | 2Wiki EM ↑ | MuSiQue EM ↑ | TC ↑ | D-Inv ↓ | Search/Q ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.142 | 0.153 | 0.212 | 0.029 | - | - | 0.000 |
| Prompted Search | 0.250 | 0.199 | 0.238 | 0.057 | 0.930 | 0.069 | 2.215 |
| Search-R1 | 0.352 | 0.248 | 0.267 | 0.087 | 0.917 | 0.049 | 3.281 |
| GiGPO | 0.422 | 0.296 | 0.330 | 0.126 | 0.935 | 0.047 | 3.131 |
| AgentPRM | 0.380 | 0.293 | 0.328 | 0.112 | 0.958 | 0.042 | 3.340 |
| GPT-5 Judge | 0.402 | 0.280 | 0.324 | 0.101 | 0.946 | 0.038 | 3.238 |
| ARPO | 0.389 | 0.273 | 0.304 | 0.109 | 0.924 | 0.055 | 2.679 |
| IGPO | 0.414 | 0.292 | 0.355 | 0.137 | 0.934 | 0.056 | 3.028 |
| Search-E1 | 0.437 | **0.314** | **0.372** | 0.142 | 0.939 | 0.054 | 2.980 |
| SAAS | 0.428 | 0.301 | 0.355 | 0.138 | 0.951 | 0.045 | 2.122 |
| β-GRPO | 0.419 | 0.295 | 0.344 | 0.129 | 0.948 | 0.047 | 2.227 |
| KbPO | 0.431 | 0.305 | 0.358 | 0.140 | 0.955 | 0.043 | 2.160 |
| **Search-G1** | **0.453** | 0.306 | 0.368 | **0.143** | **0.976** | **0.034** | **2.094** |

### Qwen2.5-7B-Instruct

| Method | NQ EM ↑ | HotpotQA EM ↑ | 2Wiki EM ↑ | MuSiQue EM ↑ | TC ↑ | D-Inv ↓ | Search/Q ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.192 | 0.183 | 0.250 | 0.031 | - | - | 0.000 |
| Prompted Search | 0.349 | 0.299 | 0.235 | 0.058 | 0.936 | 0.061 | 2.171 |
| Search-R1 | 0.393 | 0.370 | 0.414 | 0.146 | 0.932 | 0.046 | 3.090 |
| GiGPO | 0.464 | 0.416 | 0.436 | 0.189 | 0.947 | 0.043 | 2.930 |
| AgentPRM | 0.455 | 0.405 | 0.430 | 0.205 | 0.963 | 0.037 | 3.080 |
| GPT-5 Judge | 0.470 | 0.410 | 0.440 | 0.215 | 0.952 | 0.033 | 2.980 |
| ARPO | 0.462 | 0.398 | 0.420 | 0.208 | 0.931 | 0.050 | 2.430 |
| IGPO | 0.480 | 0.425 | 0.455 | 0.235 | 0.942 | 0.050 | 2.760 |
| Search-E1 | 0.495 | 0.440 | 0.465 | 0.245 | 0.947 | 0.048 | 2.720 |
| SAAS | 0.486 | 0.432 | 0.458 | 0.239 | 0.958 | 0.038 | 2.050 |
| β-GRPO | 0.478 | 0.424 | 0.448 | 0.231 | 0.953 | 0.040 | 2.164 |
| KbPO | 0.490 | 0.436 | 0.462 | 0.242 | 0.962 | 0.036 | 2.106 |
| **Search-G1** | **0.512** | **0.450** | **0.475** | **0.254** | **0.981** | **0.028** | **1.860** |

<p align="center">
  <img src="assets/estimator_auc_four_panel.png" width="94%" alt="Evidence-sensitivity estimator fidelity">
</p>

The evidence-sensitivity readout reaches AUC 0.959/0.971 on NQ and 0.871/0.914 on
2WikiMultiHopQA for the 3B/7B backbones, respectively (four-run SD at most 0.014). These AUCs
refer to held-out evidence-deletion-sensitivity labels.

### Statistical scope of the headline comparison

On NQ, hierarchical seed-by-question bootstrap inference resamples training runs and paired
questions, with Holm correction over EM, TC, D-Inv, and Search/Q within each
comparator/scale family. Against the strongest baseline for each metric, the manuscript
reports Holm-adjusted p-values:

| Scale | ΔEM | ΔTC | D-Inv benefit | ΔSearch/Q |
|---|---:|---:|---:|---:|
| 3B | +0.016 [-0.012, +0.044]; p=.320 | +0.018 [+0.004, +0.032]; p=.032 | +0.004 [-0.003, +0.011]; p=.320 | -0.028 [-0.114, +0.058]; p=.520 |
| 7B | +0.017 [-0.006, +0.040]; p=.180 | +0.018 [+0.006, +0.030]; p=.016 | +0.005 [+0.001, +0.009]; p=.048 | -0.190 [-0.286, -0.094]; p=.008 |

Accordingly, the supported claim is **competitive EM with stronger operational grounding**;
the manuscript does not claim statistically superior EM over the strongest baseline. On
2WikiMultiHopQA, the four-run means provide a second-dataset grounding check:

| Scale | Method | EM ↑ | TC ↑ | D-Inv ↓ | Search/Q ↓ |
|---|---|---:|---:|---:|---:|
| 3B | Prompted Search | 0.238 | 0.877 | 0.085 | 2.951 |
| 3B | Search-R1 | 0.267 | 0.913 | 0.058 | 3.281 |
| 3B | **Search-G1** | **0.368** | **0.972** | **0.031** | **2.812** |
| 7B | Prompted Search | 0.235 | 0.887 | 0.079 | 2.892 |
| 7B | Search-R1 | 0.414 | 0.932 | 0.050 | 3.090 |
| 7B | **Search-G1** | **0.475** | **0.978** | **0.026** | **2.495** |

### Human, shortcut, and trajectory-cost checks

A confirmatory same-question human evaluation compares Search-G1 with GPT-5 Judge on 270
matched correct-searched pairs. Search-G1 improves support-when-correct by +0.048
[+0.026, +0.070] and full support by +0.060 [+0.032, +0.088], while reducing the
correct-but-unsupported rate by 0.037 [0.017, 0.057]. This paired analysis is separate from the
manuscript's descriptive six-method table, which uses distinct 180-example pools per method.

For the 3B readout, scalar shortcut controls reach AUC 0.536–0.603, the strongest output-logit
baseline reaches 0.731/0.638 on NQ/2Wiki, and the full representation reaches 0.959/0.871.
After exact matching on search count and response length, the representation retains
0.934/0.842 AUC. These checks argue against the tested simple shortcuts; they do not rule out
all possible probe gaming.

The NQ token audit localizes the efficiency result to response-side trajectories:

| Method (3B) | EM ↑ | Search/Q ↓ | Generated tokens/Q ↓ | Observation tokens/Q ↓ |
|---|---:|---:|---:|---:|
| Base | 0.142 | 0.000 | 30.0 | 0.0 |
| Prompted Search | 0.250 | 2.215 | 120.0 | 500.0 |
| Search-R1 | 0.352 | 3.281 | 149.0 | 510.0 |
| **Search-G1** | **0.453** | **2.094** | **84.8** | **468.1** |

Relative to Search-R1, this is about 43% fewer policy-generated tokens and 16% fewer total
response-side tokens under the matched interaction budget. It is a trajectory/token-cost
result, not a wall-clock, FLOP, monetary-cost, or universal executed-retrieval claim.

## Repository Map

| Component | Main files |
|---|---|
| Calibration rollouts and counterfactual labels | `ts_probe_dump.py` |
| Readout fitting and grouped validation | `ts_probe_train.py` |
| Hot-reloaded representation evaluator | `probe_server.py` |
| Necessity-gated shaping and repeated-marker cost | `verl/trainer/main_ppo_eco.py` |
| Masked multi-turn rollout generation | `search_r1/llm_agent/generation.py` |
| Masked GRPO optimization | `verl/trainer/ppo/`, `verl/workers/actor/` |
| Reward--policy co-evolution | `ts_coevolve.py` |
| Counterfactual grounding evaluation | `cf_judge_server.py` |
| Dense retrieval service | `search_r1/search/retrieval_server.py` |
| Analysis and plotting | `analysis/` |

## Installation

The reference environment uses Python 3.10, PyTorch with CUDA, Transformers `<4.48`, and
vLLM `<=0.6.3`. Create a CUDA environment appropriate for your hardware, then install:

```bash
git clone https://github.com/Rosy0912/Search-G1.git
cd Search-G1
pip install -e .
```

`requirements.txt` includes CPU FAISS for portability. Replace it with the matching GPU
FAISS package when running a GPU-resident index. FlashAttention is optional if your hardware
uses the xFormers/SDPA path.

## Data and Retrieval

The QA data follow the Search-R1 parquet schema. This example reserves 300 training-split
questions for calibration and another 300 for validation/model selection:

```bash
python scripts/prepare_qa.py --dataset nq --split train \
  --output data/nq_search/train.parquet \
  --calibration-output data/nq_search/calibration.parquet \
  --calibration-size 300 --calibration-offset 3000 \
  --validation-output data/nq_search/validation.parquet \
  --validation-size 300 --validation-offset 3300
python scripts/prepare_qa.py --dataset nq --split test \
  --output data/nq_search/test.parquet
```

The calibration and validation ranges must not overlap. The preparation script writes both
holdouts separately and removes them from the policy-training parquet, making all three
sets question-disjoint by construction. The validation set is used for model and coefficient
selection; the benchmark test set is reserved for final evaluation.

The manuscript evaluates NQ, HotpotQA, 2WikiMultiHopQA, and MuSiQue. Dataset aliases accepted
by the preparation script are documented in `python scripts/prepare_qa.py --help`.

Search-G1 does not redistribute the Wikipedia corpus, FAISS index, model checkpoints, or
benchmark datasets. Once a compatible corpus and index are available, start the retriever:

```bash
INDEX_PATH=/path/to/e5_Flat.index \
CORPUS_PATH=/path/to/wiki.jsonl \
bash recipes/start_retriever.sh
```

## Training Workflow

All recipes are ordinary shell scripts and accept environment-variable overrides. They do
not contain cluster-specific paths or scheduler directives.

### 1. Initial representation calibration

```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
DATA_FILE=data/nq_search/calibration.parquet \
DATASET_TAG=nq \
EXPERIMENT_NAME=search_g1 \
bash recipes/calibrate.sh
```

This creates a manifest at `artifacts/probes/live_search_g1/manifest.json`. The paper defaults
(`Xa` final-answer state and `flip` evidence-ablation target) are explicit in the recipe.

### 2. Start the readout server

```bash
EXPERIMENT_NAME=search_g1 bash recipes/start_probe_server.sh
```

### 3. Train Search-G1

```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
TRAIN_FILE=data/nq_search/train.parquet \
VAL_FILE=data/nq_search/validation.parquet \
RETRIEVER_URL=http://127.0.0.1:8000/retrieve \
PROBE_URL=http://127.0.0.1:8002/judge_probe \
EXPERIMENT_NAME=search_g1 \
bash recipes/train_search_g1.sh
```

The recipe exposes the paper defaults `ECO_BAL_GROUND=0.6`,
`ECO_BAL_NOSEARCH_BONUS=0.5`, and `ECO_W_COST=0.02`. The grounding and cost coefficients
were fixed from calibration/validation sweeps rather than selected on a test split; adapted
baselines use the same validation rule and interaction budget.

### 4. Run co-evolution alongside training

Use a separate GPU for calibration and point the controller at the same experiment name:

```bash
EXPERIMENT_NAME=search_g1 \
DATA_FILE=data/nq_search/calibration.parquet \
RETRIEVER_URL=http://127.0.0.1:8000/retrieve \
bash recipes/coevolve.sh
```

The controller only publishes a new evaluator after both readouts have been fitted
successfully; otherwise the server keeps the latest complete round.

## Evaluation

EM and Search/Q evaluation requires the retriever only:

```bash
CKPT=verl_checkpoints/search_g1/actor/global_step_250 \
VAL_FILE=data/nq_search/test.parquet \
RETRIEVER_URL=http://127.0.0.1:8000/retrieve \
bash recipes/evaluate.sh
```

For counterfactual-document grounding metrics, first start the judge and enable it in the
evaluation recipe:

```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct bash recipes/start_cf_judge.sh

ECO_USE_CF=1 \
CF_JUDGE_URL=http://127.0.0.1:8001/judge_batch \
CKPT=/path/to/checkpoint \
VAL_FILE=data/nq_search/test.parquet \
bash recipes/evaluate.sh
```

- **TC (higher is better)** is graded counterfactual-document trust consistency over
  originally correct trajectories. Correct direct answers receive 1; correct searched
  answers receive 1/0.6/0.3 for True/Ambiguous/Hall outcomes under document replacement.
- **D-Inv (lower is better)** is the fraction of classified originally correct searched
  trajectories that remain correct and invariant after the document intervention.
- **Search/Q (lower is better)** counts parsed response-side `<search>` markers only; markup
  examples in the instruction prompt are excluded. It should not be interpreted as an
  executed-retrieval, latency, or compute measure.

TC and D-Inv are operational and related rather than independent: TC retains the graded
ambiguous outcome and includes all correct trajectories, whereas D-Inv isolates searched
trajectories that remain invariant. They triangulate grounding behavior under a fixed
retriever and intervention design; neither metric alone proves causal evidence use.

## Reproducibility Notes

- Set all framework, sampling, and data-loader seeds for multi-run reporting.
- Keep policy-training, calibration, validation/model-selection, and final test question IDs
  disjoint. Never tune coefficients or select checkpoints on the benchmark test set.
- Keep `--commit-position Xa` and `--commit-label flip` fixed across refresh rounds for the
  paper's evidence-deletion-sensitivity construct.
- Log the exact model checkpoint, corpus/index version, retriever model, dataset split,
  readout manifest, and counterfactual-judge checkpoint for every evaluation.
- The repository currently provides implementation and recipes, but does not bundle the
  paper's checkpoints, benchmark data, FAISS corpus/index, seed-by-item predictions, human
  annotations, or complete reproduction artifacts. The manuscript reports retained final
  aggregates and states that a row-level artifact bundle is planned for release with the
  paper. Generated artifacts remain ignored by Git until such a versioned release is made.

## Acknowledgments

Search-G1 builds on [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and
[veRL](https://github.com/volcengine/verl). Their Apache-2.0 license and attribution are
preserved in `LICENSE` and `Notice.txt`.

## License

Apache License 2.0. See `LICENSE` and `Notice.txt`.
