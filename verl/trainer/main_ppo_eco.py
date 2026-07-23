# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Search-G1 training entry built on the Search-R1/veRL runtime.

The primary training path uses two representation readouts served by
``probe_server.py``. The answer-commit readout estimates answer sensitivity to
evidence deletion, and the prompt-state readout estimates closed-book sufficiency.
Their product after the retrieval-necessity gate forms the representation-based
grounding bonus. A bounded cost penalizes repeated response-side search markers,
and retrieved observations are masked by the rollout/trainer code.

The same reward manager also exposes counterfactual-document metrics during
evaluation.  Those metrics are monitoring signals and are not required for the
representation-reward training path.
"""

from verl import DataProto
import torch
import re
import numpy as np
import requests
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

_SEARCH_RE = re.compile(r"<search>")
_INFO_END_RE = re.compile(r"</information>")
_INFO_BLOCK_RE = re.compile(r"<information>(.*?)</information>", re.DOTALL)
_QUESTION_RE = re.compile(r"Question:\s*(.+?)\s*(?:<\|im_end\|>|\n|$)", re.DOTALL)


def _causal_trust(solution_str: str, floor: float = 0.3) -> float:
    """Fallback temporal-ordering heuristic when the counterfactual judge is unavailable."""
    searches = re.findall(r"<search>(.*?)</search>", solution_str, re.DOTALL)
    info_end = [m.end() for m in _INFO_END_RE.finditer(solution_str)]
    ans = solution_str.rfind("<answer>")
    if not searches or not info_end or ans == -1:
        return floor
    if ans <= info_end[-1]:
        return floor
    return 0.7


def _extract_question(solution_str: str) -> str:
    m = _QUESTION_RE.search(solution_str)
    return m.group(1).strip() if m else ""


def _extract_info_raw(solution_str: str) -> str:
    blocks = _INFO_BLOCK_RE.findall(solution_str)
    return " ".join(b.strip() for b in blocks)


class EcoRewardManager():
    """Search-G1 reward and evaluation manager.

    Core environment variables:
      ECO_FLOOR        : float, D_trust for PARAM_HALL (default 0.3)
      ECO_ALPHA        : float, cost discount strength (default 0.3)
      ECO_NO_SELF      : "1" to fix alpha (ablation)
      ECO_COST_ONLY    : "1" -> D_trust≡1 (ablation)
      ECO_TRUST_ONLY   : "1" -> no cost factor (ablation)
      CF_JUDGE_URL     : counterfactual judge endpoint
      ECO_USE_CF       : "1" (default) use online counterfactual; "0" use temporal fallback
    """

    BONUS = 0.2
    PENALTY = 0.1
    MAX_SEARCH = 2     # norm: searching 2 times already = full cost (strong 1-search preference)
    ALPHA_INIT = 0.5   # stronger cost: 2 searches -> factor 0.5 (was 0.875 at alpha=0.25)
    ALPHA_MAX = 0.7
    ALPHA_MIN = 0.5    # one-way tighten: never relax below init (was 0.1, which weakened cost)
    M_PERIOD = 25
    THETA_HIGH = 0.34
    THETA_LOW = 0.28
    THETA_D = 0.60
    EMA_M = 0.9

    # counterfactual D_trust levels
    D_TRUE_TOOL = 1.0    # answer changed under fake docs
    D_AMBIGUOUS = 0.6    # answer didn't change but also wrong on fake docs
    # D_PARAM_HALL = floor (answer unchanged & still correct on fake docs)

    def __init__(self, tokenizer, num_examine, format_score=0., reward_mode='eco'):
        import os
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.reward_mode = reward_mode
        self.floor = float(os.environ.get("ECO_FLOOR", "0.3"))
        self.alpha = float(os.environ.get("ECO_ALPHA", str(self.ALPHA_INIT)))
        self.alpha_max = float(os.environ.get("ECO_ALPHA_MAX", str(self.ALPHA_MAX)))
        self.alpha_min = float(os.environ.get("ECO_ALPHA_MIN", str(self.ALPHA_MIN)))
        self.no_self = os.environ.get("ECO_NO_SELF", "0") == "1"
        self.cost_only = os.environ.get("ECO_COST_ONLY", "0") == "1"
        self.trust_only = os.environ.get("ECO_TRUST_ONLY", "0") == "1"
        self.use_cf = os.environ.get("ECO_USE_CF", "1") == "1"
        self.cf_url = os.environ.get("CF_JUDGE_URL", "http://127.0.0.1:8001/judge_batch")
        # ---- trust-centric reward variants (anti-hallucination) ----
        # legacy    : reward = d_trust * cost_factor  (trust as a soft multiplier, original)
        # gate      : PARAM_HALL (memorized) counted as failure -> 0  (Method 1: hard grounding gate)
        # additive  : reward = w_trust*d_trust + w_perf*correct - w_cost*cost  (Method 2: trust as main term)
        # cf_primary: counterfactual IS the primary signal, PARAM_HALL penalized < 0  (Method 3)
        self.trust_variant = os.environ.get("ECO_TRUST_VARIANT", "legacy")
        # which trust signal feeds the 'balanced' grounding bonus during TRAINING:
        #   cf      : counterfactual injection (fake entity)         [judge /judge_batch]
        #   lexical : answer string appears in retrieved docs        [no judge needed]
        #   nli     : evidence entails the answer (faithfulness)     [judge /judge_faithfulness]
        #   removal : redact evidence and re-answer (leave-one-out)  [judge /judge_removal]
        # Validation/eval always uses cf injection for a comparable cross-method metric.
        self.trust_signal = os.environ.get("ECO_TRUST_SIGNAL", "cf")
        self.w_trust = float(os.environ.get("ECO_W_TRUST", "1.0"))
        self.w_perf = float(os.environ.get("ECO_W_PERF", "0.3"))
        self.w_cost = float(os.environ.get("ECO_W_COST", "0.02"))
        self.hall_penalty = float(os.environ.get("ECO_HALL_PENALTY", "0.5"))
        # 'balanced' variant: perf floor + additive grounding bonus + cost (no perf-trust artifact)
        self.bal_nosearch_bonus = float(os.environ.get("ECO_BAL_NOSEARCH_BONUS", "0.5"))
        self.bal_ground = float(os.environ.get("ECO_BAL_GROUND", "0.6"))
        # scale of grounding bonus on WRONG answers (anti-collapse; <<1 so wrongness
        # never becomes profitable, yet >0 keeps within-group variance in all-wrong groups)
        self.wrong_ground_scale = float(os.environ.get("ECO_WRONG_GROUND_SCALE", "0.15"))
        # Reward floor for a TRIED-BUT-WRONG answer. MUST satisfy the ordering
        #   -PENALTY (no answer / invalid)  <  wrong_floor  <  1 - w_cost (correct)
        # so that "answer something" always beats "answer nothing" (kills the slacker
        # pit) while "answer correctly" still dominates. Default 0.05 (PENALTY=0.1).
        self.wrong_floor = float(os.environ.get("ECO_WRONG_FLOOR", "0.05"))
        # ---- Probe-into-base (option B): make the CORRECT base itself depend on the
        # commit probe (flip), so a truly-grounded correct answer scores higher than a
        # memorized/lucky correct one. When ECO_PROBE_BASE>0, the correct base becomes
        #   base = (1 - probe_base) + probe_base * flip      (in [1-pb, 1])
        # i.e. a fully-grounded correct answer keeps base=1.0, a non-grounded (flip=0)
        # correct answer is discounted to (1-probe_base). Default 0 = legacy fixed 1.0.
        self.probe_base = float(os.environ.get("ECO_PROBE_BASE", "0.0"))
        # Optional probe-branch cost multiplier. Default off to preserve existing
        # probe experiments; set ECO_PROBE_ALPHA_COST=1 when alpha should scale
        # the explicit W_COST term in _score_probe.
        self.probe_alpha_cost = os.environ.get("ECO_PROBE_ALPHA_COST", "0") == "1"
        # ---- Search-G1 representation signal (ECO_TRUST_SIGNAL=probe) ----
        self.probe_url = os.environ.get("PROBE_URL", "http://127.0.0.1:8002/judge_probe")
        self.probe_gate = os.environ.get("ECO_PROBE_GATE", "1") == "1"
        # Probe is PART OF THE REWARD DEFINITION. It must never be silently skipped
        # (that would train on EM/cost-only and silently corrupt the experiment).
        # Strategy B (unattended-safe): block-and-retry the probe call essentially
        # forever (capped backoff), so a transient probe-server outage/restart just
        # PAUSES training instead of either crashing it or polluting it. Set
        # ECO_PROBE_MAX_WAIT (seconds) to bound the wait; <=0 means wait indefinitely.
        self.probe_max_wait = float(os.environ.get("ECO_PROBE_MAX_WAIT", "0"))  # 0 = forever
        self.probe_timeout = float(os.environ.get("ECO_PROBE_TIMEOUT", "600"))
        self.probe_backoff_cap = float(os.environ.get("ECO_PROBE_BACKOFF_CAP", "30"))
        self._probe_session = requests.Session()
        self._probe_session.trust_env = False
        self.ema_em = 0.0
        self.ema_trust = 0.0
        self._init = False
        self._calls = 0
        # counterfactual outcome counters (for logging)
        self._cf_session = requests.Session()
        self._cf_session.trust_env = False
        self._metrics_acc = None
        print(f"[ECO-INIT] mode={reward_mode} variant={self.trust_variant} signal={self.trust_signal} "
              f"floor={self.floor} alpha={self.alpha} alpha_range=[{self.alpha_min},{self.alpha_max}] "
              f"bal_ground={self.bal_ground} "
              f"w_trust={self.w_trust} w_perf={self.w_perf} w_cost={self.w_cost} hall_penalty={self.hall_penalty} "
              f"use_cf={self.use_cf} cf_url={self.cf_url} no_self={self.no_self} "
              f"cost_only={self.cost_only} trust_only={self.trust_only} "
              f"probe_alpha_cost={self.probe_alpha_cost} "
              f"probe_max_wait={self.probe_max_wait} probe_backoff_cap={self.probe_backoff_cap}")

    def reset_metrics(self):
        self._metrics_acc = {
            'n_total': 0,
            'n_correct': 0,
            'n_correct_searched': 0,
            'n_nosearch': 0,
            'n_oversearch': 0,
            'sum_trust': 0.0,
            'sum_trust_correct': 0.0,
            'sum_trust_searched': 0.0,
            'sum_search': 0.0,
            'sum_reward': 0.0,
            'cf_true': 0,
            'cf_hall': 0,
            'cf_amb': 0,
        }

    def get_metrics(self):
        acc = self._metrics_acc
        if not isinstance(acc, dict) or acc.get('n_total', 0) <= 0:
            return getattr(self, 'last_metrics', None)
        n_total = max(1, acc['n_total'])
        n_correct = max(1, acc['n_correct'])
        n_correct_searched = max(1, acc['n_correct_searched'])
        cf_tot = acc['cf_true'] + acc['cf_hall'] + acc['cf_amb']
        return {
            'perf/em': float(acc['n_correct'] / n_total),
            'trust/trust_at_correct': float(acc['sum_trust_correct'] / n_correct),
            'trust/trust_at_searched_correct': float(acc['sum_trust_searched'] / n_correct_searched),
            'trust/trust_all_mean': float(acc['sum_trust'] / n_total),
            'trust/cf_true': float(acc['cf_true']),
            'trust/cf_hall': float(acc['cf_hall']),
            'trust/cf_amb': float(acc['cf_amb']),
            'trust/hall_rate': float(acc['cf_hall'] / max(1, cf_tot)),
            'cost/search_per_query': float(acc['sum_search'] / n_total),
            'cost/nosearch_rate': float(acc['n_nosearch'] / n_total),
            'cost/oversearch_rate': float(acc['n_oversearch'] / n_total),
        }

    # ---- online counterfactual call ----
    def _judge_batch(self, items, path="/judge_batch"):
        """POST items to the judge service at `path`. Returns results list or None.
        path: /judge_batch (cf injection) | /judge_faithfulness (nli) | /judge_removal."""
        if not items:
            return []
        base = self.cf_url.rsplit('/', 1)[0]
        url = base + path
        payload = {"items": items}
        import time as _t
        for attempt in range(10):
            try:
                resp = self._cf_session.post(url, json=payload, timeout=600)
                resp.raise_for_status()
                return resp.json()["results"]
            except Exception as e:
                print(f"[cf_judge] {path} request failed (attempt {attempt+1}/10): {e}", flush=True)
                _t.sleep(min(30, 5 * (attempt + 1)))
        print(f"[cf_judge] {path} UNAVAILABLE -> temporal-ordering fallback", flush=True)
        return None

    def _probe_call(self, items):
        """POST rollouts to the Search-G1 probe server. Blocks and retries until
        the probe server responds (Strategy B), so a transient outage/restart PAUSES
        training rather than corrupting it. Returns the results list. Only gives up
        (RuntimeError) if ECO_PROBE_MAX_WAIT>0 is exceeded."""
        if not items:
            return []
        import time as _t
        attempt = 0
        waited = 0.0
        while True:
            attempt += 1
            try:
                resp = self._probe_session.post(self.probe_url, json={"items": items},
                                                timeout=self.probe_timeout)
                resp.raise_for_status()
                if attempt > 1:
                    print(f"[probe] recovered after {attempt-1} retries (~{waited:.0f}s waited); "
                          f"resuming with valid trust reward", flush=True)
                return resp.json()["results"]
            except Exception as e:
                backoff = min(self.probe_backoff_cap, 4 * attempt)
                print(f"[probe] UNAVAILABLE (attempt {attempt}, waited ~{waited:.0f}s): {e} "
                      f"-> PAUSING this batch, retry in {backoff:.0f}s "
                      f"(probe is required; NOT training on trust=0)", flush=True)
                _t.sleep(backoff)
                waited += backoff
                if self.probe_max_wait > 0 and waited >= self.probe_max_wait:
                    raise RuntimeError(
                        f"[probe] still UNAVAILABLE after {waited:.0f}s (> ECO_PROBE_MAX_WAIT="
                        f"{self.probe_max_wait}s). Aborting to avoid corrupting the run.")

    def _score_probe(self, row, pr):
        """Search-G1 reward (applies to all rollouts, including wrong ones, so
        all-wrong zero-variance groups still get a gradient toward grounded behavior).
            ground = (1 - boundary) * flip      # gate: reward grounding only when UNKNOWN
            reward = perf_floor + bal_ground*ground - w_cost*cost
        """
        # Outcome ordering is enforced by keeping invalid < wrong < correct.
        # Search cost is applied only within the correct searched branch.
        n_search = row['n_search']
        correct = row['correct']
        # Slacker pit guard: no valid answer extracted == worst outcome.
        if row['answer'] is None:
            return -self.PENALTY, 0.0, None
        flip = float(pr['flip']) if pr else 0.0
        boundary = float(pr['boundary']) if pr else 0.0
        gate = (1.0 - boundary) if self.probe_gate else 1.0
        norm_cost = min(1.0, max(0.0, (n_search - 1)) / max(1.0, self.MAX_SEARCH - 1))
        cls = 'true' if flip >= 0.5 else 'hall'
        # no-search branch: reward answering directly *when the model knows it* (boundary high)
        if n_search == 0:
            if not correct:
                # tried (gave an answer) but wrong & didn't search: small positive,
                # strictly above the no-answer floor (-PENALTY) so "answer something"
                # always beats "answer nothing".
                return self.wrong_floor, 0.0, None
            return 1.0 + self.bal_nosearch_bonus * boundary, flip, 'true'
        # CORRECT base: legacy fixed 1.0, OR (option B) probe-modulated so a grounded
        # correct answer outscores a memorized/lucky one within the same group.
        if correct and self.probe_base > 0.0:
            base = (1.0 - self.probe_base) + self.probe_base * flip
        else:
            base = 1.0 if correct else 0.0
        if correct:
            # CORRECT: full grounding bonus, cost charged here (where we can afford it).
            cost_weight = self.w_cost * (self.alpha if self.probe_alpha_cost else 1.0)
            reward = base + self.bal_ground * gate * flip - cost_weight * norm_cost
        else:
            # WRONG but tried & searched: a small POSITIVE floor plus a tiny grounding
            # signal for within-group variance. NO cost penalty (cost must never push a
            # tried answer below the no-answer floor and recreate the slacker pit).
            # Strictly: wrong_floor < correct_min, and wrong_floor > -PENALTY (no-answer).
            reward = self.wrong_floor + self.bal_ground * gate * flip * self.wrong_ground_scale
        return reward, flip, cls

    def _train_ground(self, row, cf_res):
        """Grounding signal in [0,1] for the 'balanced' bonus during training, dispatched
        by ECO_TRUST_SIGNAL. Returns (ground, cf_class) where cf_class in true/amb/hall."""
        s = self.trust_signal
        if s == 'lexical':
            info = _extract_info_raw(row['seq'])
            ans = qa_em.normalize_answer(row['answer'] or "")
            ok = bool(ans) and ans in qa_em.normalize_answer(info)
            return (1.0 if ok else 0.0), ('true' if ok else 'hall')
        if s == 'nli':
            if cf_res is None or 'supported' not in cf_res:
                return 0.5, 'amb'
            ok = int(cf_res.get('supported', 0)) == 1
            return (1.0 if ok else 0.0), ('true' if ok else 'hall')
        if s == 'removal':
            if cf_res is None or not cf_res.get('removed', False):
                return 0.5, 'amb'
            changed = qa_em.normalize_answer(row['answer']) != qa_em.normalize_answer(cf_res.get('ans_cf', ''))
            return (1.0 if changed else 0.0), ('true' if changed else 'hall')
        # default 'cf': counterfactual injection
        d = self._cf_trust(row['answer'], cf_res, row['seq'])
        if d >= 0.99:
            return 1.0, 'true'
        if abs(d - self.floor) < 1e-6:
            return 0.0, 'hall'
        return 0.5, 'amb'

    def _cf_trust(self, answer_real, cf_res, solution_str):
        """Map counterfactual outcome to D_trust."""
        if cf_res is None or not cf_res.get("injected", False):
            return _causal_trust(solution_str, floor=self.floor)  # fallback
        changed = qa_em.normalize_answer(answer_real) != qa_em.normalize_answer(cf_res["ans_cf"])
        if changed:
            return self.D_TRUE_TOOL          # really used the tool
        if cf_res.get("em_cf", 0) == 1:
            return self.floor                # PARAM_HALL: ignored docs, still correct
        return self.D_AMBIGUOUS

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print = {}

        # ---- Phase 1: decode + extract per-sample info; collect CF requests ----
        rows = []
        cf_reqs = []          # list of (row_idx, {question, info, gold})
        for i in range(len(data)):
            di = data[i]
            prompt_ids = di.batch['prompts']
            plen = prompt_ids.shape[-1]
            valid_plen = di.batch['attention_mask'][:plen].sum()
            valid_prompt = prompt_ids[-valid_plen:]
            resp_ids = di.batch['responses']
            valid_rlen = di.batch['attention_mask'][plen:].sum()
            valid_resp = resp_ids[:valid_rlen]
            seq_str = self.tokenizer.decode(torch.cat((valid_prompt, valid_resp)))
            prompt_str = self.tokenizer.decode(valid_prompt)
            resp_str = self.tokenizer.decode(valid_resp)
            gt = di.non_tensor_batch['reward_model']['ground_truth']
            src = di.non_tensor_batch['data_source']

            answer = qa_em.extract_solution(seq_str)
            # Count actions emitted by the policy only.  The instruction prompt contains
            # literal markup examples that must not contribute to Search/Q or search cost.
            n_search = len(_SEARCH_RE.findall(resp_str))
            correct = (answer is not None) and bool(qa_em.em_check(answer, gt['target']))
            row = {'i': int(i), 'valid_rlen': int(valid_rlen), 'seq': seq_str, 'gt': gt,
                   'src': src, 'answer': answer, 'n_search': n_search, 'correct': correct,
                   'prompt': prompt_str, 'resp': resp_str}
            rows.append(row)

            # Correct & searched rollouts ALWAYS get a cf judge call for MONITORING
            # (trust@correct & hall_rate must use the same yardstick across all methods).
            # The reward itself still uses ECO_TRUST_SIGNAL (probe/lexical/cf/etc).
            if (self.reward_mode in ('eco', 'eval') and self.use_cf and not self.cost_only
                    and correct and n_search > 0):
                gold = gt['target']
                gold = list(gold) if isinstance(gold, (list, tuple, np.ndarray)) else [str(gold)]
                cf_reqs.append((len(rows) - 1,
                                {"question": _extract_question(seq_str),
                                 "info": _extract_info_raw(seq_str),
                                 "gold": [str(g) for g in gold],
                                 "answer": str(answer) if answer is not None else ""}))

        # ---- Phase 2: batch judge (endpoint depends on mode/signal) ----
        cf_map = {}
        if cf_reqs:
            if self.reward_mode == 'eval' or self.trust_signal == 'cf':
                path = '/judge_batch'
            elif self.trust_signal == 'nli':
                path = '/judge_faithfulness'
            elif self.trust_signal == 'removal':
                path = '/judge_removal'
            else:
                path = '/judge_batch'
            results = self._judge_batch([r[1] for r in cf_reqs], path)
            if results is not None:
                for (ridx, _), res in zip(cf_reqs, results):
                    cf_map[ridx] = res

        # ---- Phase 2b: Search-G1 readouts (all rollouts, including wrong ones) ----
        probe_map = {}
        if self.reward_mode == 'eco' and self.trust_signal == 'probe':
            pres = self._probe_call([{"prompt": r['prompt'], "response": r['resp']}
                                     for r in rows])
            if pres is not None:
                probe_map = {idx: pres[idx] for idx in range(len(pres))}

        # ---- Phase 3: score ----
        # MONITORING uses cf judge uniformly (same yardstick for all methods).
        # REWARD uses method-specific logic (probe / lexical / cf / etc).
        n_total = n_correct = n_nosearch = n_oversearch = n_correct_searched = 0
        sum_trust = sum_trust_searched = sum_trust_correct = sum_search = sum_reward = 0.0
        cf_true = cf_hall = cf_amb = 0
        for ridx, row in enumerate(rows):
            # --- reward (method-specific, unchanged) ---
            if self.reward_mode == 'eco':
                if self.trust_signal == 'probe':
                    score, _reward_trust, _reward_cls = self._score_probe(row, probe_map.get(ridx))
                else:
                    score, _reward_trust, _reward_cls = self._score_row(row, cf_map.get(ridx))
            else:
                # Validation/evaluation returns plain EM for comparability.
                score = qa_em.compute_score_em(row['seq'], row['gt'], format_score=self.format_score)

            # --- monitoring trust (ALWAYS from cf judge, unified across methods) ---
            cls = None
            if row['correct'] and row['n_search'] == 0:
                trust = 1.0  # answered correctly without searching → trivially grounded
            elif row['correct'] and row['n_search'] > 0:
                if self.cost_only:
                    trust = 1.0
                elif self.use_cf:
                    trust = self._cf_trust(row['answer'], cf_map.get(ridx), row['seq'])
                    cls = ('true' if trust >= 0.99 else
                           ('hall' if abs(trust - self.floor) < 1e-6 else 'amb'))
                else:
                    trust = _causal_trust(row['seq'], floor=self.floor)
            else:
                trust = 0.0

            if cls == 'true': cf_true += 1
            elif cls == 'hall': cf_hall += 1
            elif cls == 'amb': cf_amb += 1
            reward_tensor[row['i'], row['valid_rlen'] - 1] = score

            n_total += 1
            sum_reward += score
            sum_search += row['n_search']
            sum_trust += trust
            if row['correct']:
                n_correct += 1
                sum_trust_correct += trust
                if row['n_search'] > 0:
                    n_correct_searched += 1
                    sum_trust_searched += trust
            if row['n_search'] == 0:
                n_nosearch += 1
            if row['n_search'] > 1:
                n_oversearch += 1

            if row['src'] not in already_print:
                already_print[row['src']] = 0
            if already_print[row['src']] < self.num_examine:
                already_print[row['src']] += 1
                print(row['seq'])

        if n_total > 0:
            em = n_correct / n_total
            # FIX: trust@correct must average trust over CORRECT rollouts only.
            # Previously divided sum_trust (over ALL rollouts) by n_correct, which
            # blew up >1 (e.g. 4.9) when EM collapsed -- a misleading monitor signal.
            trust_at_correct = sum_trust_correct / max(1, n_correct)
            trust_at_searched_correct = sum_trust_searched / max(1, n_correct_searched)
            trust_all_mean = sum_trust / n_total
            search_mean = sum_search / n_total
            nosearch_rate = n_nosearch / n_total
            oversearch_rate = n_oversearch / n_total
            reward_mean = sum_reward / n_total
            tag = 'TRUST-VAL' if self.reward_mode == 'eval' else ('EM-VAL' if self.reward_mode == 'em' else 'TRUST')

            if self._metrics_acc is not None:
                acc = self._metrics_acc
                acc['n_total'] += n_total
                acc['n_correct'] += n_correct
                acc['n_correct_searched'] += n_correct_searched
                acc['n_nosearch'] += n_nosearch
                acc['n_oversearch'] += n_oversearch
                acc['sum_trust'] += sum_trust
                acc['sum_trust_correct'] += sum_trust_correct
                acc['sum_trust_searched'] += sum_trust_searched
                acc['sum_search'] += sum_search
                acc['sum_reward'] += sum_reward
                acc['cf_true'] += cf_true
                acc['cf_hall'] += cf_hall
                acc['cf_amb'] += cf_amb

            if self.reward_mode == 'eco':
                if not self._init:
                    self.ema_em, self.ema_trust = em, trust_at_correct
                    self._init = True
                else:
                    m = self.EMA_M
                    self.ema_em = m * self.ema_em + (1 - m) * em
                    self.ema_trust = m * self.ema_trust + (1 - m) * trust_at_correct
                self._calls += 1
                if (not self.no_self) and self._calls % self.M_PERIOD == 0:
                    old = self.alpha
                    if self.ema_em > self.THETA_HIGH and self.ema_trust > self.THETA_D:
                        self.alpha = min(self.alpha_max, self.alpha + 0.05)
                    elif self.ema_em < self.THETA_LOW:
                        self.alpha = max(self.alpha_min, self.alpha - 0.05)
                    if abs(self.alpha - old) > 1e-9:
                        print(f"[TRUST-ALPHA] step~{self._calls} ema_em={self.ema_em:.3f} "
                              f"ema_trust={self.ema_trust:.3f} alpha {old:.2f}->{self.alpha:.2f}")

            cf_tot = cf_true + cf_hall + cf_amb
            hall_rate = cf_hall / max(1, cf_tot)
            cf_str = (f"cf[TRUE={cf_true} HALL={cf_hall} AMB={cf_amb} "
                      f"hall_rate={hall_rate:.2f}]") if cf_tot else "cf[none]"
            self.last_metrics = {
                'perf/em': float(em),
                'trust/trust_at_correct': float(trust_at_correct),
                'trust/trust_at_searched_correct': float(trust_at_searched_correct),
                'trust/trust_all_mean': float(trust_all_mean),
                'trust/cf_true': float(cf_true),
                'trust/cf_hall': float(cf_hall),
                'trust/cf_amb': float(cf_amb),
                'trust/hall_rate': float(hall_rate),
                'cost/search_per_query': float(search_mean),
                'cost/nosearch_rate': float(nosearch_rate),
                'cost/oversearch_rate': float(oversearch_rate),
            }
            print(f"[{tag}] call~{self._calls} | PERF em={em:.3f} "
                  f"| TRUST trust@correct={trust_at_correct:.3f} trust@searched={trust_at_searched_correct:.3f} {cf_str} "
                  f"| COST search/q={search_mean:.3f} nosearch={nosearch_rate:.3f} oversearch={oversearch_rate:.3f} alpha={self.alpha:.2f} "
                  f"| reward={reward_mean:.3f}")

        return reward_tensor

    def _score_row(self, row, cf_res):
        """Returns (reward, trust_value, cf_class)."""
        answer = row['answer']
        n_search = row['n_search']
        correct = row['correct']
        v = self.trust_variant
        if answer is None:
            return 0.0, 0.0, None

        # ---- no-search branch (no tool use -> nothing to hallucinate) ----
        if n_search == 0:
            if not correct:
                return -self.PENALTY, 0.0, None
            if v == 'additive':
                return self.w_perf + self.w_trust, 1.0, None
            if v == 'cf_primary':
                return 1.0, 1.0, None
            if v == 'balanced':
                return 1.0 + self.bal_nosearch_bonus, 1.0, None
            return 1.0 + self.BONUS, 1.0, None   # legacy / gate

        # ---- searched & wrong ----
        if not correct:
            return 0.0, 0.0, None

        # ---- searched & correct: counterfactual D_trust ----
        if self.cost_only:
            d_trust, cls = 1.0, None
        elif self.use_cf:
            d_trust = self._cf_trust(answer, cf_res, row['seq'])
            cls = ('true' if d_trust >= 0.99 else
                   ('hall' if abs(d_trust - self.floor) < 1e-6 else 'amb'))
        else:
            d_trust, cls = _causal_trust(row['seq'], floor=self.floor), None

        norm_cost = min(1.0, max(0.0, (n_search - 1)) / max(1.0, self.MAX_SEARCH - 1))
        cost_factor = 1.0 if self.trust_only else (1.0 - self.alpha * norm_cost)

        if v == 'gate':
            # Method 1: answer-by-memory (PARAM_HALL) is treated as failure.
            if cls == 'hall':
                return 0.0, d_trust, cls
            return d_trust * cost_factor, d_trust, cls

        if v == 'additive':
            # Method 2: trust is the dominant additive term, not a multiplier.
            reward = self.w_trust * d_trust + self.w_perf * 1.0 - self.w_cost * norm_cost
            return reward, d_trust, cls

        if v == 'cf_primary':
            # Method 3: the counterfactual outcome IS the primary reward signal.
            #   TRUE_TOOL (answer changed under fake docs) -> +1.0
            #   PARAM_HALL (ignored docs, still correct)   -> negative penalty
            #   AMB                                        -> small positive (floor)
            if cls == 'true':
                base = 1.0
            elif cls == 'hall':
                base = -self.hall_penalty
            else:
                base = self.floor
            return base - self.w_cost * norm_cost, d_trust, cls

        if v == 'balanced':
            # perf floor (never zero a correct answer) + additive grounding bonus + cost.
            # Grounding comes from ECO_TRUST_SIGNAL (cf / lexical / nli / removal):
            #   ground in {1.0 (grounded), 0.5 (ambiguous), 0.0 (not grounded / memorized)}.
            # PARAM_HALL / not-grounded simply gets no bonus, but is NOT penalized.
            g, gcls = self._train_ground(row, cf_res)
            return 1.0 + self.bal_ground * g - self.w_cost * norm_cost, g, gcls

        # legacy (default): trust as soft multiplier
        return d_trust * cost_factor, d_trust, cls


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }
    global_pool_id = 'global_pool'
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {k: global_pool_id for k in [Role.ActorRollout, Role.Critic, Role.RefPolicy]}

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = EcoRewardManager(tokenizer=tokenizer, num_examine=0, reward_mode='eco')
    val_reward_fn = EcoRewardManager(tokenizer=tokenizer, num_examine=1, reward_mode='eval')

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
