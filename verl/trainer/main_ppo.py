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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
import requests
from verl.trainer.main_ppo_eco import _SEARCH_RE, _extract_question, _extract_info_raw

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0., eval_metrics=False) -> None:
        import os
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.eval_metrics = eval_metrics
        self.floor = float(os.environ.get("ECO_FLOOR", "0.3"))
        self.use_cf = os.environ.get("EVAL_USE_CF", "1") == "1"
        self.cf_url = os.environ.get("CF_JUDGE_URL", "http://127.0.0.1:8001/judge_batch")
        self.last_metrics = {}
        self._cf_session = requests.Session()
        self._cf_session.trust_env = False

    def _judge_batch(self, items):
        if not items:
            return []
        import time as _t
        payload = {"items": items}
        for attempt in range(3):
            try:
                resp = self._cf_session.post(self.cf_url, json=payload, timeout=600)
                resp.raise_for_status()
                return resp.json()["results"]
            except Exception as e:
                print(f"[baseline-eval cf_judge] request failed (attempt {attempt+1}/3): {e}", flush=True)
                _t.sleep(min(30, 5 * (attempt + 1)))
        return None

    def __call__(self, data: DataProto):
        """Compute EM reward; validation additionally logs performance/trust/cost metrics."""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        rows = []
        cf_reqs = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            response_str = self.tokenizer.decode(valid_response_ids)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)
            answer = qa_em.extract_solution(sequences_str)
            correct = (answer is not None) and bool(qa_em.em_check(answer, ground_truth['target']))
            n_search = len(_SEARCH_RE.findall(response_str))
            row = {
                'i': i,
                'valid_response_length': valid_response_length,
                'seq': sequences_str,
                'ground_truth': ground_truth,
                'data_source': data_source,
                'score': score,
                'answer': answer,
                'correct': correct,
                'n_search': n_search,
            }
            rows.append(row)
            if self.eval_metrics and self.use_cf and correct and n_search > 0:
                gold = ground_truth['target']
                gold = list(gold) if isinstance(gold, (list, tuple, np.ndarray)) else [str(gold)]
                cf_reqs.append((len(rows) - 1, {
                    "question": _extract_question(sequences_str),
                    "info": _extract_info_raw(sequences_str),
                    "gold": [str(g) for g in gold],
                }))

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        cf_map = {}
        if cf_reqs:
            results = self._judge_batch([r[1] for r in cf_reqs])
            if results is not None:
                for (ridx, _), res in zip(cf_reqs, results):
                    cf_map[ridx] = res

        if self.eval_metrics and rows:
            n_total = len(rows)
            n_correct = n_nosearch = n_oversearch = n_correct_searched = 0
            sum_trust = sum_trust_searched = sum_search = 0.0
            cf_true = cf_hall = cf_amb = 0
            for ridx, row in enumerate(rows):
                trust = 0.0
                if row['correct']:
                    n_correct += 1
                    if row['n_search'] == 0:
                        trust = 1.0
                    elif row['n_search'] > 0:
                        n_correct_searched += 1
                        cf_res = cf_map.get(ridx)
                        if cf_res is None or not cf_res.get("injected", False):
                            trust = self.floor
                            cls = 'amb'
                        else:
                            changed = qa_em.normalize_answer(row['answer']) != qa_em.normalize_answer(cf_res["ans_cf"])
                            if changed:
                                trust, cls = 1.0, 'true'
                            elif cf_res.get("em_cf", 0) == 1:
                                trust, cls = self.floor, 'hall'
                            else:
                                trust, cls = 0.6, 'amb'
                        if cls == 'true': cf_true += 1
                        elif cls == 'hall': cf_hall += 1
                        elif cls == 'amb': cf_amb += 1
                        sum_trust_searched += trust
                sum_trust += trust
                sum_search += row['n_search']
                if row['n_search'] == 0:
                    n_nosearch += 1
                if row['n_search'] > 1:
                    n_oversearch += 1

            em = n_correct / n_total
            trust_at_correct = sum_trust / max(1, n_correct)
            trust_at_searched_correct = sum_trust_searched / max(1, n_correct_searched)
            search_mean = sum_search / n_total
            nosearch_rate = n_nosearch / n_total
            oversearch_rate = n_oversearch / n_total
            cf_tot = cf_true + cf_hall + cf_amb
            hall_rate = cf_hall / max(1, cf_tot)
            self.last_metrics = {
                'perf/em': float(em),
                'trust/trust_at_correct': float(trust_at_correct),
                'trust/trust_at_searched_correct': float(trust_at_searched_correct),
                'trust/cf_true': float(cf_true),
                'trust/cf_hall': float(cf_hall),
                'trust/cf_amb': float(cf_amb),
                'trust/hall_rate': float(hall_rate),
                'cost/search_per_query': float(search_mean),
                'cost/nosearch_rate': float(nosearch_rate),
                'cost/oversearch_rate': float(oversearch_rate),
            }
            cf_str = (f"cf[TRUE={cf_true} HALL={cf_hall} AMB={cf_amb} hall_rate={hall_rate:.2f}]") if cf_tot else "cf[none]"
            print(f"[BASE-VAL] PERF em={em:.3f} | TRUST trust@correct={trust_at_correct:.3f} "
                  f"trust@searched={trust_at_searched_correct:.3f} {cf_str} | COST search/q={search_mean:.3f} "
                  f"nosearch={nosearch_rate:.3f} oversearch={oversearch_rate:.3f}")

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
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
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, eval_metrics=True)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
