"""
SCST (Self-Critical Sequence Training) for CIDEr optimization
基于 pycocoevalcap 的 CIDEr scorer 实现强化学习训练
"""
import torch
import numpy as np
import re
from collections import defaultdict


class CiderReward:
    """
    用于 SCST 的 CIDEr 奖励计算器
    支持 corpus-level IDF 预计算，解决 batch=1 时 CIDEr 退化为 0 的问题
    """
    def __init__(self, n=4, sigma=6.0, corpus_gts=None):
        from evaluation.pycocoevalcap.cider.cider_scorer import CiderScorer
        self.n = n
        self.sigma = sigma
        self.CiderScorer = CiderScorer
        self.document_frequency = defaultdict(float)
        self.ref_len = 0.0
        self.idf_precomputed = False
        
        if corpus_gts is not None:
            self._precompute_idf(corpus_gts)

    def _precompute_idf(self, corpus_gts):
        """
        从整个语料库预计算 IDF 统计
        :param corpus_gts: list of list of str, 整个训练集的所有 GT captions
        """
        from evaluation.pycocoevalcap.cider.cider_scorer import cook_refs
        
        crefs = []
        for refs in corpus_gts:
            crefs.append(cook_refs(refs))
        
        for refs in crefs:
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                self.document_frequency[ngram] += 1
        
        self.ref_len = np.log(float(len(crefs)))
        self.idf_precomputed = True
        print(f"[CiderReward] IDF precomputed: {len(crefs)} samples, {len(self.document_frequency)} unique n-grams")

    @staticmethod
    def normalize_ground_truths(ground_truths):
        """确保每条样本的 GT 为 list[str]，避免对 str 按字符迭代。"""
        normalized = []
        for refs in ground_truths:
            if isinstance(refs, str):
                normalized.append([refs])
            elif isinstance(refs, (list, tuple)):
                if len(refs) == 0:
                    normalized.append([""])
                elif isinstance(refs[0], str):
                    normalized.append(list(refs))
                elif isinstance(refs[0], (list, tuple)):
                    normalized.append([str(x) for x in refs[0]])
                else:
                    normalized.append([str(x) for x in refs])
            else:
                normalized.append([str(refs)])
        return normalized
        
    def _compute_cider_with_precomputed_idf(self, gts, res):
        """
        使用预计算的 IDF 计算 CIDEr 分数
        """
        from evaluation.pycocoevalcap.cider.cider_scorer import cook_refs, cook_test

        imgIds = list(gts.keys())
        
        crefs = [cook_refs(gts[i]) for i in imgIds]
        ctest = [cook_test(res[i][0]) for i in imgIds]
        
        scores = []
        for test, refs in zip(ctest, crefs):
            vec, norm, length = self._counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = self._counts2vec(ref)
                score += self._sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score)
            score_avg /= len(refs)
            score_avg *= 10.0
            scores.append(score_avg)
        
        return np.mean(np.array(scores)), np.array(scores)

    def _counts2vec(self, cnts):
        """
        使用预计算的 IDF 将 n-gram counts 转换为 TF-IDF 向量
        """
        vec = [defaultdict(float) for _ in range(self.n)]
        length = 0
        norm = [0.0 for _ in range(self.n)]
        
        for (ngram, term_freq) in cnts.items():
            df = np.log(max(1.0, self.document_frequency.get(ngram, 0)))
            n = len(ngram) - 1
            vec[n][ngram] = float(term_freq) * (self.ref_len - df)
            norm[n] += pow(vec[n][ngram], 2)
            if n == 1:
                length += term_freq
        
        norm = [np.sqrt(n) for n in norm]
        return vec, norm, length

    def _sim(self, vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
        """
        计算两个 TF-IDF 向量的余弦相似度
        """
        delta = float(length_hyp - length_ref)
        val = np.array([0.0 for _ in range(self.n)])
        
        for n in range(self.n):
            for (ngram, count) in vec_hyp[n].items():
                val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
            
            if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                val[n] /= (norm_hyp[n] * norm_ref[n])
            
            val[n] *= np.exp(-(delta**2) / (2 * self.sigma**2))
        
        return val
        
    def compute_score(self, gts, res):
        """
        计算 CIDEr 分数
        :param gts: dict, {image_id: [ref1, ref2, ...]}
        :param res: dict, {image_id: [hypothesis]}
        :return: (mean_score, scores_per_image)
        """
        assert gts.keys() == res.keys()
        
        if self.idf_precomputed:
            return self._compute_cider_with_precomputed_idf(gts, res)
        
        imgIds = list(gts.keys())
        
        cider_scorer = self.CiderScorer(n=self.n, sigma=self.sigma)
        
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) > 0
            
            cider_scorer += (hypo[0], ref)
        
        (score, scores) = cider_scorer.compute_score()
        return score, scores
    
    def preprocess_caption(self, caption):
        """预处理 caption，与评估时保持一致"""
        caption = re.sub(r"([_!,'\"()*#:;~])", " ", caption.lower())
        caption = re.sub(r"\s{2,}", " ", caption)
        caption = caption.rstrip("\n").strip(" ")
        return caption
    
    def compute_batch_reward(self, hypotheses, ground_truths):
        """
        计算 batch 内每个样本的 CIDEr 奖励
        :param hypotheses: list of str, 模型生成的描述
        :param ground_truths: list of list of str, 每个样本的多个参考描述
        :return: np.array, 每个样本的 CIDEr 分数
        """
        ground_truths = self.normalize_ground_truths(ground_truths)
        gts = {}
        res = {}
        
        for i, (hypo, refs) in enumerate(zip(hypotheses, ground_truths)):
            hypo_clean = self.preprocess_caption(hypo)
            refs_clean = [self.preprocess_caption(r) for r in refs]
            
            gts[i] = refs_clean
            res[i] = [hypo_clean]
        
        _, scores = self.compute_score(gts, res)
        return scores


def get_scst_ground_truths(train_dataset, image_ids):
    """从数据集按 video key 取该样本的全部 GT captions（与评估一致）。"""
    if isinstance(image_ids, (list, tuple)):
        keys = list(image_ids)
    else:
        keys = [image_ids]
    return [train_dataset.get_captions_by_key(k) for k in keys]


class SCSTTrainer:
    """
    SCST 训练器
    混合交叉熵和强化学习损失
    """
    def __init__(
        self,
        model,
        tokenizer,
        cider_reward,
        temperature=0.8,
        scst_weight=0.5,
        use_baseline=True,
        end_symbols=None,
    ):
        self.tokenizer = tokenizer
        self.cider_reward = cider_reward
        self.temperature = temperature
        self.scst_weight = scst_weight
        self.use_baseline = use_baseline
        self.stop_token_ids = self._build_stop_token_ids(tokenizer, end_symbols)
        # 处理 DDP 包装
        if hasattr(model, 'module'):
            self.model = model.module
        else:
            self.model = model

    @staticmethod
    def _build_stop_token_ids(tokenizer, end_symbols=None):
        end_symbols = end_symbols or ["\n", ".", " ."]
        stop_ids = set()
        for sym in end_symbols:
            ids = tokenizer.encode(sym, add_special_tokens=False)
            if ids:
                stop_ids.add(ids[-1])
        if tokenizer.eos_token_id is not None:
            stop_ids.add(tokenizer.eos_token_id)
        return sorted(stop_ids)
        
    def _generate_step(
        self,
        input_ids,
        inputs_embeds,
        attention_mask,
        max_new_tokens=32,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
    ):
        """
        手动生成循环，使用 inputs_embeds + KV cache
        :return: generated_ids [B, T], log_probs [B, T], seq_mask [B, T]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        llama_model = self.model.llama_model
        stop_ids = torch.tensor(self.stop_token_ids, device=device, dtype=torch.long)

        generated_ids = []
        log_probs_list = []
        seq_mask_list = []
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            if step == 0:
                outputs = llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=None,
                )
            else:
                outputs = llama_model(
                    input_ids=next_token,
                    attention_mask=cur_mask,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            if step == 0:
                cur_mask = attention_mask
            cur_mask = torch.cat(
                [cur_mask, torch.ones(batch_size, 1, device=device, dtype=cur_mask.dtype)],
                dim=1,
            )
            
            if do_sample:
                scaled_logits = logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    probs = probs.masked_fill(indices_to_remove, 0.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                
                next_token = torch.multinomial(probs, num_samples=1)
                step_log_prob = torch.log(probs.clamp(min=1e-8)).gather(1, next_token)
            else:
                scaled_logits = logits
                next_token = torch.argmax(scaled_logits, dim=-1, keepdim=True)
                step_log_prob = torch.log_softmax(scaled_logits, dim=-1).gather(1, next_token)
            
            step_mask = (~finished).float().unsqueeze(1)
            log_probs_list.append(step_log_prob * step_mask)
            seq_mask_list.append(step_mask)
            generated_ids.append(next_token)

            finished = finished | torch.isin(next_token.squeeze(-1), stop_ids)
            if finished.all():
                break
        
        generated_ids = torch.cat(generated_ids, dim=1)
        log_probs = torch.cat(log_probs_list, dim=1)
        seq_mask = torch.cat(seq_mask_list, dim=1)
        
        return generated_ids, log_probs, seq_mask
    
    def generate_with_sampling(
        self,
        input_ids,
        inputs_embeds,
        attention_mask,
        max_new_tokens=32,
        temperature=0.8,
        top_p=0.9,
    ):
        """采样生成"""
        return self._generate_step(
            input_ids, inputs_embeds, attention_mask,
            max_new_tokens, temperature, top_p, do_sample=True,
        )
    
    def generate_greedy(
        self,
        input_ids,
        inputs_embeds,
        attention_mask,
        max_new_tokens=32,
    ):
        """Greedy 生成基线"""
        return self._generate_step(
            input_ids, inputs_embeds, attention_mask,
            max_new_tokens, temperature=1.0, top_p=1.0, do_sample=False,
        )
    
    def decode_batch(self, token_ids):
        """解码 token ids 为文本"""
        captions = []
        for ids in token_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            captions.append(text)
        return captions
    
    def compute_scst_loss(
        self,
        input_ids,
        inputs_embeds,
        attention_mask,
        ground_truths,
        max_new_tokens=32,
        ce_loss=None,
    ):
        """
        计算 SCST 损失
        :param ground_truths: list[list[str]]，每个样本多条参考描述
        :return: total_loss, metrics
        """
        device = attention_mask.device
        ground_truths = CiderReward.normalize_ground_truths(ground_truths)
        
        sampled_ids, sample_log_probs, seq_mask = self.generate_with_sampling(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
        )
        
        if self.use_baseline:
            with torch.no_grad():
                greedy_ids, _, _ = self.generate_greedy(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                )
        
        sampled_captions = self.decode_batch(sampled_ids)
        greedy_captions = self.decode_batch(greedy_ids) if self.use_baseline else sampled_captions
        
        reward_sampled = self.cider_reward.compute_batch_reward(
            sampled_captions, ground_truths
        )
        
        if self.use_baseline:
            reward_greedy = self.cider_reward.compute_batch_reward(
                greedy_captions, ground_truths
            )
            advantage = reward_sampled - reward_greedy
        else:
            advantage = reward_sampled
        
        advantage_tensor = torch.tensor(advantage, device=device, dtype=torch.float32)
        
        masked_log_probs = sample_log_probs * seq_mask
        denom = seq_mask.sum().clamp(min=1.0)
        rl_loss = -(advantage_tensor.unsqueeze(1) * masked_log_probs).sum() / denom
        
        if ce_loss is not None:
            total_loss = (1 - self.scst_weight) * ce_loss + self.scst_weight * rl_loss
        else:
            total_loss = rl_loss
        
        metrics = {
            'rl_loss': rl_loss.item(),
            'cider_sampled': float(np.mean(reward_sampled)),
            'cider_greedy': float(np.mean(reward_greedy)) if self.use_baseline else 0.0,
            'advantage': float(np.mean(advantage)),
        }
        
        return total_loss, metrics
