"""A layer that samples the next tokens from the model's outputs."""
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.parallel_utils.tensor_parallel import (
    gather_from_tensor_model_parallel_region)
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceOutputs


class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence and frequency penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> Dict[int, SequenceOutputs]:
        # Get the hidden states that we use for sampling.
        hidden_states = _prune_hidden_states(hidden_states, input_metadata)
        # sampler流程
        # 从hidden states挑选出logits后
        # 按照presence penalty, freq penalty, temperature处理logits
        # logits计算得到logp
        # 按照topk和toppc处理logp
        # 按照best_of和temperature生成next tokens, temperature为0时要求best of为1
        # 根据logprobs和next tokens生成next_logprobs

        # Get the logits for the next tokens.
        logits = torch.matmul(hidden_states, embedding.t())
        logits = gather_from_tensor_model_parallel_region(logits)
        # Remove paddings in vocab (if any).
        logits = logits[:, :self.vocab_size]

        # Apply presence and frequency penalties.
        output_tokens = _get_output_tokens(input_metadata)
        assert len(output_tokens) == logits.shape[0]
        presence_penalties, frequency_penalties = _get_penalties(input_metadata)
        assert len(presence_penalties) == logits.shape[0]
        assert len(frequency_penalties) == logits.shape[0]
        # presence_penalties只要出现了, 该位置的logits就减少presence_penalties
        # frequency_penalties是对应位置的logits减少出现次数乘上frequency_penalties
        # 这里是对logits进行操作, 而不是logp上进行操作. 如果在logp上进行操作, 操作完之后, 应该要重新归一化
        logits = _apply_penalties(
            logits, output_tokens, presence_penalties, frequency_penalties,
            self.vocab_size)

        # 这部分好像没有体现温度为0的时候, 就是确定的
        # ------ 确定性体现在_sample_from_prompt和_sample_from_generation_tokens中使用torch.argmax（greedy）而非multinomial采样
        # Apply temperature scaling.
        temperatures = _get_temperatures(input_metadata)
        assert len(temperatures) == logits.shape[0]
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(
                temperatures, dtype=logits.dtype, device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities (before applying top-p and top-k).
        logprobs = torch.log(probs)

        # topp为什么不在logits上进行操作 ------- top-p（nucleus sampling）基于累积概率，必须在概率空间操作。在logits上操作无法直接计算累积概率和。Top-k可以在logits上操作（取前k个），但top-p不能
        # 下面直接在prob上操作, 某些位置的prob可能设为0, 这样所有加和不为1, 这样合适吗? ------ multinomial接受非归一化的概率分布，内部会自动归一化。代码将不需要的token概率置0，multinomial只会从非零位置采样。
        # Apply top-p and top-k truncation.
        top_ps, top_ks = _get_top_p_top_k(input_metadata, self.vocab_size)
        assert len(top_ps) == len(top_ks) == probs.shape[0]
        if any(p < 1.0 for p in top_ps) or any(k != self.vocab_size for k in top_ks):
            probs = _apply_top_p_top_k(probs, top_ps, top_ks)

        # Sample the next tokens.
        return _sample(probs, logprobs, input_metadata)


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    input_metadata: InputMetadata,
) -> torch.Tensor:
    # 前面是prefill请求后面是decode请求, 因此last token ind才这样计算
    start_idx = 0
    last_token_indicies: List[int] = []
    for prompt_len in input_metadata.prompt_lens:
        last_token_indicies.append(start_idx + prompt_len - 1)
        start_idx += prompt_len
    last_token_indicies.extend(
        range(start_idx, start_idx + input_metadata.num_generation_tokens))
    return hidden_states[last_token_indicies]


def _get_penalties(
    input_metadata: InputMetadata,
) -> Tuple[List[float], List[float]]:
    # Collect the presence and frequency penalties.
    presence_penalties: List[float] = []
    frequency_penalties: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        p = sampling_params.presence_penalty
        f = sampling_params.frequency_penalty
        if i < input_metadata.num_prompts:
            # A prompt input.
            presence_penalties.append(p)
            frequency_penalties.append(f)
        else:
            # A generation token.
            presence_penalties += [p] * len(seq_ids)
            frequency_penalties += [f] * len(seq_ids)
    return presence_penalties, frequency_penalties


def _get_output_tokens(
    input_metadata: InputMetadata,
) -> List[List[int]]:
    output_tokens: List[List[int]] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, _ = seq_group
        if i < input_metadata.num_prompts:
            # A prompt input.
            # NOTE: While the prompt input usually has no output tokens,
            # it may have output tokens in the case of recomputation.
            seq_id = seq_ids[0]
            seq_data = input_metadata.seq_data[seq_id]
            output_tokens.append(seq_data.output_token_ids)
        else:
            # A generation token.
            for seq_id in seq_ids:
                seq_data = input_metadata.seq_data[seq_id]
                output_tokens.append(seq_data.output_token_ids)
    return output_tokens


def _apply_penalties(
    logits: torch.Tensor,
    output_tokens: List[List[int]],
    presence_penalties: List[float],
    frequency_penalties: List[float],
    vocab_size: int,
) -> torch.Tensor:
    num_seqs = logits.shape[0]
    # Collect the indices of sequences that have non-zero penalties.
    indices = []
    for i in range(num_seqs):
        if not output_tokens[i]:
            continue
        p = presence_penalties[i]
        f = frequency_penalties[i]
        if p == 0.0 and f == 0.0:
            continue
        indices.append(i)

    # Return early if all sequences have zero penalties.
    if not indices:
        return logits

    bin_counts = []
    for i in indices:
        bin_counts.append(np.bincount(output_tokens[i], minlength=vocab_size))
    bin_counts = np.stack(bin_counts, axis=0)
    bin_counts = torch.from_numpy(bin_counts).to(dtype=logits.dtype,
                                                 device=logits.device)

    frequency_penalties = [frequency_penalties[i] for i in indices]
    frequency_penalties = torch.tensor(
        frequency_penalties, dtype=logits.dtype, device=logits.device)
    presence_penalties = [presence_penalties[i] for i in indices]
    presence_penalties = torch.tensor(
        presence_penalties, dtype=logits.dtype, device=logits.device)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits[indices] -= frequency_penalties.unsqueeze(dim=1) * bin_counts
    presence_mask = (bin_counts > 0.0).to(dtype=logits.dtype)
    logits[indices] -= presence_penalties.unsqueeze(dim=1) * presence_mask
    return logits


def _get_temperatures(
    input_metadata: InputMetadata,
) -> List[float]:
    # Collect the temperatures for the logits.
    temperatures: List[float] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        temperature = sampling_params.temperature
        if temperature == 0.0:
            # NOTE: Zero temperature means deterministic sampling
            # (i.e., greedy sampling or beam search).
            # Set the temperature to 1 to avoid division by zero.
            temperature = 1.0

        if i < input_metadata.num_prompts:
            # A prompt input.
            temperatures.append(temperature)
        else:
            # A generation token.
            temperatures += [temperature] * len(seq_ids)
    return temperatures


def _get_top_p_top_k(
    input_metadata: InputMetadata,
    vocab_size: int,
) -> Tuple[List[float], List[int]]:
    top_ps: List[float] = []
    top_ks: List[int] = []
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        top_p = sampling_params.top_p
        # k should not be greater than the vocab size.
        top_k = min(sampling_params.top_k, vocab_size)
        # k=-1 means no truncation.
        top_k = vocab_size if top_k == -1 else top_k
        if i < input_metadata.num_prompts:
            # A prompt input.
            top_ps.append(top_p)
            top_ks.append(top_k)
        else:
            # A generation token.
            top_ps += [top_p] * len(seq_ids)
            top_ks += [top_k] * len(seq_ids)
    return top_ps, top_ks


def _apply_top_p_top_k(
    probs: torch.Tensor,
    top_ps: List[float],
    top_ks: List[int],
) -> torch.Tensor:
    p = torch.tensor(top_ps, dtype=probs.dtype, device=probs.device)
    k = torch.tensor(top_ks, dtype=torch.int, device=probs.device)
    # prob是一个二维的, [num seq, vocab size]
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)

    # Apply top-p.
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    top_p_mask = (probs_sum - probs_sort) > p.unsqueeze(dim=1)
    probs_sort[top_p_mask] = 0.0

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
    top_k_mask = top_k_mask.expand(probs_idx.shape[0], -1)
    top_k_mask = top_k_mask >= k.unsqueeze(dim=1)
    probs_sort[top_k_mask] = 0.0

    # Re-sort the probabilities.
    probs = torch.gather(
        probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))
    return probs


def _get_topk_logprobs(
    logprobs: torch.Tensor,
    num_logprobs: Optional[int],
) -> Dict[int, float]:
    if num_logprobs is None or num_logprobs == 0:
        return {}

    topk_logprobs, topk_ids = torch.topk(logprobs, num_logprobs)
    if num_logprobs == 1:
        topk_logprobs = [topk_logprobs.item()]
        topk_ids = [topk_ids.item()]
    else:
        topk_logprobs = topk_logprobs.tolist()
        topk_ids = topk_ids.tolist()

    token_to_logprob: Dict[int, float] = {}
    for token_id, logprob in zip(topk_ids, topk_logprobs):
        token_to_logprob[token_id] = logprob
    return token_to_logprob


def _sample_from_prompt(
    prob: torch.Tensor,
    sampling_params: SamplingParams,
) -> List[int]:
    if sampling_params.use_beam_search:
        # Beam search.
        beam_width = sampling_params.best_of
        _, next_token_ids = torch.topk(prob, beam_width)
        next_token_ids = next_token_ids.tolist()
    elif sampling_params.temperature == 0.0:
        # Greedy sampling.
        assert sampling_params.best_of == 1
        next_token_id = torch.argmax(prob)
        next_token_ids = [next_token_id.item()]
    else:
        # Random sampling.
        # Sample `best_of` tokens for the prompt.
        num_seqs = sampling_params.best_of
        next_token_ids = torch.multinomial(
            prob, num_samples=num_seqs, replacement=True) # replacement=True说明可以重复采样同一个样本
        next_token_ids = next_token_ids.tolist()
    return next_token_ids


def _sample_from_generation_tokens(
    seq_ids: List[int],
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    seq_logprobs: List[float],
    sampling_params: SamplingParams,
) -> Tuple[List[int], List[int]]:
    # NOTE(woosuk): sampling_params.best_of can be greater than
    # len(seq_ids) because some sequences in the group might have
    # been already terminated.
    if sampling_params.use_beam_search:
        # TODO
        # Beam search.
        # Add cumulative logprobs for the sequences in the group.
        seq_logprobs = torch.tensor(
            seq_logprobs, dtype=torch.float, device=logprobs.device)
        logprobs = logprobs + seq_logprobs.unsqueeze(dim=1)
        # NOTE 为什么beam search使用的是logp的累加值, 而不是累计乘值等其他 -------
        # 率的加法可以通过对数概率的加法来实现。由于对数函数的单调性，log(a * b) = log(a) + log(b)，所以在 Beam Search 中，我们用 对数概率的累加值 来累积概率，以避免数值下溢问题。
        # log操作可以减少上溢出和下溢出, 随着接近0的过程中放大, 同时也减少了扩张趋势

        vocab_size = logprobs.size(-1)
        beam_width = len(seq_ids)
        # logprobs是二维, [num seq, vocab size]
        _, topk_ids = torch.topk(logprobs.flatten(), beam_width)
        topk_ids = topk_ids.tolist()
        seq_idx = [i // vocab_size for i in topk_ids] # [num seq * beam width]
        beam_seq_ids = [seq_ids[i] for i in seq_idx] # [num seq * beam width]
        token_ids = [i % vocab_size for i in topk_ids] # [num seq * beam width]

        beam_outputs: Dict[int, Tuple[int, int]] = {}
        outstanding_beams: List[Tuple[int, int]] = []
        # If a beam survives, continue with it.
        for seq_id, token_id in zip(beam_seq_ids, token_ids):
            if seq_id not in beam_outputs:
                beam_outputs[seq_id] = (seq_id, token_id)
            else:
                # 出现在outstanding_beams说明, 一个seq有超过一个token被采样到了
                outstanding_beams.append((seq_id, token_id))

        # If a beam is discarded, fork another beam.
        for seq_id in seq_ids:
            if seq_id not in beam_outputs:
                beam_outputs[seq_id] = outstanding_beams.pop()
        assert not outstanding_beams

        parent_seq_ids = [beam_outputs[seq_id][0] for seq_id in seq_ids]
        next_token_ids = [beam_outputs[seq_id][1] for seq_id in seq_ids]
    elif sampling_params.temperature == 0.0:
        # Greedy sampling.
        assert len(seq_ids) == 1
        next_token_id = torch.argmax(probs, dim=-1)
        next_token_ids = [int(next_token_id.item())]
        parent_seq_ids = seq_ids
    else:
        # Random sampling.
        # Sample 1 token for each sequence in the group.
        next_token_ids = torch.multinomial(
            probs, num_samples=1, replacement=True)
        next_token_ids = next_token_ids.squeeze(dim=-1).tolist()
        parent_seq_ids = seq_ids
    return parent_seq_ids, next_token_ids


def _sample(
    probs: torch.Tensor,
    logprobs: torch.Tensor,
    input_metadata: InputMetadata,
) -> Dict[int, SequenceOutputs]:
    seq_outputs: Dict[int, SequenceOutputs] = {}

    # TODO(woosuk): Optimize.
    idx = 0
    for i, seq_group in enumerate(input_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        if i < input_metadata.num_prompts:
            # Generate the next tokens for a prompt input.
            assert len(seq_ids) == sampling_params.best_of
            prob = probs[idx]
            logprob = logprobs[idx]
            idx += 1

            # logprobs控制返回信息中的最大logp的数量, 与sample next token无关
            # Sample the next tokens.
            next_token_ids = _sample_from_prompt(prob, sampling_params)
            # Get top-k log probabilities for the next tokens.
            next_logprobs = _get_topk_logprobs(
                logprob, sampling_params.logprobs)

            # Build the output.
            for seq_id, next_token_id in zip(seq_ids, next_token_ids):
                # 最后返回的output_logprobs是两部分的并集, 一部分是logprobs设置最大某些数量的logp, 一个是next tokens对应的logp
                output_logprobs = next_logprobs.copy()
                output_logprobs[next_token_id] = logprob[next_token_id].item()
                seq_outputs[seq_id] = SequenceOutputs(
                    seq_id, seq_id, next_token_id, output_logprobs)
        else:
            # Generate the next tokens for generation tokens.
            prob = probs[idx:idx + len(seq_ids)]
            logprob = logprobs[idx:idx + len(seq_ids)]
            idx += len(seq_ids)

            # Sample the next tokens.
            seq_logprobs = [
                input_metadata.seq_data[seq_id].cumulative_logprob
                for seq_id in seq_ids]
            # TODO parent_seq_ids
            parent_seq_ids, next_token_ids = _sample_from_generation_tokens(
                seq_ids, prob, logprob, seq_logprobs, sampling_params)

            # Get top-k log probabilities for the next tokens.
            next_logprobs: Dict[int, Dict[int, float]] = {}
            for i, seq_id in enumerate(seq_ids):
                next_logprobs[seq_id] = _get_topk_logprobs(
                    logprob[i], sampling_params.logprobs)

            # Build the output.
            for seq_id, parent_seq_id, next_token_id in zip(
                seq_ids, parent_seq_ids, next_token_ids):
                i = seq_ids.index(parent_seq_id)
                output_logprobs = next_logprobs[parent_seq_id].copy()
                output_logprobs[next_token_id] = logprob[i, next_token_id].item()
                seq_outputs[seq_id] = SequenceOutputs(
                    seq_id,
                    parent_seq_id,
                    next_token_id,
                    output_logprobs,
                )

    return seq_outputs
