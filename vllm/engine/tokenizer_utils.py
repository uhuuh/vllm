from typing import List, Tuple, Union

from transformers import (AutoConfig, AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from vllm.logger import init_logger

logger = init_logger(__name__)

_MODEL_TYPES_WITH_SLOW_TOKENIZER = []


def get_tokenizer(
    model_name: str,
    *args,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    config = AutoConfig.from_pretrained(model_name)
    if "open_llama" in model_name:
        kwargs["use_fast"] = False
        logger.info(
            "OpenLLaMA models do not support the fast tokenizer. "
            "Using the slow tokenizer instead.")
    elif config.model_type == "llama" and getattr(kwargs, "use_fast", True):
        # LLaMA fast tokenizer causes protobuf errors in some environments.
        # However, we found that the below LLaMA fast tokenizer works well in
        # most environments.
        model_name = "hf-internal-testing/llama-tokenizer"
        logger.info(
            f"Using the LLaMA fast tokenizer in '{model_name}' to avoid "
            "potential protobuf errors.")
    elif config.model_type in _MODEL_TYPES_WITH_SLOW_TOKENIZER:
        if getattr(kwargs, "use_fast", False) == True:
            raise ValueError(
                f"Cannot use the fast tokenizer for {config.model_type} due to "
                "bugs in the fast tokenizer.")
        logger.info(
            f"Using the slow tokenizer for {config.model_type} due to bugs in "
            "the fast tokenizer. This could potentially lead to performance "
            "degradation.")
        kwargs["use_fast"] = False
    return AutoTokenizer.from_pretrained(model_name, *args, **kwargs)


def detokenize_incrementally(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prev_output_tokens: List[str],
    new_token_id: int,
    skip_special_tokens: bool,
) -> Tuple[str, str]:
    """Detokenizes the new token in conjuction with the previous output tokens.
    # TODO pr介绍说这种增量detokenize有性能优化, 为什么呢?

    NOTE: This function does not update prev_output_tokens.

    Returns:
        new_token: The new token as a string.
        output_text: The new output text as a string.
    """
    # special_tokens 一般划分内容, 比如<|endoftext|>, <|pad|>, <|mask|>等, 一般不用返回给用户
    # skip_special_tokens为False, 这些特殊token对应的内容也会加到返回文本中
    # convert_ids_to_tokens 从token id到token, convert_tokens_to_string从token到text
    # convert_tokens_to_string的内部逻辑是先将 Token 映射回 Byte 序列，然后再执行 bytes.decode('utf-8', errors='replace')
    # 当stream中, 一个token值对应一个中文字的一部分utf-8字节序列中的一部分中, 进行bytes.decode应该会有一些不正常的输出
    # 需要使用特殊的decode, 如TextStreamer
    new_token = tokenizer.convert_ids_to_tokens(
        new_token_id, skip_special_tokens=skip_special_tokens)
    output_tokens = prev_output_tokens + [new_token]

    # Convert the tokens to a string.
    # Optimization: If the tokenizer does not have `added_tokens_encoder`,
    # then we can directly use `convert_tokens_to_string`.
    if not getattr(tokenizer, "added_tokens_encoder", {}):
        output_text = tokenizer.convert_tokens_to_string(output_tokens)
        return new_token, output_text

    # Adapted from https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921
    # NOTE(woosuk): The following code is slow because it runs a for loop over
    # the output_tokens. In Python, running a for loop over a list can be slow
    # even when the loop body is very simple.
    sub_texts = []
    current_sub_text = []
    for token in output_tokens:
        if skip_special_tokens and token in tokenizer.all_special_ids:
            continue
        # added_tokens_encoder 只有在 运行时 使用 add_tokens() 方法扩展 tokenizer 的词汇表时才会出现
        # 下面是为了修复https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921中提到的一个bug
        # added token不经过convert_tokens_to_string
        if token in tokenizer.added_tokens_encoder:
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                sub_texts.append(sub_text)
                current_sub_text = []
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
        sub_texts.append(sub_text)
    output_text = " ".join(sub_texts)
    return new_token, output_text
