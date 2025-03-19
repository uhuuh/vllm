import os
# os.environ['VLLM_LOGGING_LEVEL'] = "DEBUG"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['NCCL_DEBUG'] = "TRACE"
# os.environ['VLLM_TRACE_FUNCTION'] = "1"


from vllm import LLM, SamplingParams
import os
import time

os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="/mnt/e/SOME_PROGRAM/ckpt/opt-125m", 
    gpu_memory_utilization=0.5,
    enforce_eager=True,
)


# print("start debug")
# # unset DEBUGPY_ADAPTER_ENDPOINTS 避免debugpy报错lost stderr, 参考https://github.com/microsoft/vscode-python-debugger/issues/612
# del os.environ['DEBUGPY_ADAPTER_ENDPOINTS']
#
# import debugpy
# debugpy.listen(('127.0.0.1', 9342))  # 监听所有网络接口的5678端口
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print("end debug")

llm.start_profile()
outputs = llm.generate(prompts, sampling_params)
llm.stop_profile()


for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

time.sleep(10)

# 在线推理用法
# VLLM_TORCH_PROFILER_DIR=./vllm_profile python -m vllm.entrypoints.openai.api_server --model /mnt/e/SOME_PROGRAM/ckpt/opt-125m --enforce-eager

