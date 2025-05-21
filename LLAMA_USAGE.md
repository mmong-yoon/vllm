# Running Llama Models with vLLM

This document summarizes how to run Meta's Llama models using vLLM, both for offline inference and when serving the model with the OpenAI-compatible API.

## Offline Inference

vLLM provides the `LLM` class for offline generation. The example script `examples/offline_inference/basic/generate.py` defaults to the Llama 3.2 1B Instruct model and shows the basic workflow:

```python
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

# Engine arguments are parsed from the command line
EngineArgs.add_cli_args(parser)
parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
...
llm = LLM(**args)
```

The script creates an `LLM` instance, sets sampling parameters, and generates text for a few prompts:

```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
outputs = llm.generate(prompts, sampling_params)
```

See the complete example in `examples/offline_inference/basic/generate.py` for details.

## Serving via the OpenAI-Compatible API

To expose a Llama model over an API, use the `vllm serve` command. The `openai_chat_completion_client.py` example notes that the server can be started as follows:

```bash
vllm serve meta-llama/Llama-2-7b-chat-hf
```

Once the server is running, a client can query it through the OpenAI Chat Completion API:

```python
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(messages=messages, model=client.models.list().data[0].id)
```

## CPU Example

For CPU-only environments, `docs/source/getting_started/installation/cpu.md` shows how to launch the server with tensor parallelism and explicit thread binding:

```bash
VLLM_CPU_KVCACHE_SPACE=40 VLLM_CPU_OMP_THREADS_BIND="0-31|32-63" \
    vllm serve meta-llama/Llama-2-7b-chat-hf -tp=2 --distributed-executor-backend mp
```

This command reserves 40 GiB for the KV cache and splits the model across two CPU ranks.
