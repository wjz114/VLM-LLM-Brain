"""
Model registry for the LLM-VLM brain alignment project.

Each entry specifies how to load a model and tokenizer. The extraction script
(``1_extract_attention.py``) dispatches on ``model_class`` to select the
correct Hugging Face loader.

"""

BF16_EAGER_LOAD_KWARGS = {
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "attn_implementation": "eager",
}

MODELS: dict[str, dict] = {
    "qwen2-7b": {
        "hf_name": "Qwen/Qwen2-7B",
        "model_class": "AutoModelForCausalLM",
        "load_kwargs": dict(BF16_EAGER_LOAD_KWARGS),
        "quantize_8bit": False,
    },
    "qwen2-vl-7b": {
        "hf_name": "Qwen/Qwen2-VL-7B",
        "model_class": "Qwen2VLTextModel",
        "load_kwargs": dict(BF16_EAGER_LOAD_KWARGS),
        "quantize_8bit": False,
    },
    "deepseek-llm-7b-base": {
        "hf_name": "deepseek-ai/deepseek-llm-7b-base",
        "model_class": "AutoModelForCausalLM",
        "load_kwargs": {
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "attn_implementation": "eager",
        },
        "quantize_8bit": False,
    },
    "deepseek-vl-7b-base": {
        "hf_name": "deepseek-ai/deepseek-vl-7b-base",
        # DeepSeek-VL uses a custom `multi_modality` config registered
        # by importing `deepseek_vl.models` before AutoModel loading.
        "model_class": "DeepseekVLTextBackbone",
        "load_kwargs": {
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "attn_implementation": "eager",
        },
        "quantize_8bit": False,
    },
    "mistral-7b-v0.1": {
        "hf_name": "mistralai/Mistral-7B-v0.1",
        "model_class": "AutoModelForCausalLM",
        "load_kwargs": dict(BF16_EAGER_LOAD_KWARGS),
        "quantize_8bit": False,
    },
    "idefics2-8b-base": {
        "hf_name": "HuggingFaceM4/idefics2-8b-base",
        # Prefer explicit class for multimodal checkpoints.
        "model_class": "Idefics2ForConditionalGeneration",
        "load_kwargs": dict(BF16_EAGER_LOAD_KWARGS),
        "quantize_8bit": False,
        "allow_processor_tokenizer_fallback": True,
    },
}
