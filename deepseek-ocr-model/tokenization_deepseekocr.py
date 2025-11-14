from transformers import AutoTokenizer

class DeepseekOCRTokenizer:
    """
    This is a thin wrapper for using an existing tokenizer (e.g., DeepSeek or GPT2)
    under the custom model_type 'deepseekocr'.
    """

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # You can swap this base model if your tokenizer came from another checkpoint
        return AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder", *args, **kwargs)
