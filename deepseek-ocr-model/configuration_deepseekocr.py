# configuration_deepseekocr.py
# ------------------------------------------------------------
# Configuration class for the Deepseek-OCR model
# ------------------------------------------------------------
from transformers.utils import logging
from .configuration_deepseek_v2 import DeepseekV2Config

logger = logging.get_logger(__name__)

DEEPSEEK_OCR_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

class DeepseekOCRConfig(DeepseekV2Config):
    """
    Config for Deepseek-OCR.

    Inherits all language-model fields from DeepseekV2Config
    (hidden_size, hidden_act, attention_bias, etc.) and adds
    OCR / vision specific metadata.
    """

    model_type = "deepseekocr"

    def __init__(
        self,
        # OCR / vision specific
        candidate_resolutions=None,
        global_view_pos="head",
        tile_tag="2D",
        projector_config=None,
        vision_config=None,
        language_config=None,
        **kwargs,
    ):
        # If a nested language_config dict is provided in config.json,
        # merge it into kwargs so DeepseekV2Config sees all LM params.
        if language_config is not None and isinstance(language_config, dict):
            base = dict(language_config)  # copy
            base.update(kwargs)           # top-level overrides nested
            kwargs = base

        # Let DeepseekV2Config handle all core model parameters.
        # NOTE: we do NOT pass torch_dtype explicitly here, it will be
        # picked from kwargs if present, so no "multiple values" error.
        super().__init__(**kwargs)

        # Store OCR-specific attributes
        self.candidate_resolutions = candidate_resolutions or [[1024, 1024]]
        self.global_view_pos = global_view_pos
        self.tile_tag = tile_tag

        # Keep sub-configs around for the modeling code
        self.projector_config = projector_config
        self.vision_config = vision_config
        self.language_config = language_config

        logger.info("✅ DeepseekOCRConfig initialized (inherits DeepseekV2Config).")


