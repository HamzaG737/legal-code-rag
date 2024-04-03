from enum import Enum


class MistralSupportedModels(str, Enum):
    MISTRAL_EMBED = "mistral-embed"


class OpenAISupportedModels(str, Enum):
    ADA = "text-embedding-ada-002"
    V3_SMALL = "text-embedding-3-small"
    V3_LARGE = "text-embedding-3-large"
