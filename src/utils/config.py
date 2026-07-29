from pydantic_settings import BaseSettings
from functools import lru_cache

class NLSettings(BaseSettings):
    spacy_model: str = "en_core_web_sm"
    cache_ttl: int = 3600
    max_text_length: int = 10000
    default_language: str = "en"

@lru_cache
def get_settings() -> NLSettings:
    return NLSettings()
