import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=f"{os.path.dirname(os.path.abspath(__file__))}/../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ARXIV_QUERY: str
    ARXIV_MAX_RESULTS: int
    SLACK_TOKEN: str
    SLACK_CHANNEL: str
    GEMINI_API_KEY: str

env = Env()
