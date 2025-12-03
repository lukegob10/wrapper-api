from __future__ import annotations

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field("roo-compatible-completions-api", env="APP_NAME")
    internal_endpoint: str | None = Field(None, env="INTERNAL_COMPLETIONS_URL")
    vertex_project: str | None = Field(None, env="VERTEX_PROJECT")
    vertex_location: str | None = Field("us-central1", env="VERTEX_LOCATION")
    vertex_model: str | None = Field("text-bison@001", env="VERTEX_MODEL")
    vertex_base_url: str | None = Field(None, env="VERTEX_BASE_URL")
    lm_studio_base_url: str | None = Field(None, env="LM_STUDIO_BASE_URL")
    custom_user_id: str = Field(default_factory=lambda: os.getenv("USER", ""), env="CUSTOM_USER_ID")
    custom_header_name: str = Field("x-custom-userid", env="CUSTOM_HEADER_NAME")
    ssl_cert_file: str = Field("CAChain_PROD.pem", env="SSL_CERT_FILE")
    helix_profile: str | None = Field(None, env="HELIX_PROFILE")
    helix_access_token_cmd: str = Field("helix auth access-token print -a", env="HELIX_ACCESS_TOKEN_CMD")
    helix_token_ttl: int = Field(600, env="HELIX_TOKEN_TTL")
    helix_refresh_margin: int = Field(60, env="HELIX_REFRESH_MARGIN")
    request_timeout: float = Field(30.0, env="REQUEST_TIMEOUT_SECONDS")
    max_retries: int = Field(2, env="REQUEST_MAX_RETRIES")
    pool_connections: int = Field(100, env="HTTPX_MAX_CONNECTIONS")
    pool_keepalive: int = Field(20, env="HTTPX_MAX_KEEPALIVE")
    log_level: str = Field("info", env="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
