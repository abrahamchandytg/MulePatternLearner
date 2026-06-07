from typing import ClassVar

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = Field(
        default="",
        description="TigerGraph host URL",
    )
    graphname: str = Field(
        default="",
        description="Name of the graph",
    )
    secret: SecretStr = Field(
        default=SecretStr(""),
        description="REST++ secret used to mint auth tokens",
    )

    @field_validator("host", "graphname")
    @classmethod
    def _str_required(cls, v: str, info: ValidationInfo) -> str:
        if not v:
            name = info.field_name or "field"
            raise ValueError(f"{name} must be set in .env")
        return v

    @field_validator("secret")
    @classmethod
    def _secret_required(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value():
            raise ValueError("secret must be set in .env")
        return v
