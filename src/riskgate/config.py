from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    data_dir: Path = Field(default=Path("data/raw"), validation_alias="RISKGATE_DATA_DIR")
    artifacts_dir: Path = Field(
        default=Path("artifacts"),
        validation_alias="RISKGATE_ARTIFACTS_DIR",
    )
    results_dir: Path = Field(default=Path("results"), validation_alias="RISKGATE_RESULTS_DIR")

    train_frac: float = Field(default=0.70, ge=0.0, le=1.0, validation_alias="RISKGATE_TRAIN_FRAC")
    val_frac: float = Field(default=0.15, ge=0.0, le=1.0, validation_alias="RISKGATE_VAL_FRAC")

    cost_fn: float = Field(default=100.0, gt=0.0, validation_alias="RISKGATE_COST_FN")
    cost_fp: float = Field(default=1.0, gt=0.0, validation_alias="RISKGATE_COST_FP")
    target_fpr: float = Field(default=0.01, gt=0.0, lt=1.0, validation_alias="RISKGATE_TARGET_FPR")

    random_state: int = Field(default=42, validation_alias="RISKGATE_RANDOM_STATE")

    @property
    def test_frac(self) -> float:
        remainder = 1.0 - self.train_frac - self.val_frac
        if remainder <= 0.0:
            msg = "train_frac + val_frac must be < 1"
            raise ValueError(msg)
        return remainder


def get_settings() -> Settings:
    return Settings()
