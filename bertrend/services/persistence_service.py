from __future__ import annotations

import dill
from pathlib import Path
from loguru import logger


BERTREND_FILE = "bertrend.dill"


class PersistenceService:
    """Isolate (de)serialisation concerns from domain logic."""

    # -------- BERTrend -------------------------------------------------- #
    @staticmethod
    def save_bertrend(obj: "BERTrend", path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / BERTREND_FILE).open("wb") as fh:
            dill.dump(obj, fh)
        logger.info("BERTrend saved at %s", path)

    @staticmethod
    def load_bertrend(path: Path) -> "BERTrend":
        from bertrend.BERTrend import BERTrend  # local import to avoid cycle

        file = path / BERTREND_FILE
        if not file.exists():
            raise FileNotFoundError(file)
        with file.open("rb") as fh:
            obj: BERTrend = dill.load(fh)
        logger.info("BERTrend restored from %s", path)
        return obj
