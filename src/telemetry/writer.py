from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any


class JsonlWriter:
    def __init__(self):
        self._lock = RLock()

    def append(self, path: Path | str, payload: dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock, target.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
