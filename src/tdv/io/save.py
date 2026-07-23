import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def save_intermediate(path: str | Path, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def save_json(path: str | Path, data: Any, precision: int = 4) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    class FloatEncoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            return super().default(o)

        def encode(self, o: Any) -> str:
            return self._encode(o)

        def _encode(self, o: Any, indent: int = 0) -> str:
            if isinstance(o, float):
                return f"{o:.{precision}f}"
            if isinstance(o, dict):
                items = []
                for k, v in o.items():
                    items.append(f'{"  " * (indent + 1)}"{k}": {self._encode(v, indent + 1)}')
                inner = ",\n".join(items)
                return "{\n" + inner + "\n" + "  " * indent + "}"
            if isinstance(o, list):
                items = [self._encode(v, indent + 1) for v in o]
                if all(len(item) < 60 for item in items):
                    inner = ", ".join(items)
                    return f"[{inner}]"
                inner = ",\n".join(f'{"  " * (indent + 1)}{item}' for item in items)
                return "[\n" + inner + "\n" + "  " * indent + "]"
            return json.dumps(o)

    with open(path, "w") as f:
        f.write(FloatEncoder().encode(data))
        f.write("\n")
