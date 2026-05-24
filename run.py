#!/usr/bin/env python3
"""
Правильный запускатель RPG API.
Решает проблему relative imports при структуре src/.
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "src"

# Добавляем src в путь
sys.path.insert(0, str(SRC))

# Теперь импортируем приложение как часть пакета
import importlib.util

# Загружаем src как пакет
spec = importlib.util.spec_from_file_location(
    "src", 
    str(SRC / "__init__.py"),
    submodule_search_locations=[str(SRC)]
)
src_pkg = importlib.util.module_from_spec(spec)
sys.modules["src"] = src_pkg
spec.loader.exec_module(src_pkg)

# Теперь можно импортировать main как src.main
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
