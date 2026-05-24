#!/usr/bin/env python3
"""
Финальная надёжная версия запуска RPG API.

Этот скрипт решает проблему relative imports раз и навсегда
для проектов со структурой src/.
"""
import sys
import os
from pathlib import Path

def main():
    project_root = Path(__file__).parent.resolve()
    src_dir = project_root / "src"

    # 1. Добавляем src в начало sys.path
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # 2. Теперь импортируем приложение
    #    Важно: мы импортируем "main", а не "src.main"
    import main

    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() in ("true", "1", "yes")
    log_level = os.getenv("LOG_LEVEL", "info")

    print(f"🚀 Starting RPG API")
    print(f"   URL: http://{host}:{port}")
    print(f"   LLM: {os.getenv('OPENROUTER_CHAT_MODEL', 'deepseek/deepseek-v4-flash')}")
    print(f"   Embeddings: {os.getenv('OPENROUTER_EMBED_MODEL', 'qwen/qwen3-embedding-8b')}")

    uvicorn.run(
        main.app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
