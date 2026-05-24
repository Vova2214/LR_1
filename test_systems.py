#!/usr/bin/env python3
"""
Тестовый запуск и проверка всех основных систем проекта.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

print("=" * 60)
print("ПРОВЕРКА СИСТЕМ RPG API")
print("=" * 60)

errors = []

# 1. Конфигурация моделей
print("\n[1] Проверка конфигурации LLM...")
try:
    from src.db import (
        DEEPSEEK_V4_FLASH_MODEL,
        OPENROUTER_EMBED_MODEL,
        USE_EMBEDDINGS,
        OPENROUTER_API_KEY,
    )
    print(f"    DeepSeek V4 Flash: {DEEPSEEK_V4_FLASH_MODEL}")
    print(f"    Embeddings: {OPENROUTER_EMBED_MODEL} (enabled={USE_EMBEDDINGS})")
    if not OPENROUTER_API_KEY:
        print("    ⚠️  OPENROUTER_API_KEY не задан")
    print("    ✓ Конфигурация OK")
except Exception as e:
    errors.append(f"Config: {e}")
    print(f"    ✗ Ошибка: {e}")

# 2. OpenRouter клиент
print("\n[2] Проверка OpenRouter клиента...")
try:
    from src.llm.openrouter_chat_client import chat as openrouter_chat, generate_text, generate_json
    print("    ✓ OpenRouter client загружен")
except Exception as e:
    errors.append(f"LLM Client: {e}")
    print(f"    ✗ Ошибка: {e}")

# 3. Схемы
print("\n[3] Проверка схем...")
try:
    from src import schemas
    print("    ✓ Схемы загружены")
except Exception as e:
    errors.append(f"Schemas: {e}")
    print(f"    ✗ Ошибка: {e}")

# 4. Turn Application Service
print("\n[4] Проверка Turn Application Service...")
try:
    from src.application import turn_application_service
    print("    ✓ Turn service загружен")
except Exception as e:
    errors.append(f"Turn Service: {e}")
    print(f"    ✗ Ошибка: {e}")

# 5. Playground endpoint logic (симуляция)
print("\n[5] Проверка логики /playground/chat...")
try:
    # Просто проверяем, что импорт не падает на этом этапе
    from src.main import playground_chat
    print("    ✓ Эндпоинт /playground/chat импортирован")
except Exception as e:
    errors.append(f"Playground endpoint: {e}")
    print(f"    ✗ Ошибка: {e}")

print("\n" + "=" * 60)
if errors:
    print("РЕЗУЛЬТАТ: Обнаружены ошибки")
    for e in errors:
        print(f"  - {e}")
else:
    print("РЕЗУЛЬТАТ: Основные системы загружаются без критических ошибок")
print("=" * 60)
