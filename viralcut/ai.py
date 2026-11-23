"""ИИ-помощник: советы по готовым клипам через внешний API.

Работает с любым OpenAI-совместимым API (`/chat/completions`): DeepSeek,
OpenAI, локальные серверы (Ollama/LM Studio/vLLM) — пользователь выбирает
провайдера и вставляет свой ключ. Ключ хранится в `data/settings.json` с правами
0600 и никогда не попадает в ответы API и логи.

Запрос уходит только по кнопке «Совет ИИ»: приложение ничего не отправляет
само, и в подсказке в интерфейсе это написано.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

PROVIDERS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "label": "DeepSeek",
        "base": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key_url": "https://platform.deepseek.com/api_keys",
        "note": "Основной провайдер: свой ключ DeepSeek, тарификация напрямую.",
    },
    "polza": {
        "label": "Polza.ai",
        "base": "https://polza.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "key_url": "https://polza.ai/dashboard/api-keys",
        "note": "Агрегатор: один ключ на сотни моделей (OpenAI, Claude, Qwen, DeepSeek и др.).",
    },
    "openai": {
        "label": "OpenAI",
        "base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_url": "https://platform.openai.com/api-keys",
    },
    "custom": {
        "label": "Свой сервер",
        "base": "",
        "model": "",
        "key_url": "",
    },
}

MAX_TRANSCRIPT_CHARS = 6000

# Если каталог моделей закрыт ключом (так у DeepSeek), показываем известные
# модели своего провайдера — выпадающий список не должен быть пустым.
FALLBACK_MODELS: Dict[str, List[str]] = {
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "polza": ["openai/gpt-4o-mini", "openai/gpt-4o", "deepseek/deepseek-chat"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "custom": [],
}
TIMEOUT = 60


class AIError(RuntimeError):
    """Понятная человеку ошибка обращения к ИИ."""


def _env_key() -> str:
    for name in ("VIRALCUT_AI_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def key() -> str:
    """Ключ: переменная окружения важнее сохранённого в настройках."""
    env = _env_key()
    if env:
        return env
    try:
        from . import settings

        return str(settings.get().get("ai_key") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _conf() -> Dict[str, str]:
    try:
        from . import settings

        data = settings.get()
    except Exception:  # noqa: BLE001
        data = {}
    provider = str(data.get("ai_provider") or "deepseek").lower()
    if provider not in PROVIDERS:
        provider = "deepseek"
    preset = PROVIDERS[provider]
    base = str(data.get("ai_base") or preset["base"]).strip().rstrip("/")
    model = str(data.get("ai_model") or preset["model"]).strip()
    return {"provider": provider, "base": base, "model": model,
            "label": preset["label"], "key_url": preset["key_url"]}


def configured() -> bool:
    conf = _conf()
    return bool(key() and conf["base"] and conf["model"])


def status() -> Dict[str, Any]:
    """Что показывать в настройках: без самого ключа."""
    conf = _conf()
    saved = key()
    return {
        "provider": conf["provider"],
        "providers": [{"key": k, "label": v["label"], "base": v["base"],
                       "model": v["model"], "key_url": v["key_url"]}
                      for k, v in PROVIDERS.items()],
        "base": conf["base"],
        "model": conf["model"],
        "key_url": conf["key_url"],
        "note": PROVIDERS[conf["provider"]].get("note", ""),
        "configured": configured(),
        "key_hint": (saved[:4] + "…" + saved[-4:]) if len(saved) >= 10 else ("сохранён" if saved else ""),
        "from_env": bool(_env_key()),
    }


_MODELS_CACHE: Dict[str, Any] = {"ts": 0.0, "base": "", "data": []}
_MODELS_TTL = 600.0


def models(force: bool = False, provider: Optional[str] = None) -> Dict[str, Any]:
    """Каталог моделей провайдера (OpenAI-совместимый GET /models).

    `provider` позволяет спросить каталог ещё до сохранения настроек — так
    выпадающий список обновляется сразу при переключении в интерфейсе.
    У Polza.ai список публичный, но ключ, если он есть, отправляем тоже.
    """
    conf = _conf()
    if provider and provider in PROVIDERS and provider != conf["provider"]:
        preset = PROVIDERS[provider]
        saved = {}
        try:
            from . import settings

            saved = settings.get()
        except Exception:  # noqa: BLE001
            saved = {}
        conf = {
            "provider": provider,
            "base": preset["base"],
            "model": preset["model"],
            "label": preset["label"],
            "key_url": preset["key_url"],
        }
        # для custom адрес берём из настроек пользователя
        if provider == "custom":
            conf["base"] = str(saved.get("ai_base") or "").strip().rstrip("/")
    if force:
        _MODELS_CACHE["ts"] = 0.0
    now = time.time()
    if (not force and _MODELS_CACHE["data"] and _MODELS_CACHE["base"] == conf["base"]
            and now - float(_MODELS_CACHE["ts"]) < _MODELS_TTL):
        return {"models": _MODELS_CACHE["data"], "base": conf["base"], "cached": True}
    if not conf["base"]:
        raise AIError("Не выбран адрес API. Укажите базовый URL своего сервера.")
    request = urllib.request.Request(f"{conf['base']}/models", method="GET")
    api_key = key()
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            known = FALLBACK_MODELS.get(conf["provider"]) or [conf["model"]]
            return {"models": [{"id": name, "name": name, "owner": "",
                                "description": ""} for name in known if name],
                    "base": conf["base"], "fallback": True,
                    "warning": ("Ключ не принят сервером — показываю известные "
                                "модели. Проверьте ключ." if api_key else "")}
        raise AIError(f"Каталог моделей недоступен (ошибка {exc.code}).") from exc
    except urllib.error.URLError as exc:
        raise AIError("Не удалось связаться с сервером, чтобы получить список моделей.") from exc
    raw = data.get("data") if isinstance(data, dict) else data
    result: List[Dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "chat")
        if kind != "chat":
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        result.append({
            "id": model_id,
            "name": str(item.get("name") or model_id),
            "owner": str(item.get("owned_by") or ""),
            "description": str(item.get("short_description") or "")[:240],
        })
    result.sort(key=lambda item: (item["owner"], item["id"]))
    _MODELS_CACHE.update({"ts": now, "base": conf["base"], "data": result})
    return {"models": result, "base": conf["base"], "cached": False}


def _post(url: str, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001
            pass
        if exc.code in (401, 403):
            raise AIError("Ключ не подошёл: сервер ответил «доступ запрещён». "
                          "Проверьте ключ и выбранного провайдера.") from exc
        if exc.code == 429:
            raise AIError("Слишком много запросов или закончился баланс ключа. "
                          "Попробуйте позже.") from exc
        raise AIError(f"Сервер ИИ ответил ошибкой {exc.code}. {detail}".strip()) from exc
    except urllib.error.URLError as exc:
        raise AIError("Не удалось связаться с сервером ИИ. Проверьте интернет "
                      "или адрес своего сервера.") from exc
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise AIError("Сервер ИИ вернул неожиданный ответ.") from exc


def _clip_block(project: Dict[str, Any], clip: Optional[Dict[str, Any]]) -> str:
    """Описание клипа для модели: тайминги, текст субтитров, хук."""
    lines: List[str] = []
    if clip:
        start = float(clip.get("start") or 0.0)
        end = float(clip.get("end") or 0.0)
        lines.append(f"Клип: {start:.1f}–{end:.1f} сек ({max(0.0, end - start):.1f} сек).")
        title = str(clip.get("title") or "").strip()
        if title:
            lines.append(f"Название клипа: {title}")
        hook = str(clip.get("hook") or "").strip()
        if hook:
            lines.append(f"Хук на экране: {hook}")
        words = clip.get("words") or []
        if isinstance(words, list) and words:
            text = " ".join(str(w.get("word") or "") for w in words if isinstance(w, dict))
            lines.append("Субтитры клипа: " + text[:MAX_TRANSCRIPT_CHARS])
        elif clip.get("text"):
            lines.append("Субтитры клипа: " + str(clip.get("text"))[:MAX_TRANSCRIPT_CHARS])
    transcript = project.get("transcript_text") or ""
    if transcript and not lines:
        lines.append("Текст записи: " + str(transcript)[:MAX_TRANSCRIPT_CHARS])
    if not lines:
        lines.append("Данных о клипе мало — дайте общие советы по оформлению "
                     "вертикальных клипов.")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "Ты — редактор коротких вертикальных видео для TikTok, Reels и Shorts. "
    "Отвечай по-русски, коротко и по делу. Не выдумывай фактов, которых нет в тексте. "
    "Давай конкретные действия, которые можно применить в приложении: "
    "текст хука, разбивку на строки, момент начала, длину, что убрать."
)


def build_prompt(project: Dict[str, Any], clip: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Сообщения для API: системная роль + данные проекта."""
    blocks = [
        f"Название записи: {project.get('title') or 'без названия'}",
        f"Язык: {project.get('language') or 'не определён'}",
    ]
    duration = project.get("duration")
    if duration:
        blocks.append(f"Длительность записи: {float(duration):.0f} сек.")
    clips = project.get("clips") or []
    if clips:
        blocks.append(f"Всего клипов в проекте: {len(clips)}")
    blocks.append(_clip_block(project, clip))
    blocks.append(
        "Дай 5 коротких пунктов:\n"
        "1) хук первой строкой — готовый текст;\n"
        "2) где лучше начать/закончить и почему;\n"
        "3) как разбить субтитры на строки;\n"
        "4) что убрать или добавить в монтаже;\n"
        "5) идея подписи к публикации."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]


def advice(project: Dict[str, Any], clip: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Совет по проекту/клипу. Бросает AIError с понятным текстом."""
    conf = _conf()
    api_key = key()
    if not api_key:
        raise AIError("Не задан ключ ИИ. Откройте настройки → «ИИ-помощник» и вставьте ключ.")
    if not conf["base"] or not conf["model"]:
        raise AIError("Не выбран провайдер или модель ИИ. Проверьте настройки.")
    payload = {
        "model": conf["model"],
        "messages": build_prompt(project, clip),
        "temperature": 0.4,
        "max_tokens": 700,
    }
    started = time.time()
    data = _post(f"{conf['base']}/chat/completions", payload, api_key)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AIError("Сервер ИИ вернул ответ без текста.") from exc
    text = str(content or "").strip()
    if not text:
        raise AIError("Сервер ИИ вернул пустой ответ.")
    return {
        "advice": text,
        "model": conf["model"],
        "provider": conf["provider"],
        "took": round(time.time() - started, 2),
    }
