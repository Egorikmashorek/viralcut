"""Действия помощника: он не только объясняет, но и делает.

Помощник возвращает список команд, а этот модуль их проверяет и исполняет.
Ничего опасного сюда не попадает: только белый список, никаких удалений,
никаких ключей. Всё, что меняет состояние, выполняется на сервере — поэтому
работает и без открытого интерфейса.

Команды:
  * `set_settings` — мощность, лимит процессора, выбор видеокарты;
  * `start_cut` — запустить нарезку по ссылке или уже загруженному проекту
    (с ИИ-режимом, количеством клипов и форматом);
  * `advice` — совет ИИ по проекту;
  * `open_settings`, `open_project`, `set_ai_mode` — подсказки интерфейсу.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from . import db, jobs, pipeline, resources

# Что снимаем с модели: строгая схема, лишнее отбрасываем.
ACTIONS_SPEC = """Доступные действия (только из этого списка):
{"type":"set_settings","power":0-100,"cpu_limit":25-100,"gpu_preference":"auto|integrated|discrete"}
{"type":"set_ai_mode","enabled":true|false,"prompt":"текст задания"}
{"type":"start_cut","url":"ссылка","project_id":"id","count":1-50,"min_dur":3-120,"max_dur":5-180,"aspect":"9:16|1:1|16:9|4:5","ai_mode":true|false,"ai_prompt":"текст"}
{"type":"advice","project_id":"id"}
{"type":"open_settings"}
{"type":"open_project","project_id":"id"}
Если пользователь просит что-то сделать — выполни и коротко подтверди результат.
Если действие не нужно — не добавляй его. В конце ответа добавь ОДИН блок:
```json
{"actions":[{"type":"...", ...}]}
```"""

_ALLOWED = {"set_settings", "set_ai_mode", "start_cut", "advice", "open_settings", "open_project"}
_UI_ONLY = {"open_settings", "open_project", "set_ai_mode"}

# Локальный разбор без модели: цифры и числительные в разных падежах.
_WORD_PATTERNS = [
    (r"\bдевяност\w*", 90),
    (r"\bвос[ье]м[иь]десят\w*", 80),
    (r"\bсем[иь]десят\w*", 70),
    (r"\bшест[иь]десят\w*", 60),
    (r"\bпят[иь]десят\w*", 50),
    (r"\bсорок\w*", 40),
    (r"\bтридцат\w*", 30),
    (r"\bдвадцат\w*", 20),
    (r"\bпятнадцат\w*", 15),
    (r"\bдесят\w*", 10),
    (r"\bсто\b", 100),
]


def _number(text: str) -> Optional[int]:
    match = re.search(r"(\d{1,3})", text)
    if match:
        return int(match.group(1))
    for pattern, value in _WORD_PATTERNS:
        if re.search(pattern, text):
            return value
    return None


def validate(raw: Any) -> List[Dict[str, Any]]:
    """Оставляет только известные команды с допустимыми значениями."""
    if not isinstance(raw, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "").strip()
        if kind not in _ALLOWED:
            continue
        action: Dict[str, Any] = {"type": kind}
        if kind == "set_settings":
            if isinstance(item.get("power"), (int, float)):
                action["power"] = max(0, min(100, int(item["power"])))
            if isinstance(item.get("cpu_limit"), (int, float)):
                action["cpu_limit"] = max(25, min(100, int(item["cpu_limit"])))
            pref = str(item.get("gpu_preference") or "")
            if pref in ("auto", "integrated", "discrete"):
                action["gpu_preference"] = pref
            if len(action) > 1:
                result.append(action)
        elif kind == "set_ai_mode":
            action["enabled"] = bool(item.get("enabled", True))
            action["prompt"] = str(item.get("prompt") or "")[:600]
            result.append(action)
        elif kind == "start_cut":
            for key in ("url", "project_id", "ai_prompt"):
                value = str(item.get(key) or "").strip()
                if value:
                    action[key] = value[:600]
            for key, low, high in (("count", 1, 50), ("min_dur", 3, 120), ("max_dur", 5, 180)):
                if isinstance(item.get(key), (int, float)):
                    action[key] = max(low, min(high, int(item[key])))
            if str(item.get("aspect") or "") in ("9:16", "1:1", "16:9", "4:5"):
                action["aspect"] = item["aspect"]
            if "ai_mode" in item:
                action["ai_mode"] = bool(item["ai_mode"])
            if action.get("url") or action.get("project_id"):
                result.append(action)
        elif kind == "advice":
            pid = str(item.get("project_id") or "").strip()
            if pid:
                action["project_id"] = pid
                result.append(action)
        elif kind == "open_project":
            pid = str(item.get("project_id") or "").strip()
            if pid:
                action["project_id"] = pid
                result.append(action)
        elif kind == "open_settings":
            result.append(action)
    return result


def parse_from_text(text: str) -> List[Dict[str, Any]]:
    """Вытаскивает JSON-блок с командами из ответа модели."""
    cleaned = (text or "").strip()
    start = cleaned.rfind("{")
    while start != -1:
        fragment = cleaned[start:]
        end = fragment.find("}")
        while end != -1:
            candidate = fragment[: end + 1]
            try:
                import json

                data = json.loads(candidate)
            except ValueError:
                end = fragment.find("}", end + 1)
                continue
            if isinstance(data, dict) and isinstance(data.get("actions"), list):
                return validate(data["actions"])
            end = fragment.find("}", end + 1)
        start = cleaned.rfind("{", 0, start)
    return []


def strip_actions(text: str) -> str:
    """Убирает JSON-блок с командами из текста ответа."""
    cleaned = text or ""
    match = re.search(r"```(?:json)?\s*\{\s*\"actions\".*?\}\s*```", cleaned, re.S)
    if match:
        cleaned = cleaned[: match.start()] + cleaned[match.end():]
    else:
        start = cleaned.rfind("{\"actions\"")
        if start != -1:
            cleaned = cleaned[:start]
    return cleaned.strip()


def parse_local(question: str) -> List[Dict[str, Any]]:
    """Разбирает просьбу без модели: частые формулировки на русском."""
    text = (question or "").lower()
    if not text.strip():
        return []
    actions: List[Dict[str, Any]] = []
    settings_action: Dict[str, Any] = {"type": "set_settings"}

    if re.search(r"(лимит|потолок|не более|огранич).*(процессор|cpu|ядер)", text) or \
            re.search(r"(процессор|cpu).*(лимит|потолок|не более)", text):
        value = _number(text)
        if value:
            settings_action["cpu_limit"] = max(25, min(100, value))
    if "мощност" in text:
        value = _number(text)
        if value:
            settings_action["power"] = max(0, min(100, value))
    if "дискретн" in text:
        settings_action["gpu_preference"] = "discrete"
    elif "встроенн" in text and "карт" in text:
        settings_action["gpu_preference"] = "integrated"
    elif "авто" in text and "карт" in text:
        settings_action["gpu_preference"] = "auto"
    if len(settings_action) > 1:
        actions.append(settings_action)

    if re.search(r"(включ|вруб).*(ии|ии-режим|иишк)", text):
        prompt = ""
        match = re.search(r"задани[ея][:\s]+(.+)", question, re.I)
        if match:
            prompt = match.group(1).strip()[:600]
        actions.append({"type": "set_ai_mode", "enabled": True, "prompt": prompt})

    cut = re.search(r"(нарежь|нарез|сделай .*клип|порежь|собери клип)", text)
    if cut:
        action: Dict[str, Any] = {"type": "start_cut"}
        if re.search(r"(ии|ии-режим|иишк|модел)", text):
            action["ai_mode"] = True
        count = re.search(r"(\d{1,2})\s*(клип|нарез|штук)", text)
        if count:
            action["count"] = max(1, min(50, int(count.group(1))))
        aspect = re.search(r"(9:16|1:1|16:9|4:5)", text)
        if aspect:
            action["aspect"] = aspect.group(1)
        pid = re.search(r"([0-9a-f]{12})", text)
        if pid:
            action["project_id"] = pid.group(1)
        link = re.search(r"https?://\S+", question or "")
        if link:
            action["url"] = link.group(0).rstrip(".,)")
        actions.append(action)

    if re.search(r"(открой|покажи).*(настройк)", text):
        actions.append({"type": "open_settings"})
    return actions


def execute(action: Dict[str, Any],
            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Выполняет одну команду. Возвращает результат для интерфейса."""
    kind = action.get("type")
    if kind == "set_settings":
        from . import ai as ai_module  # noqa: F401 — ключи не трогаем
        from . import cpu_limit, gpu_pref, settings

        patch = {key: action[key] for key in ("power", "cpu_limit", "gpu_preference")
                 if key in action}
        settings.update(patch)
        resources.invalidate()
        cpu_limit.apply()
        gpu_pref.apply()
        summary = resources.summary()
        return {"ok": True, "type": kind, "applied": patch,
                "summary": {key: summary.get(key) for key in
                            ("power", "cpu_limit", "render_workers", "ffmpeg_threads")},
                "message": "Настройки применены: " + ", ".join(
                    f"{key}={value}" for key, value in patch.items())}

    if kind == "set_ai_mode":
        # режим живёт в интерфейсе, но запоминаем задание в настройках проекта
        return {"ok": True, "type": kind, "ui": {"ai_mode": bool(action.get("enabled")),
                                                 "prompt": action.get("prompt") or ""},
                "message": ("ИИ-режим включён" if action.get("enabled")
                            else "ИИ-режим выключен")}

    if kind == "start_cut":
        return _start_cut(action, context or {})

    if kind == "advice":
        from . import ai as ai_module

        project = db.get_project(action.get("project_id") or "")
        if not project:
            return {"ok": False, "type": kind,
                    "message": "Проект не найден — сначала запустите нарезку."}
        clips = db.list_clips(project["id"])
        clip = max(clips, key=lambda item: float(item.get("score") or 0.0)) if clips else None
        payload = {"title": project.get("title"), "language": project.get("language"),
                   "duration": project.get("duration"), "clips": clips}
        try:
            result = ai_module.advice(payload, clip)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "type": kind, "message": str(exc)[:300]}
        return {"ok": True, "type": kind, "advice": result.get("advice"),
                "model": result.get("model"), "message": "Совет готов"}

    if kind == "open_settings":
        return {"ok": True, "type": kind, "ui": {"open_settings": True},
                "message": "Открываю настройки"}

    if kind == "open_project":
        pid = action.get("project_id") or ""
        if not db.get_project(pid):
            return {"ok": False, "type": kind, "message": "Проект не найден"}
        return {"ok": True, "type": kind, "ui": {"open_project": pid},
                "message": "Открываю проект"}

    return {"ok": False, "type": str(kind), "message": "Неизвестное действие"}


def _start_cut(action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Запускает нарезку: по ссылке или по уже загруженному проекту."""
    options: Dict[str, Any] = {}
    for key in ("count", "min_dur", "max_dur", "aspect"):
        if key in action:
            options[key] = action[key]
    if action.get("ai_mode"):
        options["ai_mode"] = True
        options["autostart"] = True
        options["ai_prompt"] = action.get("ai_prompt") or ""
    project_id = str(action.get("project_id") or "").strip()
    url = str(action.get("url") or "").strip()

    if not project_id and not url:
        current = (context.get("project") or {})
        if current.get("id"):
            project_id = str(current["id"])
        else:
            return {"ok": False, "type": "start_cut",
                    "message": "Сначала пришлите ссылку или загрузите видео."}

    if project_id:
        project = db.get_project(project_id)
        if not project:
            return {"ok": False, "type": "start_cut", "message": "Проект не найден."}
        status = str(project.get("status") or "")
        if status in ("queued", "preparing", "processing", "running"):
            return {"ok": True, "type": "start_cut", "project_id": project_id,
                    "message": "Нарезка уже идёт — подождите результат."}
        db.update_project(project_id, options=pipeline.normalize_options(
            {**(project.get("options") or {}), **options}))
        if status in ("new", "", "error", "cancelled"):
            jobs.submit_prepare(project_id)
        else:
            jobs.submit_process(project_id)
        return {"ok": True, "type": "start_cut", "project_id": project_id,
                "message": "Запустил нарезку проекта — клипы появятся сами."}

    normalized = pipeline.normalize_options(options)
    project = db.create_project("pending", normalized, title=url[:120] or "Новый проект")
    db.update_project(project["id"], url=url)
    jobs.submit_prepare(project["id"])
    return {"ok": True, "type": "start_cut", "project_id": project["id"],
            "message": "Принял ссылку — скачиваю и нарезаю."}
