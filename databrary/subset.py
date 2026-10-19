"""Select a diverse VEDB/Databrary session subset for world-camera downloads."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _measures(session: dict[str, Any], category_id: int) -> dict[str, Any]:
    for rec in session.get("fileRecords") or []:
        if rec.get("categoryId") == category_id:
            return rec.get("measures") or {}
    return {}


def session_task(session: dict[str, Any]) -> str:
    measures = _measures(session, 6)
    return str(measures.get("29") or measures.get("30") or "unknown")


def session_setting(session: dict[str, Any]) -> str:
    measures = _measures(session, 7)
    return str(measures.get("33") or "unknown")


def is_world_file(item: dict[str, Any]) -> bool:
    name = str(item.get("name") or item.get("filename") or "").lower()
    return name == "worldprivate" or name.startswith("world")


def pick_world_file(files: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    world = [f for f in files if is_world_file(f)]
    if not world:
        return None
    public = [f for f in world if str(f.get("releaseLevel") or "").lower() == "public"]
    return (public or world)[0]


def world_dest_path(dest_dir: Path, session_id: Any, file_name: Any) -> Path:
    stem = str(file_name or "worldPrivate")
    if not stem.lower().endswith(".mp4"):
        stem = f"{stem}.mp4"
    return Path(dest_dir) / f"session_{session_id}_{stem}"


def stratified_sample(
    sessions: list[dict[str, Any]],
    n: int,
    *,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Round-robin across settings, then tasks, so pond-walks do not dominate."""
    if n <= 0 or not sessions:
        return []
    rng = random.Random(seed)
    by_setting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        by_setting[session_setting(session)].append(session)

    ordered_groups: dict[str, list[dict[str, Any]]] = {}
    for setting, group in by_setting.items():
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for session in group:
            by_task[session_task(session)].append(session)
        task_lists = list(by_task.values())
        rng.shuffle(task_lists)
        for lst in task_lists:
            rng.shuffle(lst)
        mixed: list[dict[str, Any]] = []
        while any(task_lists):
            for lst in task_lists:
                if lst:
                    mixed.append(lst.pop())
        ordered_groups[setting] = mixed

    keys = sorted(ordered_groups)
    chosen: list[dict[str, Any]] = []
    cursor = {key: 0 for key in keys}
    while len(chosen) < n and any(cursor[k] < len(ordered_groups[k]) for k in keys):
        for key in keys:
            if len(chosen) >= n:
                break
            i = cursor[key]
            if i < len(ordered_groups[key]):
                chosen.append(ordered_groups[key][i])
                cursor[key] += 1
    return chosen
