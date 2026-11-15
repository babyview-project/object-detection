"""Category filters for valid7018 figures and public release.

``BODY_PART_CATEGORIES`` — CDI body-part labels (excluded from figure picks).
``MONTAGE_EXCLUDE_CATEGORIES`` — extra privacy exclusions (faces in crop).
``PUBLIC_EXCLUDE_CATEGORIES`` — union used for public crop/embedding release.
"""
from __future__ import annotations

BODY_PART_CATEGORIES = frozenset(
    {
        "ankle",
        "arm",
        "ear",
        "eye",
        "face",
        "finger",
        "foot",
        "hair",
        "hand",
        "leg",
        "mouth",
        "neck",
        "nose",
        "toe",
        "tooth",
    }
)

# Montage / privacy: exemplars routinely show faces (e.g. worn glasses).
MONTAGE_EXCLUDE_CATEGORIES = frozenset({"glasses"})

PUBLIC_EXCLUDE_CATEGORIES = BODY_PART_CATEGORIES | MONTAGE_EXCLUDE_CATEGORIES

PUBLIC_EXCLUDE_REASONS: dict[str, str] = {
    **{c: "body_part" for c in sorted(BODY_PART_CATEGORIES)},
    **{c: "face_privacy" for c in sorted(MONTAGE_EXCLUDE_CATEGORIES)},
}
