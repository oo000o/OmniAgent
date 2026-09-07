"""Shared path helpers for workspace-scoped tools."""

import re
import unicodedata
from pathlib import Path

from nanobot.config.paths import get_media_dir
from nanobot.security.workspace_policy import resolve_allowed_path

_MEDIA_FILENAME_PUNCTUATION_SPACING = re.compile(
    r"\s*([\-\u2010-\u2015()\[\]{}（）【】])\s*"
)


def _media_filename_key(name: str) -> str:
    """Normalize only presentation spacing around common filename punctuation."""
    normalized = unicodedata.normalize("NFKC", name)
    return _MEDIA_FILENAME_PUNCTUATION_SPACING.sub(r"\1", normalized)


def resolve_unique_media_file(requested: Path) -> Path | None:
    """Recover one media filename when an LLM only changed punctuation spacing.

    The lookup stays in the requested media subdirectory, rejects symlinks that
    escape the media root, and succeeds only when there is one unique match.
    """
    media_root = get_media_dir().expanduser().resolve(strict=False)
    parent = requested.parent.expanduser().resolve(strict=False)
    try:
        parent.relative_to(media_root)
    except ValueError:
        return None
    if not parent.is_dir():
        return None

    requested_key = _media_filename_key(requested.name)
    matches: list[Path] = []
    try:
        for entry in parent.iterdir():
            if _media_filename_key(entry.name) != requested_key:
                continue
            candidate = entry.resolve(strict=False)
            try:
                candidate.relative_to(media_root)
            except ValueError:
                continue
            if candidate.is_file():
                matches.append(candidate)
    except OSError:
        return None
    return matches[0] if len(matches) == 1 else None


def resolve_workspace_path(
    path: str,
    workspace: Path | None = None,
    allowed_dir: Path | None = None,
    extra_allowed_dirs: list[Path] | None = None,
    extra_allowed_files: list[Path] | None = None,
    include_media_dir: bool = True,
) -> Path:
    """Resolve path against workspace and enforce allowed directory containment."""
    media_roots = [get_media_dir()] if include_media_dir else []
    extra_roots = [*media_roots, *(extra_allowed_dirs or [])] if allowed_dir else None
    return resolve_allowed_path(
        path,
        workspace=workspace,
        allowed_root=allowed_dir,
        extra_allowed_roots=extra_roots,
        extra_allowed_files=extra_allowed_files,
    )
