"""Utilities for loading stage settings from the shared YAML configuration."""

from pathlib import Path
from typing import Any, Iterable
import os

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")

_PATH_KEYS = {
    "step1": ("track_file", "data_dir", "workdir"),
    "step2": ("indir", "outdir"),
}


def load_stage_config(
    config_path: str,
    section: str,
    stage: str,
) -> dict[str, Any]:
    """Load section.stage and resolve its relative paths from the YAML location."""
    path = Path(os.path.expandvars(config_path)).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}

    if not isinstance(document, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    if section not in document or not isinstance(document[section], dict):
        raise ValueError(f"Missing mapping '{section}' in configuration: {path}")
    section_data = document[section]
    if stage not in section_data or not isinstance(section_data[stage], dict):
        raise ValueError(
            f"Missing mapping '{section}.{stage}' in configuration: {path}"
        )

    settings = dict(section_data[stage])
    for key in _PATH_KEYS.get(stage, ()):
        value = settings.get(key)
        if value is None:
            continue
        value_path = Path(os.path.expandvars(str(value))).expanduser()
        if not value_path.is_absolute():
            value_path = path.parent / value_path
        settings[key] = str(value_path.resolve())

    return settings


def render_output_filename(template: str, **values: Any) -> str:
    """Render and validate one output filename supplied by configuration."""
    try:
        filename = str(template).format(**values)
    except KeyError as exc:
        raise ValueError(
            f"Unknown placeholder {exc!s} in output filename: {template}"
        ) from exc
    if not filename or Path(filename).name != filename:
        raise ValueError(
            f"Output filename must be a name without directory components: {filename!r}"
        )
    return filename


def apply_config_defaults(
    args: Any,
    settings: dict[str, Any],
    required: Iterable[str],
) -> Any:
    """Fill CLI options from YAML while preserving explicit CLI overrides."""
    missing = []
    for key in required:
        if getattr(args, key, None) is not None:
            continue
        if key not in settings:
            missing.append(key)
            continue
        setattr(args, key, settings[key])

    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required configuration value(s): {joined}")
    return args
