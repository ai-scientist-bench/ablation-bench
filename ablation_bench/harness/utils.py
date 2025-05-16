"""Common utilities for ablations-bench harness."""

from pydantic_settings import BaseSettings


def get_field_description(settings_class: type[BaseSettings], field_name: str) -> str:
    """Get the description from a Pydantic field."""
    return settings_class.model_fields[field_name].description or ""
