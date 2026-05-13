"""
Expose a simplified public configuration API for the search system.

This module acts as a thin compatibility layer that re-exports selected
configuration objects from the internal ``config`` package, allowing other
parts of the application to import from a single stable location.

Exports:
    get_settings: Factory/helper to load the application settings instance.
    Settings: Typed settings model containing configuration values.
    SearchMode: Enum describing supported search behavior modes.
    PROJECT_ROOT: Absolute path reference to the project root directory.

The ``__all__`` definition explicitly limits what is exported during
``from ... import *`` and documents the intended public interface.
Car Brochure Search System — Centralized Configuration

Re-export settings from config package.
"""

from config import get_settings, Settings, SearchMode, PROJECT_ROOT

__all__ = ["get_settings", "Settings", "SearchMode", "PROJECT_ROOT"]
