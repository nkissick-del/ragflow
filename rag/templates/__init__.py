#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Template Registry with Auto-Discovery.

This module provides automatic discovery of chunking templates. Templates are
Python modules in the `rag/templates/` directory that have a `chunk()` function.

Usage:
    from rag.templates import TEMPLATE_REGISTRY, get_template, list_templates

    # Get a template by name
    template = get_template("semantic")
    chunks = template.chunk(filename, binary=data, ...)

    # List all available templates for UI dropdown
    options = list_templates()  # Returns [{"value": "semantic", "label": "Semantic"}, ...]
"""

import importlib
import logging
import pkgutil
import threading
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Template metadata for UI display (display name, description, sort order)
# Templates not listed here will use auto-generated labels
TEMPLATE_METADATA = {
    "naive": {"label": "General", "description": "Token-based chunking", "order": 0, "hidden": False},
    "semantic": {"label": "Semantic", "description": "Structure-aware chunking with header hierarchy", "order": 1, "hidden": False},
    "qa": {"label": "Q&A", "description": "Question and answer extraction", "order": 2, "hidden": False},
    "paper": {"label": "Paper", "description": "Academic paper parsing", "order": 3, "hidden": False},
    "book": {"label": "Book", "description": "Book/chapter parsing", "order": 4, "hidden": False},
    "laws": {"label": "Laws", "description": "Legal document parsing", "order": 5, "hidden": False},
    "manual": {"label": "Manual", "description": "Technical manual parsing", "order": 6, "hidden": False},
    "presentation": {"label": "Presentation", "description": "Slide presentation parsing", "order": 7, "hidden": False},
    "resume": {"label": "Resume", "description": "Resume/CV parsing", "order": 8, "hidden": False},
    "table": {"label": "Table", "description": "Tabular data extraction", "order": 9, "hidden": False},
    "single_chunk": {"label": "One", "description": "Single chunk for entire document", "order": 10, "hidden": False},
    # Hidden templates (not shown in UI dropdown)
    "picture": {"label": "Picture", "description": "Image processing", "order": 100, "hidden": True},
    "audio": {"label": "Audio", "description": "Audio transcription", "order": 101, "hidden": True},
    "email": {"label": "Email", "description": "Email parsing", "order": 102, "hidden": True},
    "general": {"label": "General (Alias)", "description": "Alias for naive", "order": 103, "hidden": True},
}


class TemplateRegistry:
    """Registry for automatically discovered chunking templates."""

    def __init__(self):
        self._templates: Dict[str, Any] = {}
        self._discovered = False
        self._lock = threading.Lock()

    def discover(self) -> None:
        """
        Auto-discover all template modules with a chunk() function.

        Templates are Python modules in the same directory as this file.
        A valid template must have a `chunk` function at module level.
        """
        if self._discovered:
            return

        with self._lock:
            if self._discovered:
                return

            # Import all modules in this package
            package_path = __path__  # type: ignore
            for importer, modname, ispkg in pkgutil.iter_modules(package_path):
                if modname.startswith("_") or ispkg:
                    continue

                try:
                    module = importlib.import_module(f".{modname}", __name__)
                    if hasattr(module, "chunk") and callable(getattr(module, "chunk")):
                        self._templates[modname] = module
                        logger.debug(f"[TemplateRegistry] Discovered template: {modname}")
                    else:
                        logger.debug(f"[TemplateRegistry] Skipped {modname} (no chunk function)")
                except Exception as e:
                    logger.warning(f"[TemplateRegistry] Failed to import template {modname}: {e}")

            self._discovered = True
            logger.info(f"[TemplateRegistry] Discovered {len(self._templates)} templates")

    def get(self, name: str) -> Optional[Any]:
        """
        Get a template module by name.

        Args:
            name: Template name (e.g., "semantic", "naive")

        Returns:
            Template module or None if not found
        """
        self.discover()
        return self._templates.get(name.lower())

    def list_for_ui(self, include_hidden: bool = False) -> List[Dict[str, str]]:
        """
        List templates formatted for UI dropdown display.

        Args:
            include_hidden: If True, include hidden templates (audio, picture, etc.)

        Returns:
            List of dicts with "value" and "label" keys, sorted by order
        """
        self.discover()
        result = []

        for name in self._templates.keys():
            meta = TEMPLATE_METADATA.get(name, {})

            # Skip hidden templates unless explicitly requested
            if meta.get("hidden", False) and not include_hidden:
                continue

            # Generate label from name if not in metadata
            label = meta.get("label", name.replace("_", " ").title())
            order = meta.get("order", 50)  # Default to middle

            result.append(
                {
                    "value": name,
                    "label": label,
                    "order": order,
                }
            )

        # Sort by order, then alphabetically by label
        result.sort(key=lambda x: (x["order"], x["label"]))

        # Remove order from output (internal use only)
        return [{"value": r["value"], "label": r["label"]} for r in result]

    def list_all(self) -> List[str]:
        """List all discovered template names."""
        self.discover()
        return list(self._templates.keys())

    def get_factory_dict(self) -> Dict[str, Any]:
        """
        Get a dict mapping template names to modules.

        This is compatible with task_executor.py's FACTORY format.

        Returns:
            Dict mapping template name -> module
        """
        self.discover()
        return dict(self._templates)

    def __contains__(self, name: str) -> bool:
        self.discover()
        return name.lower() in self._templates

    def __getitem__(self, name: str) -> Any:
        self.discover()
        template = self._templates.get(name.lower())
        if template is None:
            raise KeyError(f"Template not found: {name}")
        return template


# Global registry instance
TEMPLATE_REGISTRY = TemplateRegistry()


def get_template(name: str) -> Optional[Any]:
    """Get a template module by name."""
    return TEMPLATE_REGISTRY.get(name)


def list_templates(include_hidden: bool = False) -> List[Dict[str, str]]:
    """List templates for UI dropdown display."""
    return TEMPLATE_REGISTRY.list_for_ui(include_hidden)


def get_factory() -> Dict[str, Any]:
    """Get FACTORY-compatible dict for task_executor."""
    return TEMPLATE_REGISTRY.get_factory_dict()
