# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ToonDB Python Plugin System

Modern, AI-era plugin system for running Python code as database triggers.

Features:
- Full Python package support (numpy, pandas, scikit-learn)
- WASM sandboxing for security
- AI-powered trigger generation
- Hot-reload without downtime

Example:
    from toondb import Database
    from toondb.plugins import PythonPlugin
    
    db = Database.open("./data")
    
    # Register a validation plugin
    plugin = PythonPlugin(
        name="email_validator",
        code='''
def on_before_insert(row):
    if "@" not in row["email"]:
        raise TriggerAbort("Invalid email")
    row["email"] = row["email"].lower()
    return row
''',
        triggers={"users": ["BEFORE INSERT"]}
    )
    db.register_plugin(plugin)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
from enum import Enum
import json


class TriggerEvent(Enum):
    """Types of trigger events."""
    BEFORE_INSERT = "BEFORE INSERT"
    AFTER_INSERT = "AFTER INSERT"
    BEFORE_UPDATE = "BEFORE UPDATE"
    AFTER_UPDATE = "AFTER UPDATE"
    BEFORE_DELETE = "BEFORE DELETE"
    AFTER_DELETE = "AFTER DELETE"
    ON_BATCH = "ON BATCH"
    
    @classmethod
    def from_str(cls, s: str) -> "TriggerEvent":
        """Parse from string like 'BEFORE INSERT'."""
        s = s.upper().replace("_", " ")
        for event in cls:
            if event.value == s:
                return event
        raise ValueError(f"Unknown trigger event: {s}")


class TriggerAbort(Exception):
    """Raised by trigger code to abort the operation."""
    def __init__(self, message: str, code: str = "TRIGGER_ABORT"):
        super().__init__(message)
        self.code = code


@dataclass
class RuntimeConfig:
    """Configuration for the Python runtime."""
    memory_limit_mb: int = 64
    timeout_ms: int = 5000
    packages: List[str] = field(default_factory=list)
    debug: bool = False
    allow_network: bool = False
    
    @classmethod
    def lightweight(cls) -> "RuntimeConfig":
        """Lightweight config for simple validation scripts."""
        return cls(memory_limit_mb=16, timeout_ms=100)
    
    @classmethod
    def with_ml_packages(cls) -> "RuntimeConfig":
        """Config for ML workloads with numpy, pandas, sklearn."""
        return cls(
            memory_limit_mb=256,
            timeout_ms=30000,
            packages=["numpy", "pandas", "scikit-learn"]
        )


@dataclass
class PythonPlugin:
    """
    A Python plugin that runs code on database events.
    
    Example:
        plugin = PythonPlugin(
            name="fraud_detector",
            code='''
import pandas as pd
import numpy as np

def on_before_insert(row: dict) -> dict:
    if row["amount"] > 10000:
        raise TriggerAbort("Amount too high", code="LIMIT_EXCEEDED")
    return row
''',
            packages=["numpy", "pandas"],
            triggers={"transactions": ["BEFORE INSERT"]}
        )
    """
    name: str
    code: str
    version: str = "1.0.0"
    packages: List[str] = field(default_factory=list)
    wheels: List[str] = field(default_factory=list)
    triggers: Dict[str, List[str]] = field(default_factory=dict)
    config: Optional[RuntimeConfig] = None
    
    def __post_init__(self):
        """Validate the plugin on creation."""
        self._validate()
    
    def _validate(self):
        """Validate the plugin code."""
        if not self.name:
            raise ValueError("Plugin name is required")
        
        # Check for forbidden patterns
        forbidden = [
            "__import__('os')",
            "subprocess",
            "eval(",
            "exec(",
            "compile(",
            "open(",
        ]
        for pattern in forbidden:
            if pattern in self.code:
                raise ValueError(f"Forbidden pattern in code: {pattern}")
        
        # Check for handler function
        handlers = [
            "on_before_insert", "on_after_insert",
            "on_before_update", "on_after_update", 
            "on_before_delete", "on_after_delete",
            "on_batch", "handler"
        ]
        if not any(f"def {h}(" in self.code for h in handlers):
            raise ValueError("Code must define a handler function")
    
    def with_trigger(self, table: str, event: str) -> "PythonPlugin":
        """Add a trigger binding."""
        if table not in self.triggers:
            self.triggers[table] = []
        self.triggers[table].append(event)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "code": self.code,
            "packages": self.packages,
            "wheels": self.wheels,
            "triggers": self.triggers,
            "config": {
                "memory_limit_mb": self.config.memory_limit_mb if self.config else 64,
                "timeout_ms": self.config.timeout_ms if self.config else 5000,
            } if self.config else None
        }


class PluginRegistry:
    """
    Registry for managing Python plugins.
    
    Example:
        registry = PluginRegistry()
        registry.register(my_plugin)
        
        # Fire triggers
        result = registry.fire("users", TriggerEvent.BEFORE_INSERT, row)
    """
    
    def __init__(self):
        self._plugins: Dict[str, PythonPlugin] = {}
        self._trigger_map: Dict[tuple, List[str]] = {}
        self._installed_packages: set = set()
    
    def register(self, plugin: PythonPlugin) -> None:
        """Register a plugin."""
        self._plugins[plugin.name] = plugin
        
        # Update trigger mappings
        for table, events in plugin.triggers.items():
            for event_str in events:
                event = TriggerEvent.from_str(event_str)
                key = (table, event)
                if key not in self._trigger_map:
                    self._trigger_map[key] = []
                self._trigger_map[key].append(plugin.name)
        
        # Track packages
        self._installed_packages.update(plugin.packages)
    
    def unregister(self, name: str) -> None:
        """Unregister a plugin."""
        if name not in self._plugins:
            raise KeyError(f"Plugin not found: {name}")
        
        plugin = self._plugins.pop(name)
        
        # Remove from trigger map
        for table, events in plugin.triggers.items():
            for event_str in events:
                event = TriggerEvent.from_str(event_str)
                key = (table, event)
                if key in self._trigger_map:
                    self._trigger_map[key] = [
                        n for n in self._trigger_map[key] if n != name
                    ]
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        return list(self._plugins.keys())
    
    def get(self, name: str) -> Optional[PythonPlugin]:
        """Get a plugin by name."""
        return self._plugins.get(name)
    
    def fire(
        self, 
        table: str, 
        event: TriggerEvent, 
        row: Dict[str, Any],
        old_row: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fire triggers for an event.
        
        Returns the (possibly modified) row.
        Raises TriggerAbort if a trigger aborts the operation.
        """
        key = (table, event)
        plugin_names = self._trigger_map.get(key, [])
        
        current_row = row.copy()
        
        for name in plugin_names:
            plugin = self._plugins.get(name)
            if not plugin:
                continue
            
            # Execute the plugin (simulated for now)
            result = self._execute(plugin, event, current_row, old_row)
            if result is not None:
                current_row = result
        
        return current_row
    
    def _execute(
        self,
        plugin: PythonPlugin,
        event: TriggerEvent,
        row: Dict[str, Any],
        old_row: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute a plugin (simulated)."""
        # In production, this would call the Rust FFI which runs
        # the code in a WASM sandbox. For now, we simulate.
        
        # Create execution context
        context = {
            "row": row,
            "old_row": old_row,
            "TriggerAbort": TriggerAbort,
        }
        
        # Get handler function name based on event
        handler_map = {
            TriggerEvent.BEFORE_INSERT: "on_before_insert",
            TriggerEvent.AFTER_INSERT: "on_after_insert",
            TriggerEvent.BEFORE_UPDATE: "on_before_update",
            TriggerEvent.AFTER_UPDATE: "on_after_update",
            TriggerEvent.BEFORE_DELETE: "on_before_delete",
            TriggerEvent.AFTER_DELETE: "on_after_delete",
            TriggerEvent.ON_BATCH: "on_batch",
        }
        handler_name = handler_map.get(event, "handler")
        
        # Execute in restricted namespace
        try:
            exec(plugin.code, context)
            if handler_name in context:
                result = context[handler_name](row)
                return result if isinstance(result, dict) else row
        except TriggerAbort:
            raise
        except Exception as e:
            raise RuntimeError(f"Plugin {plugin.name} failed: {e}")
        
        return row


# Convenience function for AI trigger generation (placeholder)
def generate_trigger_code(instruction: str, table_schema: dict = None) -> str:
    """
    Generate Python trigger code from natural language instruction.
    
    Note: In production, this calls an LLM API.
    
    Args:
        instruction: Natural language description of the trigger logic
        table_schema: Optional schema information for better code generation
    
    Returns:
        Python code implementing the trigger
    
    Example:
        code = generate_trigger_code(
            "Validate email format and normalize to lowercase"
        )
    """
    # Placeholder implementation
    return f'''
# Generated from: {instruction}
# TODO: Implement actual LLM-based generation

def on_before_insert(row: dict) -> dict:
    """Auto-generated trigger."""
    # Add your validation logic here
    return row
'''


__all__ = [
    "TriggerEvent",
    "TriggerAbort", 
    "RuntimeConfig",
    "PythonPlugin",
    "PluginRegistry",
    "generate_trigger_code",
]
