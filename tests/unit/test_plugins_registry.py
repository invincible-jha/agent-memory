"""Unit tests for agent_memory.plugins.registry — PluginRegistry."""

from __future__ import annotations

import importlib.metadata
import logging
from abc import ABC, abstractmethod
from unittest.mock import MagicMock, patch

import pytest

from agent_memory.plugins.registry import (
    PluginAlreadyRegisteredError,
    PluginNotFoundError,
    PluginRegistry,
)


# ---------------------------------------------------------------------------
# Test base class and concrete implementations
# ---------------------------------------------------------------------------


class BaseWidget(ABC):
    @abstractmethod
    def render(self) -> str: ...


class ConcreteWidget(BaseWidget):
    def render(self) -> str:
        return "widget"


class AnotherWidget(BaseWidget):
    def render(self) -> str:
        return "another"


# ---------------------------------------------------------------------------
# PluginNotFoundError
# ---------------------------------------------------------------------------


class TestPluginNotFoundError:
    def test_stores_plugin_name(self) -> None:
        err = PluginNotFoundError("missing-plugin", "test-registry")
        assert err.plugin_name == "missing-plugin"

    def test_stores_registry_name(self) -> None:
        err = PluginNotFoundError("missing-plugin", "test-registry")
        assert err.registry_name == "test-registry"

    def test_is_subclass_of_key_error(self) -> None:
        err = PluginNotFoundError("x", "y")
        assert isinstance(err, KeyError)

    def test_message_contains_plugin_name(self) -> None:
        err = PluginNotFoundError("my-plugin", "widgets")
        assert "my-plugin" in str(err)


# ---------------------------------------------------------------------------
# PluginAlreadyRegisteredError
# ---------------------------------------------------------------------------


class TestPluginAlreadyRegisteredError:
    def test_stores_plugin_name(self) -> None:
        err = PluginAlreadyRegisteredError("dup-plugin", "my-registry")
        assert err.plugin_name == "dup-plugin"

    def test_stores_registry_name(self) -> None:
        err = PluginAlreadyRegisteredError("dup-plugin", "my-registry")
        assert err.registry_name == "my-registry"

    def test_is_subclass_of_value_error(self) -> None:
        err = PluginAlreadyRegisteredError("x", "y")
        assert isinstance(err, ValueError)


# ---------------------------------------------------------------------------
# PluginRegistry construction and __repr__
# ---------------------------------------------------------------------------


class TestPluginRegistryInit:
    def test_starts_empty(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "widgets")
        assert len(registry) == 0

    def test_repr_contains_name_and_base_class(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "widgets")
        r = repr(registry)
        assert "widgets" in r
        assert "BaseWidget" in r

    def test_list_plugins_empty(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "widgets")
        assert registry.list_plugins() == []


# ---------------------------------------------------------------------------
# @register decorator
# ---------------------------------------------------------------------------


class TestRegisterDecorator:
    def test_decorator_registers_class(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")

        @registry.register("my-widget")
        class MyWidget(BaseWidget):
            def render(self) -> str:
                return "my"

        assert "my-widget" in registry

    def test_decorator_returns_class_unchanged(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")

        @registry.register("identity-widget")
        class IdentityWidget(BaseWidget):
            def render(self) -> str:
                return "id"

        instance = IdentityWidget()
        assert instance.render() == "id"

    def test_duplicate_registration_raises(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")

        @registry.register("dup")
        class First(BaseWidget):
            def render(self) -> str:
                return "first"

        with pytest.raises(PluginAlreadyRegisteredError):

            @registry.register("dup")
            class Second(BaseWidget):
                def render(self) -> str:
                    return "second"

    def test_type_error_when_not_subclass(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")

        with pytest.raises(TypeError):

            @registry.register("bad")
            class NotAWidget:  # type: ignore[misc]
                pass


# ---------------------------------------------------------------------------
# register_class
# ---------------------------------------------------------------------------


class TestRegisterClass:
    def test_register_class_directly(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")
        registry.register_class("direct", ConcreteWidget)
        assert "direct" in registry

    def test_register_class_duplicate_raises(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")
        registry.register_class("one", ConcreteWidget)
        with pytest.raises(PluginAlreadyRegisteredError):
            registry.register_class("one", AnotherWidget)

    def test_register_class_type_error_when_not_subclass(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")
        with pytest.raises(TypeError):
            registry.register_class("bad", object)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# deregister
# ---------------------------------------------------------------------------


class TestDeregister:
    def test_deregister_removes_plugin(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")
        registry.register_class("temp", ConcreteWidget)
        registry.deregister("temp")
        assert "temp" not in registry

    def test_deregister_missing_plugin_raises(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")
        with pytest.raises(PluginNotFoundError):
            registry.deregister("nonexistent")

    def test_deregister_reduces_len(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")
        registry.register_class("a", ConcreteWidget)
        registry.register_class("b", AnotherWidget)
        registry.deregister("a")
        assert len(registry) == 1


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_returns_registered_class(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")
        registry.register_class("concrete", ConcreteWidget)
        cls = registry.get("concrete")
        assert cls is ConcreteWidget

    def test_get_missing_raises_plugin_not_found(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")
        with pytest.raises(PluginNotFoundError):
            registry.get("nonexistent")

    def test_get_returned_class_is_instantiable(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")
        registry.register_class("concrete", ConcreteWidget)
        cls = registry.get("concrete")
        instance = cls()
        assert instance.render() == "widget"


# ---------------------------------------------------------------------------
# list_plugins
# ---------------------------------------------------------------------------


class TestListPlugins:
    def test_list_plugins_sorted_alphabetically(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")
        registry.register_class("beta", ConcreteWidget)
        registry.register_class("alpha", AnotherWidget)
        assert registry.list_plugins() == ["alpha", "beta"]

    def test_list_plugins_after_deregister(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")
        registry.register_class("a", ConcreteWidget)
        registry.register_class("b", AnotherWidget)
        registry.deregister("a")
        assert registry.list_plugins() == ["b"]


# ---------------------------------------------------------------------------
# __contains__ and __len__
# ---------------------------------------------------------------------------


class TestMagicMethods:
    def test_contains_registered_name(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")
        registry.register_class("my", ConcreteWidget)
        assert "my" in registry

    def test_not_contains_unregistered_name(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")
        assert "ghost" not in registry

    def test_len_reflects_registration_count(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")
        assert len(registry) == 0
        registry.register_class("one", ConcreteWidget)
        assert len(registry) == 1
        registry.register_class("two", AnotherWidget)
        assert len(registry) == 2


# ---------------------------------------------------------------------------
# load_entrypoints
# ---------------------------------------------------------------------------


class TestLoadEntrypoints:
    def test_load_entrypoints_registers_valid_plugin(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")

        mock_ep = MagicMock()
        mock_ep.name = "ep-widget"
        mock_ep.load.return_value = ConcreteWidget

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("test.group")

        assert "ep-widget" in registry

    def test_load_entrypoints_skips_already_registered(self, caplog: pytest.LogCaptureFixture) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")
        registry.register_class("ep-widget", ConcreteWidget)

        mock_ep = MagicMock()
        mock_ep.name = "ep-widget"
        mock_ep.load.return_value = ConcreteWidget

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            with caplog.at_level(logging.DEBUG):
                registry.load_entrypoints("test.group")

        # Should still only have 1 registration (not duplicated)
        assert len(registry) == 1

    def test_load_entrypoints_skips_on_load_error(self, caplog: pytest.LogCaptureFixture) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")

        mock_ep = MagicMock()
        mock_ep.name = "broken-ep"
        mock_ep.load.side_effect = ImportError("missing dep")

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            with caplog.at_level(logging.ERROR):
                registry.load_entrypoints("test.group")

        assert "broken-ep" not in registry

    def test_load_entrypoints_skips_invalid_subclass(self, caplog: pytest.LogCaptureFixture) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")

        mock_ep = MagicMock()
        mock_ep.name = "non-widget"
        mock_ep.load.return_value = object  # Not a BaseWidget subclass

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            with caplog.at_level(logging.WARNING):
                registry.load_entrypoints("test.group")

        assert "non-widget" not in registry

    def test_load_entrypoints_is_idempotent(self) -> None:
        registry: PluginRegistry[BaseWidget] = PluginRegistry(BaseWidget, "test-reg")

        mock_ep = MagicMock()
        mock_ep.name = "idempotent-widget"
        mock_ep.load.return_value = ConcreteWidget

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("test.group")
            registry.load_entrypoints("test.group")  # second call is a no-op

        assert len(registry) == 1
