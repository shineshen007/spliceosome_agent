"""Help matches actual interactive handlers, rather than a second command list."""
import ast
import inspect
import textwrap

import pytest

from spliceosome_agent.app import COMMAND_LIST, TUI_HELP, SpliceosomeApp
from spliceosome_agent.commands import command_names
from spliceosome_agent.config import ModelConfig, save_config
from spliceosome_agent.session import HELP, Session


def handled_commands(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            # Only actual dispatch comparisons, not arbitrary strings in messages.
            if ((isinstance(node.left, ast.Name) and node.left.id in {"command", "text"})
                    or (isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Attribute)
                        and node.left.func.attr == "strip")):
                found.update(item.value for value in node.comparators for item in ast.walk(value)
                             if isinstance(item, ast.Constant) and isinstance(item.value, str)
                             and item.value.startswith("/") and " " not in item.value)
    return found


@pytest.mark.parametrize("tui", [False, True])
def test_help_covers_actual_handlers_without_unsupported_commands(tui):
    supported = handled_commands(Session.handle)
    if tui:
        supported |= handled_commands(SpliceosomeApp._handle_async) | handled_commands(SpliceosomeApp._submit)
    text = TUI_HELP if tui else HELP
    rows = [line.split()[0] for line in text.splitlines() if line.startswith("/")]
    assert set(rows) == supported
    assert len(rows) == len(set(rows))
    assert set(command_names(tui=tui)) == supported
    if tui:
        assert COMMAND_LIST == rows
    assert rows[0] == "/tasks"
    assert all(any(line.startswith(command + " ") for line in text.splitlines())
               for command in ("/tasks", "/runs", "/help", "/approve", "/run"))
    assert "/task" not in supported


def test_help_describes_real_tui_repl_differences():
    assert "/current" in TUI_HELP and "/home" in TUI_HELP
    assert "/current" not in HELP and "/home" not in HELP
    assert "/edit [file]" in TUI_HELP and "/edit <file>" in HELP
    assert "Configure the model" in TUI_HELP and "Show model config (never keys)" in HELP
    assert "PgUp/PgDn" in TUI_HELP and "PgUp/PgDn" not in HELP


def test_repl_help_and_tasks_are_read_only_and_no_singular_alias(tmp_path):
    config = tmp_path / "models.json"
    save_config(ModelConfig(), config)
    session = Session(tmp_path / "result", config)
    before = session.store.path.read_bytes()
    assert session.handle("/help") == HELP
    assert session.handle("/tasks") == session.handle("/runs")
    with pytest.raises(ValueError, match="Unknown command"):
        session.handle("/task")
    assert session.store.path.read_bytes() == before
    assert not list(session.store.runs_dir.glob("*/run_state.json"))
