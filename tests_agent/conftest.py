from pathlib import Path
import os
import shutil
import subprocess

import pytest

from spliceosome_agent.examples import create_example
from spliceosome_agent.planning import load_study, make_plan
from spliceosome_agent.store import Store

ROOT = Path(__file__).resolve().parents[1]
# Optional project-local R library. A fresh clone has none, and nothing here may assume it exists.
R_LIBRARY = Path(os.environ.get("SPLICEOSOME_R_LIBRARY") or ROOT / ".r-library")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib-tests"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))


def _r_environment_ready():
    """True only when Rscript plus the locked numerical packages are importable.

    These tests execute the real R preflight/design/DESeq2/limma code, so without
    the locked library they are skipped rather than allowed to fail as if the
    Python package were broken.
    """
    rscript = shutil.which("Rscript")
    if not rscript:
        return False
    env = dict(os.environ)
    if R_LIBRARY.is_dir():
        env["R_LIBS_USER"] = str(R_LIBRARY)
    try:
        response = subprocess.run(
            [rscript, "--vanilla", "-e",
             'cat(all(sapply(c("jsonlite","limma","DESeq2"),requireNamespace,quietly=TRUE)))'],
            env=env, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "TRUE" in response.stdout


R_READY = _r_environment_ready()

# Tests that run real R design validation or model fitting (see README "R统计环境").
NEEDS_R = {
    ("test_flow.py", "test_confirm_is_idempotent_start_failure_retries_same_run"),
    ("test_flow.py", "test_confirm_reprobes_input_drift_and_needs_new_confirmation"),
    ("test_flow.py", "test_confirmation_double_click_concurrently_creates_one_run"),
    ("test_flow.py", "test_environment_drift_requires_new_plan"),
    ("test_flow.py", "test_full_conversation_merges_partial_fields_and_resolves_file_refs"),
    ("test_flow.py", "test_partial_or_goal_revision_revokes_approval_immediately"),
    ("test_flow.py", "test_ready_plan_deduplicates_and_no_code_before_approval"),
    ("test_flow.py", "test_runner_is_the_executed_entry_and_tamper_blocks_resume"),
    ("test_loop.py", "test_confirm_run_phrase_no_longer_approves_or_runs"),
    ("test_loop.py", "test_conversation_probes_and_creates_a_plan_without_executing"),
    ("test_loop.py", "test_natural_clarification_plan_approval_and_model_loop"),
    ("test_loop.py", "test_plan_and_run_history_commands"),
    ("test_low_input.py", "test_explicit_rds_scale_attribute_does_not_need_a_question"),
    ("test_low_input.py", "test_plain_differential_real_r_and_volcano_without_candidates"),
    ("test_low_input.py", "test_scale_confirmation_makes_plain_plan_and_legacy_hash_stays_stable"),
    ("test_low_input.py", "test_swap_changes_only_direction_and_requires_new_confirmation"),
    ("test_low_input.py", "test_unknown_scale_or_input_drift_never_approves"),
    ("test_ui.py", "test_banner_does_not_replace_task_reply_or_active_work"),
    ("test_ui.py", "test_compact_scale_card_updates_plan_without_a_questionnaire"),
    ("test_ui.py", "test_controls_hidden_without_ready_plan"),
    ("test_ui.py", "test_current_empty_and_corrupt_are_read_only"),
    ("test_ui.py", "test_every_launch_and_poll_stays_home"),
    ("test_ui.py", "test_explicit_queries_and_errors_survive_polling"),
    ("test_ui.py", "test_hidden_history_cannot_replace_current_plan"),
    ("test_ui.py", "test_history_queries_do_not_select_an_old_task"),
    ("test_ui.py", "test_home_does_not_interrupt_a_live_background_process"),
    ("test_ui.py", "test_navigation_queries_preserve_origin_records_and_approval"),
    ("test_ui.py", "test_one_confirmation_starts_exact_displayed_plan"),
    ("test_ui.py", "test_queued_confirmation_rejects_replaced_plan"),
    ("test_ui.py", "test_retry_and_corrupt_state_do_not_silently_approve"),
    ("test_ui.py", "test_stop_uses_current_run_once_without_another_confirmation"),
    ("test_ui.py", "test_study_editor_creates_new_version"),
    ("test_ui.py", "test_study_load_automatically_probes_and_presents_plan"),
    ("test_ui.py", "test_task_lifecycle_warnings_and_restart"),
    ("test_ui.py", "test_workspace_actions_present_at_small_terminal"),
    ("test_workspace.py", "test_preflight_confirmation_and_audit_paths"),
}


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_r: needs the locked R library (jsonlite, limma, DESeq2)")


def pytest_collection_modifyitems(config, items):
    if R_READY:
        return
    skip = pytest.mark.skip(
        reason="requires the locked R environment; run 'spliceosome-agent deps --install-r --library ./r-library'")
    for item in items:
        if (Path(str(item.fspath)).name, item.originalname) in NEEDS_R:
            item.add_marker(skip)


@pytest.fixture
def study(tmp_path):
    path = create_example(tmp_path / "fixture", r_library=R_LIBRARY if R_LIBRARY.is_dir() else None)
    return load_study(path)


@pytest.fixture
def approved(tmp_path, study):
    store = Store(tmp_path / "workspace")
    plan = make_plan(study, "SYNTHETIC SOFTWARE TEST")
    digest = store.save_plan(plan)
    store.approve(digest)
    return store, plan, digest
