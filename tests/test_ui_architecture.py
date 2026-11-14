"""Unit tests for ui.state and ui architecture."""
import pytest
from models.floorplan_schema import FloorPlan, Room
from ui.components import clean_html
from ui.state import STAGE_NAMES, StateManager


def test_state_manager_initialization():
    StateManager.initialize()
    assert StateManager.get_stage() == "01_floorplan"
    assert not StateManager.has_floorplan()
    assert not StateManager.has_alternatives()


def test_state_manager_stage_transitions():
    StateManager.initialize()
    for stage_id in STAGE_NAMES.keys():
        StateManager.set_stage(stage_id)
        assert StateManager.get_stage() == stage_id


def test_stage_status_logic():
    StateManager.initialize()
    # Initially 01 should be Ready, 02 Pending Input, 03 Pending Input, 04 Pending Data
    assert StateManager.get_stage_status("01_floorplan") == "Ready"
    assert StateManager.get_stage_status("02_generation") == "Pending Input"
    assert StateManager.get_stage_status("03_visualization") == "Pending Input"
    assert StateManager.get_stage_status("04_evaluation") == "Pending Data"


def test_clean_html_removes_markdown_code_block_triggers():
    """Verify clean_html removes all leading whitespace on all lines to prevent Markdown code block triggers."""
    raw_indented_html = """
        <div class="ref-metric-card">
            <div>
                <span>Title</span>
            </div>
        </div>
    """
    cleaned = clean_html(raw_indented_html)
    for line in cleaned.splitlines():
        assert not line.startswith(" "), f"Line starts with whitespace: {line!r}"
        assert not line.startswith("\t"), f"Line starts with tab: {line!r}"
    assert cleaned.startswith("<div")
    assert cleaned.endswith("</div>")

