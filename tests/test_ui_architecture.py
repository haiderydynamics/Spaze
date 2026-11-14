"""Unit tests for ui.state and ui architecture."""
import pytest
from models.floorplan_schema import FloorPlan, Room
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
