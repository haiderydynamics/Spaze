"""Workspace Pages for SPAZE."""
from ui.pages.overview_page import render_overview_page
from ui.pages.floorplan_page import render_floorplan_page
from ui.pages.generation_page import render_generation_page
from ui.pages.visualization_page import render_visualization_page
from ui.pages.evaluation_page import render_evaluation_page
from ui.pages.export_page import render_export_page
from ui.pages.docs_page import render_docs_page

__all__ = [
    "render_overview_page",
    "render_floorplan_page",
    "render_generation_page",
    "render_visualization_page",
    "render_evaluation_page",
    "render_export_page",
    "render_docs_page",
]
