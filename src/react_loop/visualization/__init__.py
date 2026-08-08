"""
Evolution commit graph visualization.

Generates a self-contained HTML file with interactive D3.js commit graph,
reward trend chart, and node detail panel.
"""

from .graph_builder import build_visualization_data
from .html_renderer import render_evolution_html


def generate_evolution_html(
    tracker,
    git_controller,
    goal: str = "",
) -> str:
    """Build and render the evolution graph as a self-contained HTML string.

    Args:
        tracker: EvolutionTracker instance.
        git_controller: GitController instance.
        goal: Evolution goal text.

    Returns:
        Complete HTML document string.
    """
    data = build_visualization_data(
        tracker=tracker,
        git_controller=git_controller,
        goal=goal,
    )
    return render_evolution_html(data)
