# docs-sphinx/_ext/benchmark_table.py
"""
Sphinx directive to generate benchmark results table from JSON files.
Uses Tabulator for interactive tables (same as Hugo version).

Usage in MyST Markdown:
    ```{benchmark-table}
    ```
"""
from docutils import nodes
from sphinx.util.docutils import SphinxDirective
from sphinx.application import Sphinx
import json
from pathlib import Path
from typing import List, Dict, Any


class BenchmarkTableDirective(SphinxDirective):
    """Generate benchmark results table using Tabulator."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0

    def run(self) -> List[nodes.Node]:
        """Generate the benchmark table with tabs."""
        benchmark_path = Path(self.config.benchmark_data_path)

        if not benchmark_path.exists():
            warning = nodes.warning(
                '',
                nodes.paragraph(
                    '',
                    f"Warning: benchmark directory not found at {benchmark_path}"
                )
            )
            return [warning]

        # Find all JSON files
        json_files = list(benchmark_path.glob("*.json"))

        if not json_files:
            return [nodes.paragraph('', 'No benchmark data available. Run benchmarks first.')]

        # Create container
        container = nodes.container()

        # Build HTML for tabs and tables (same structure as Hugo)
        html_parts = []

        # Tab buttons
        html_parts.append('<div class="benchmark-tabs" id="benchmark-tabs">')
        for json_file in sorted(json_files):
            platform = json_file.stem
            html_parts.append(f'<button data-platform="{platform}" onclick="switchTab(\'{platform}\')">{platform.upper()}</button>')
        html_parts.append('</div>')

        # Panels with data
        for json_file in sorted(json_files):
            platform = json_file.stem
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                continue

            compiler = data.get('compiler', '')
            timestamp = data.get('timestamp', '')
            operators = data.get('operators', [])

            html_parts.append(f'<div class="benchmark-panel" id="panel-{platform}">')
            html_parts.append(f'<p><strong>Compiler:</strong> {compiler} &nbsp; <strong>Timestamp:</strong> {timestamp}</p>')
            html_parts.append(f'<div id="table-{platform}"></div>')
            html_parts.append('</div>')
            html_parts.append(f'<script id="data-{platform}" type="application/json">')
            html_parts.append(json.dumps(operators))
            html_parts.append('</script>')

        html_node = nodes.raw('', '\n'.join(html_parts), format='html')
        container += html_node

        return [container]


def setup(app: Sphinx) -> Dict[str, Any]:
    """Register the directive with Sphinx."""
    app.add_config_value('benchmark_data_path', '../benchmark', 'html')
    app.add_directive('benchmark-table', BenchmarkTableDirective)
    # Add Tabulator JS
    app.add_js_file('js/tabulator.min.js')
    app.add_js_file('js/benchmark.js')

    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }