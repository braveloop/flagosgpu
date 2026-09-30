# docs-sphinx/_ext/coverage_data.py
"""
Sphinx directive to list coverage report links.

Usage in MyST Markdown:
    ```{coverage-data}
    ```
"""
from docutils import nodes
from sphinx.util.docutils import SphinxDirective
from sphinx.application import Sphinx
from pathlib import Path
from typing import List, Dict, Any


class CoverageDataDirective(SphinxDirective):
    """Generate list of coverage report links."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0

    def run(self) -> List[nodes.Node]:
        """Generate the coverage data list."""
        coverage_path = Path(self.config.coverage_data_path)

        # Create container
        container = nodes.container()

        if not coverage_path.exists():
            container += nodes.paragraph('', 'No coverage data available yet.')
            return [container]

        # Get all coverage directories (sorted by date, newest first)
        try:
            coverage_dirs = sorted(
                [d for d in coverage_path.iterdir() if d.is_dir()],
                key=lambda x: x.name,
                reverse=True
            )
        except Exception:
            container += nodes.paragraph('', 'Error reading coverage directory.')
            return [container]

        if not coverage_dirs:
            container += nodes.paragraph('', 'No coverage reports available.')
            return [container]

        # Create a bullet list
        bullet_list = nodes.bullet_list()

        for cov_dir in coverage_dirs:
            list_item = nodes.list_item()
            para = nodes.paragraph()

            # Create reference link to coverage report
            # The coverage page is at references/test/coverage/index.html
            # Static files are at _static/ at the docs root
            # So we need ../../../_static/ to get from coverage page to docs root
            ref = nodes.reference()
            ref['refuri'] = f'../../../_static/{cov_dir.name}/index.html'
            ref += nodes.Text(cov_dir.name)

            para += ref
            list_item += para
            bullet_list += list_item

        container += bullet_list

        # Add note about how coverage is generated
        note = nodes.admonition()
        note += nodes.title('', 'Note')
        note += nodes.paragraph(
            '',
            'Coverage reports are generated automatically by CI. '
            'Each directory contains HTML coverage reports for a specific test run.'
        )
        container += note

        return [container]


def setup(app: Sphinx) -> Dict[str, Any]:
    """Register the directive with Sphinx."""
    app.add_config_value('coverage_data_path', '_static/coverage', 'html')
    app.add_directive('coverage-data', CoverageDataDirective)

    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }