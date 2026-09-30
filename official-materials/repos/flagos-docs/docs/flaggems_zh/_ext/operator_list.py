# docs-sphinx/_ext/operator_list.py
"""
Sphinx directive to generate operator list table from conf/operators.yaml

Usage in MyST Markdown:
    ```{operator-list}
    :stage: stable
    ```

    ```{operator-category-list}
    ```

Options:
    :stage: Filter by stage (stable, beta, alpha)
    :kind: Filter by kind (Math, NeuralNetwork, etc.)
"""
from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective
from sphinx.application import Sphinx
from sphinx.util import logging as sphinx_logging
import yaml
import json
import re
import html
import os
from pathlib import Path
from typing import List, Dict, Any

logger = sphinx_logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helper functions (used by both directives)
# ---------------------------------------------------------------------------

def _has_stage(op: Dict[str, Any], stage: str) -> bool:
    """Check if operator has the specified stage."""
    stages = op.get('stages', [])
    for s in stages:
        if stage.lower() in [k.lower() for k in s.keys()]:
            return True
    return False


def _get_latest_stage(op: Dict[str, Any]) -> str:
    """Get the latest stage of an operator."""
    stages = op.get('stages', [])
    if not stages:
        return '-'
    last = stages[-1]
    for key in last.keys():
        return key
    return '-'


def _get_since_version(op: Dict[str, Any]) -> str:
    """Get the version when the operator was introduced."""
    stages = op.get('stages', [])
    if not stages:
        return '-'
    last = stages[-1]
    for value in last.values():
        return str(value)
    return '-'


# ---------------------------------------------------------------------------
# OperatorListDirective — existing Tabulator interactive table
# ---------------------------------------------------------------------------

class OperatorListDirective(SphinxDirective):
    """Generate operator list table from YAML data."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0

    option_spec = {
        'stage': directives.unchanged,
        'kind': directives.unchanged,
    }

    def run(self) -> List[nodes.Node]:
        """Generate the table."""
        yaml_path = Path(self.config.operator_yaml_path)

        if not yaml_path.exists():
            warning = nodes.warning(
                '',
                nodes.paragraph(
                    '',
                    f"Warning: operators.yaml not found at {yaml_path}"
                )
            )
            return [warning]

        # Load YAML data
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            error = nodes.error(
                '',
                nodes.paragraph('', f"Error loading operators.yaml: {e}")
            )
            return [error]

        ops = data.get('ops', [])

        # Apply filters
        stage_filter = self.options.get('stage')
        kind_filter = self.options.get('kind')

        if stage_filter:
            ops = [op for op in ops if _has_stage(op, stage_filter)]
        if kind_filter:
            ops = [op for op in ops if kind_filter in op.get('kind', [])]

        if not ops:
            return [nodes.paragraph('', 'No operators found matching the criteria.')]

        # Prepare data for Tabulator
        table_data = []
        kind_values = set()
        stage_values = set()
        since_values = set()

        for idx, op in enumerate(ops, 1):
            stage = _get_latest_stage(op)
            since = _get_since_version(op)
            labels = op.get('labels', [])
            kind = op.get('kind', [])
            desc = op.get('description', '').strip()

            kind_str = ', '.join(kind)
            stage_str = stage.title() if stage != '-' else '-'

            kind_values.add(kind_str if kind_str else '-')
            stage_values.add(stage_str)
            since_values.add(since)

            table_data.append({
                'no': idx,
                'id': op.get('id', ''),
                'kind': kind_str if kind_str else '-',
                'stage': stage_str,
                'since': since,
                'labels': ', '.join(labels) if labels else '-',
                'description': desc,
            })

        # Build filter values dictionaries
        kind_filter_values = {"": "All"}
        for v in sorted(kind_values):
            kind_filter_values[v] = v

        stage_filter_values = {"": "All"}
        for v in sorted(stage_values):
            stage_filter_values[v] = v

        since_filter_values = {"": "All"}
        for v in sorted(since_values):
            since_filter_values[v] = v

        # Create container
        container = nodes.container()

        # Build HTML content
        html_parts = [
            '<button id="reset-filter-btn" style="margin-bottom: 10px; padding: 5px 15px; cursor: pointer;">Reset Filters</button>',
            '<div id="operator-table"></div>',
            '<script type="application/json" id="operator-data">' + json.dumps(table_data) + '</script>',
            '<script>',
            'document.addEventListener("DOMContentLoaded", function() {',
            '    var data = JSON.parse(document.getElementById("operator-data").textContent);',
            '    var table = new Tabulator("#operator-table", {',
            '        data: data,',
            '        layout: "fitColumns",',
            '        height: "70vh",',
            '        columns: [',
            '            {title: "No.", field: "no", sorter: "number", widthGrow: 1},',
            '            {title: "Name", field: "id", sorter: "string", headerFilter: "input", headerFilterPlaceholder: "Filter...", widthGrow: 2},',
            '            {title: "Kind", field: "kind", sorter: "string", headerFilter: "list", headerFilterParams: {values: ' + json.dumps(kind_filter_values) + '}, widthGrow: 2},',
            '            {title: "Stage", field: "stage", sorter: "string", headerFilter: "list", headerFilterParams: {values: ' + json.dumps(stage_filter_values) + '}, widthGrow: 1},',
            '            {title: "Since", field: "since", sorter: "string", headerFilter: "list", headerFilterParams: {values: ' + json.dumps(since_filter_values) + '}, widthGrow: 1},',
            '            {title: "Labels", field: "labels", sorter: "string", headerFilter: "input", headerFilterPlaceholder: "Filter...", widthGrow: 2},',
            '            {title: "Description", field: "description", sorter: "string", widthGrow: 4, tooltip: function(e, cell) {',
            '                var value = cell.getValue() || "";',
            '                var div = document.createElement("div");',
            '                div.style.whiteSpace = "pre-wrap";',
            '                div.style.maxWidth = "400px";',
            '                // Parse backticks as code',
            '                var parts = value.split("`");',
            '                for (var i = 0; i < parts.length; i++) {',
            '                    if (i % 2 === 0) {',
            '                        div.appendChild(document.createTextNode(parts[i]));',
            '                    } else {',
            '                        var code = document.createElement("code");',
            '                        code.style.backgroundColor = "#f0f0f0";',
            '                        code.style.padding = "1px 4px";',
            '                        code.style.borderRadius = "3px";',
            '                        code.style.fontFamily = "monospace";',
            '                        code.textContent = parts[i];',
            '                        div.appendChild(code);',
            '                    }',
            '                }',
            '                return div;',
            '            }, formatter: function(cell) {',
            '                var value = cell.getValue() || "";',
            '                var div = document.createElement("div");',
            '                div.style.whiteSpace = "nowrap";',
            '                div.style.overflow = "hidden";',
            '                div.style.textOverflow = "ellipsis";',
            '                // Parse backticks as code',
            '                var parts = value.split("`");',
            '                for (var i = 0; i < parts.length; i++) {',
            '                    if (i % 2 === 0) {',
            '                        div.appendChild(document.createTextNode(parts[i]));',
            '                    } else {',
            '                        var code = document.createElement("code");',
            '                        code.style.backgroundColor = "#f0f0f0";',
            '                        code.style.padding = "1px 4px";',
            '                        code.style.borderRadius = "3px";',
            '                        code.style.fontFamily = "monospace";',
            '                        code.textContent = parts[i];',
            '                        div.appendChild(code);',
            '                    }',
            '                }',
            '                return div;',
            '            }},',
            '        ],',
            '    });',
            '',
            '    // Reset button functionality',
            '    document.getElementById("reset-filter-btn").addEventListener("click", function() {',
            '        table.clearHeaderFilter();',
            '    });',
            '});',
            '</script>',
        ]

        html_node = nodes.raw('', '\n'.join(html_parts), format='html')
        container += html_node

        return [container]


# ---------------------------------------------------------------------------
# Helpers — source / test URL resolution
# ---------------------------------------------------------------------------

FLAGGEMS_REPO_BASE = "https://github.com/flagos-ai/FlagGems/blob/master"


def _format_desc_html(raw: str) -> str:
    """HTML-escape text and convert backtick-quoted phrases to <code> elements."""
    parts = raw.split('`')
    result_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            result_parts.append(html.escape(part))
        else:
            result_parts.append(f'<code>{html.escape(part)}</code>')
    return ''.join(result_parts)


def _get_base_name(op_id: str) -> str:
    """Strip '_' and '_out' suffixes to get the base operator name."""
    if op_id.endswith('_out'):
        return op_id[:-4]
    if op_id.endswith('_'):
        return op_id[:-1]
    return op_id


def _resolve_source_path(op: Dict[str, Any]) -> str:
    """Return a human-readable source path description for *op*."""
    op_id = op.get('id', '')
    labels = op.get('labels', [])
    base = _get_base_name(op_id)

    if 'KernelGen' in labels:
        return 'Auto-generated by KernelGen (no standalone source file)'

    if 'fused' in labels:
        return f'src/flag_gems/fused/{op_id}.py'

    return f'src/flag_gems/ops/{base}.py'


def _resolve_source_url(op: Dict[str, Any]) -> str:
    """Return the GitHub URL for the source, or an empty string."""
    op_id = op.get('id', '')
    labels = op.get('labels', [])
    base = _get_base_name(op_id)

    if 'KernelGen' in labels:
        return ''

    if 'fused' in labels:
        return f'{FLAGGEMS_REPO_BASE}/src/flag_gems/fused/{op_id}.py'

    return f'{FLAGGEMS_REPO_BASE}/src/flag_gems/ops/{base}.py'


def _resolve_test_url(op: Dict[str, Any], test_type: str) -> str:
    """Return the GitHub URL for *test_type* (accuracy / benchmark)."""
    op_id = op.get('id', '')
    if test_type == 'accuracy':
        return f'{FLAGGEMS_REPO_BASE}/tests/test_{op_id}.py'
    else:
        return f'{FLAGGEMS_REPO_BASE}/benchmark/test_{op_id}.py'


# ---------------------------------------------------------------------------
# Operator detail page generation
# ---------------------------------------------------------------------------

def render_operator_detail_myst(op: Dict[str, Any]) -> str:
    """Generate a MyST Markdown page for a single operator."""
    op_id = op.get('id', 'unknown')
    kind = ', '.join(op.get('kind', [])) or '-'
    stage = _get_latest_stage(op)
    since = _get_since_version(op)
    description = op.get('description', '').strip()
    aten_mapping = op.get('for', [])
    labels = op.get('labels', [])
    cpp_version = op.get('cpp', '')

    source_path = _resolve_source_path(op)
    source_url = _resolve_source_url(op)

    lines = []
    lines.append('---')
    lines.append('orphan: true')
    lines.append('---')
    lines.append('')
    lines.append(f'# {op_id}')
    lines.append('')

    # Meta info bar
    meta_parts = [
        f'**Kind:** {kind}',
        f'**Stage:** {stage}',
        f'**Since:** {since}',
    ]
    if cpp_version:
        meta_parts.append(f'**C++:** {cpp_version}')
    lines.append(' | '.join(meta_parts))
    lines.append('')

    # Description
    if description:
        lines.append('## Description')
        lines.append('')
        lines.append(description)
        lines.append('')

    # ATen Mapping
    if aten_mapping and aten_mapping != ['None']:
        lines.append('## ATen Mapping')
        lines.append('')
        for target in aten_mapping:
            lines.append(f'- `{target}`')
        lines.append('')

    # Labels
    if labels:
        lines.append('## Labels')
        lines.append('')
        lines.append(', '.join(f'`{l}`' for l in labels))
        lines.append('')

    # Source Code
    lines.append('## Source Code')
    lines.append('')
    if source_url:
        lines.append(f'- [{source_path}]({source_url})')
    else:
        lines.append(f'- {source_path}')
    lines.append('')

    # Tests
    lines.append('## Tests')
    lines.append('')
    lines.append(f'- **Accuracy:** [{FLAGGEMS_REPO_BASE}/tests/test_{op_id}.py]({_resolve_test_url(op, "accuracy")})')
    lines.append(f'- **Performance:** [{FLAGGEMS_REPO_BASE}/benchmark/test_{op_id}.py]({_resolve_test_url(op, "benchmark")})')
    lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# config-inited handler — create virtual .md files before source discovery
# ---------------------------------------------------------------------------

def generate_operator_detail_pages(app: Sphinx, config) -> None:
    """Create per-operator detail .md files on disk before Sphinx scans sources."""
    yaml_path = os.path.join(app.confdir, config.operator_yaml_path)
    if not os.path.exists(yaml_path):
        logger.warning(f'operator_list: {yaml_path} not found, skipping detail pages')
        return

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Compute project directory from this extension file's location
    _ext_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(_ext_dir)
    generated_dir = os.path.join(project_dir, 'references', 'generated')
    os.makedirs(generated_dir, exist_ok=True)

    count = 0
    for op in data.get('ops', []):
        op_id = op.get('id', '')
        if not op_id:
            continue
        content = render_operator_detail_myst(op)
        filepath = os.path.join(generated_dir, f'{op_id}.md')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        count += 1

    logger.info(f'operator_list: generated {count} operator detail pages in {generated_dir}')


# ---------------------------------------------------------------------------
# OperatorCategoryListDirective — per-kind grouping table
# ---------------------------------------------------------------------------

# Desired display order for common kinds; unknowns appear at the end.
_KIND_ORDER = [
    'Math', 'Reduction', 'BLAS', 'LinearAlg',
    'NeuralNetwork', 'Activation', 'Normalization',
    'Attention', 'MoE',
    'Tensor', 'Distribution',
    'Convolution', 'Quantization',
    'Science', 'Logic',
]


def _kind_sort_key(kind_name: str) -> int:
    try:
        return _KIND_ORDER.index(kind_name)
    except ValueError:
        return len(_KIND_ORDER)


class OperatorCategoryListDirective(SphinxDirective):
    """Render operators grouped by kind as static HTML tables.

    Usage:

        ```{operator-category-list}
        ```
    """

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec: dict = {}

    def run(self) -> List[nodes.Node]:
        yaml_path = Path(self.config.operator_yaml_path)
        if not yaml_path.exists():
            msg = nodes.warning('', nodes.paragraph(
                '', f'Warning: operators.yaml not found at {yaml_path}'))
            return [msg]

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            msg = nodes.error('', nodes.paragraph(
                '', f'Error loading operators.yaml: {e}'))
            return [msg]

        ops = data.get('ops', [])

        # Group by kind (use primary kind — first in list)
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for op in ops:
            kinds = op.get('kind', [])
            primary = kinds[0] if kinds else 'Other'
            groups.setdefault(primary, []).append(op)

        # Sort groups
        sorted_kinds = sorted(groups.keys(), key=_kind_sort_key)

        result: List[nodes.Node] = []

        for kind in sorted_kinds:
            kind_ops = groups[kind]
            kind_ops.sort(key=lambda o: o.get('id', ''))

            heading_id = nodes.make_id(f'category-{kind}')

            # Build a proper docutils section so Sphinx includes it in the TOC
            section = nodes.section(ids=[heading_id])
            section += nodes.title('', f'{kind} ({len(kind_ops)} operators)')

            # Build table as raw HTML inside the section
            html_parts: List[str] = []
            html_parts.append('<div class="pst-scrollable-table-container">')
            html_parts.append('<table class="operator-category-table docutils">')
            html_parts.append('<thead><tr>'
                              '<th>Operator</th>'
                              '<th>Stage</th>'
                              '<th>Since</th>'
                              '<th>Description</th>'
                              '</tr></thead>')
            html_parts.append('<tbody>')

            for op in kind_ops:
                op_id = op.get('id', '')
                stage = _get_latest_stage(op)
                since = _get_since_version(op)
                desc = op.get('description', '').strip()

                desc_brief = desc.split('\n')[0] if desc else '-'
                if len(desc_brief) > 120:
                    desc_brief = desc_brief[:117] + '...'
                desc_html = _format_desc_html(desc_brief)

                html_parts.append('<tr>'
                                  f'<td><a href="generated/{op_id}.html"><code>{op_id}</code></a></td>'
                                  f'<td>{stage}</td>'
                                  f'<td>{since}</td>'
                                  f'<td>{desc_html}</td>'
                                  '</tr>')

            html_parts.append('</tbody></table></div>')

            section += nodes.raw('', '\n'.join(html_parts), format='html')
            result.append(section)

        return result


# ---------------------------------------------------------------------------
# Sphinx setup
# ---------------------------------------------------------------------------

def setup(app: Sphinx) -> Dict[str, Any]:
    """Register directives, config values, and event hooks."""
    app.add_config_value('operator_yaml_path', '../conf/operators.yaml', 'html')
    app.add_directive('operator-list', OperatorListDirective)
    app.add_directive('operator-category-list', OperatorCategoryListDirective)
    app.add_js_file('js/tabulator.min.js')

    # Generate per-operator detail pages at build time
    app.connect('config-inited', generate_operator_detail_pages)

    return {
        'version': '1.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
