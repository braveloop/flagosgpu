#!/usr/bin/env python3
"""
Convert Hugo markdown files (zh-cn) to MyST markdown for Sphinx.
Enhanced version for Chinese content conversion.

Usage: python convert_zhcn_to_myst.py
"""
import os
import re
from pathlib import Path


def remove_html_comments(content: str) -> str:
    """Remove HTML comments containing English text (common in zh-cn files)."""
    # Pattern to match HTML comments
    # <!-- ... -->
    pattern = r'<!--[\s\S]*?-->\s*'
    content = re.sub(pattern, '', content)
    return content


def convert_front_matter(content: str) -> str:
    """Remove Hugo front matter, keep title as heading."""
    # Pattern to match Hugo front matter
    pattern = r'^---\n.*?---\n+'

    # Extract title from front matter
    title_match = re.search(r'^title:\s*(.+)$', content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else None

    # Remove front matter
    content = re.sub(pattern, '', content, flags=re.DOTALL)

    # Check if content already starts with a heading
    first_line = content.strip().split('\n')[0] if content.strip() else ''
    has_heading = first_line.startswith('#')

    # If there's a title and no existing heading, add it
    if title and not has_heading:
        content = f'# {title}\n\n{content}'
    elif title and has_heading:
        # Title exists and heading exists - don't add duplicate
        pass

    return content


def convert_links(content: str) -> str:
    """Convert Hugo links to MyST relative links."""
    # Convert /FlagGems/zh-cn/path/ to relative path
    # Pattern: [text](/FlagGems/zh-cn/section/page/)
    content = re.sub(
        r'\[([^\]]+)\]\(/FlagGems/zh-cn/([^)]+)/?\)',
        lambda m: f'[{m.group(1)}]({convert_zhcn_path(m.group(2))})',
        content
    )

    # Fix broken links that might contain zh-cn prefix incorrectly
    # Pattern like ../zh-cnreferences/ -> ../references/
    content = re.sub(
        r'(\.\./|\./)zh-cn([a-zA-Z/]+)',
        r'\1\2',
        content
    )

    # Convert /FlagGems/path/ (without zh-cn) - for cross-language links if any
    content = re.sub(
        r'\[([^\]]+)\]\(/FlagGems/([^)]+)/?\)',
        lambda m: f'[{m.group(1)}]({convert_path(m.group(2))})',
        content
    )

    # Convert image links - both /FlagGems/images/ and ../images/
    content = re.sub(
        r'\[([^\]]*)\]\(/FlagGems/images/([^)]+)\)',
        r'[\1](_static/images/\2)',
        content
    )

    # Fix relative image paths like ../images/xxx.png -> _static/images/xxx.png
    content = re.sub(
        r'\[([^\]]*)\]\(\.\./images/([^)]+)\)',
        r'[\1](_static/images/\2)',
        content
    )

    # Handle direct image references without markdown syntax
    # <img src="/FlagGems/images/..." />
    content = re.sub(
        r'<img[^>]*src="/FlagGems/images/([^"]+)"[^>]*/?>',
        r'![](_static/images/\1)',
        content
    )

    # Handle GitHub user-attachments images
    content = re.sub(
        r'https://github\.com/user-attachments/assets/([^)\s]+)',
        r'https://github.com/user-attachments/assets/\1',
        content
    )

    return content


def convert_zhcn_path(hugo_path: str) -> str:
    """Convert Hugo zh-cn path to relative path for source-zh."""
    parts = hugo_path.strip('/').split('/')
    # zh-cn paths: /FlagGems/zh-cn/section/page/
    # Result should be relative: section/page or ../section/page
    if len(parts) >= 2:
        # Remove trailing duplicate if exists (e.g., overview/overview)
        if len(parts) >= 2 and parts[-1] == parts[-2]:
            parts = parts[:-1]
        return '../' + '/'.join(parts[-2:]) if len(parts) >= 2 else './' + parts[-1]
    return './' + parts[-1] if parts else './'


def convert_path(hugo_path: str) -> str:
    """Convert Hugo path to relative path."""
    parts = hugo_path.strip('/').split('/')
    if len(parts) >= 2:
        if parts[-1] == parts[-2]:
            parts = parts[:-1]
        return '../' + '/'.join(parts[-2:]) if len(parts) >= 2 else './' + parts[-1]
    return './' + parts[-1] if parts else './'


def convert_shortcodes(content: str) -> str:
    """Convert Hugo shortcodes to MyST directives."""
    # operator-list
    content = re.sub(
        r'\{\{<\s*operator-list\s*>\}\}',
        '\n```{operator-list}\n```\n',
        content
    )

    # operator-list with options
    content = re.sub(
        r'\{\{<\s*operator-list\s+([^>]+)\s*>\}\}',
        lambda m: f'\n```{{operator-list}}\n:{m.group(1).strip()}\n```\n',
        content
    )

    # benchmark-table
    content = re.sub(
        r'\{\{<\s*benchmark-table\s*>\}\}',
        '\n```{benchmark-table}\n```\n',
        content
    )

    # coverage-data
    content = re.sub(
        r'\{\{<\s*coverage-data\s*>\}\}',
        '\n```{coverage-data}\n```\n',
        content
    )

    # {{% steps %}} blocks - convert to numbered list
    content = re.sub(
        r'\{\{%\s*steps\s*%\}\}',
        '',
        content
    )

    return content


def convert_callouts(content: str) -> str:
    """Convert Hugo callouts to MyST admonitions."""
    # {{% callout note %}} -> ```{note}
    content = re.sub(
        r'\{\{%\s*callout\s+(\w+)\s*%\}\}',
        lambda m: f'\n```{{{m.group(1)}}}\n',
        content
    )
    content = re.sub(
        r'\{\{%\s*/callout\s*%\}\}',
        '\n```\n',
        content
    )

    return content


def convert_file(source_path: Path, dest_path: Path) -> None:
    """Convert a single Hugo markdown file to MyST."""
    print(f"Converting: {source_path} -> {dest_path}")

    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply conversions
    content = convert_front_matter(content)
    content = remove_html_comments(content)
    content = convert_links(content)
    content = convert_shortcodes(content)
    content = convert_callouts(content)

    # Clean up multiple blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Ensure file ends with single newline
    content = content.rstrip() + '\n'

    # Write to destination
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    """Main conversion function."""
    # Source: zh-cn content
    source_dir = Path(r'E:\BAAI\project\flaggems\gh-pages\FlagGems\docs\content\zh-cn')

    # Destination: docs-sphinx_zh/source
    dest_dir = Path(r'E:\BAAI\github\FlagGems\docs-sphinx_zh\source')

    # Find all markdown files
    md_files = list(source_dir.rglob('*.md'))

    print(f"Found {len(md_files)} markdown files to convert")

    for source_path in md_files:
        # Get relative path from source_dir
        rel_path = source_path.relative_to(source_dir)

        # Convert _index.md to index.md
        if rel_path.name == '_index.md':
            rel_path = rel_path.parent / 'index.md'

        # Destination path
        dest_path = dest_dir / rel_path

        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert file
        convert_file(source_path, dest_path)

    print(f"\nConverted {len(md_files)} files")


if __name__ == '__main__':
    main()