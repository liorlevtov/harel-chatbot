"""Document conversion using docling."""

from pathlib import Path

from docling.document_converter import DocumentConverter


def convert_document(input_path: Path, output_path: Path) -> None:
    """
    Convert a document (PDF, HTML, ASPX) to markdown using docling.

    Args:
        input_path: Path to source file
        output_path: Path for output markdown file

    Raises:
        Exception: If conversion fails
    """
    converter = DocumentConverter()
    result = converter.convert(str(input_path))

    # Export to markdown
    markdown_content = result.document.export_to_markdown()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write markdown with UTF-8 encoding for Hebrew support
    output_path.write_text(markdown_content, encoding="utf-8")
