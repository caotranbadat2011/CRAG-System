from pathlib import Path

from crag_ingestion.chunking import StructuralChunker
from crag_ingestion.config import ChunkingConfig
from crag_ingestion.models import ParsedDocument, TextBlock


def test_chunking_honors_limit_overlap_and_provenance() -> None:
    text = " ".join(f"Câu {number} mô tả hệ thống truy xuất." for number in range(40))
    document = ParsedDocument(
        Path("guide.md"),
        "text/markdown",
        [TextBlock(text, page=2, heading_path=("Kiến trúc",), ordinal=0)],
    )
    chunks = StructuralChunker(ChunkingConfig(max_chars=240, overlap_chars=40, min_chars=40)).chunk(
        "doc-1", document
    )
    assert len(chunks) > 2
    assert all(0 < chunk.char_count <= 240 for chunk in chunks)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.pages == (2,) for chunk in chunks)
    assert all(chunk.heading_path == ("Kiến trúc",) for chunk in chunks)
    assert chunks[0].metadata["has_overlap"] is False
    assert all(chunk.metadata["has_overlap"] is True for chunk in chunks[1:])


def test_long_unbroken_text_is_split_without_data_loss() -> None:
    text = "x" * 550
    document = ParsedDocument(Path("x.txt"), "text/plain", [TextBlock(text)])
    chunks = StructuralChunker(ChunkingConfig(200, 0, 0)).chunk("doc", document)
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(len(chunk.text) <= 200 for chunk in chunks)

