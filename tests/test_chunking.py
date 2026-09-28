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


def test_overlap_keeps_previous_page_and_exact_block_ranges() -> None:
    document = ParsedDocument(Path("pages.pdf"), "application/pdf", [
        TextBlock("Page one content. " * 6, page=1, heading_path=("First",), ordinal=0,
                  metadata={"source": {"page": 1}}),
        TextBlock("Page two content. " * 4, page=2, heading_path=("Second",), ordinal=1,
                  metadata={"source": {"page": 2}}),
    ])
    chunks = StructuralChunker(ChunkingConfig(120, 30, 0)).chunk("doc", document)
    assert chunks[1].pages == (1, 2)
    assert chunks[1].metadata["source_metadata"] == [{"source": {"page": 1}}, {"source": {"page": 2}}]
    assert chunks[1].metadata["source_spans"][0]["is_overlap"]
    for chunk in chunks:
        for span in chunk.metadata["source_spans"]:
            block = document.blocks[span["block_ordinal"]]
            assert chunk.text[span["chunk_char_start"]:span["chunk_char_end"]] == block.text[span["block_char_start"]:span["block_char_end"]]
            assert span["page"] == block.page
            assert span["heading_path"] == list(block.heading_path)


def test_split_block_offsets_survive_sentence_and_hard_splits() -> None:
    source = "  First sentence.  Second sentence.\n" + "verylongword" * 25 + " Tail. "
    doc = ParsedDocument(Path("x.txt"), "text/plain", [TextBlock(source)])
    chunks = StructuralChunker(ChunkingConfig(120, 15, 0)).chunk("doc", doc)
    for chunk in chunks:
        assert len(chunk.text) <= 120
        for span in chunk.metadata["source_spans"]:
            assert chunk.text[span["chunk_char_start"]:span["chunk_char_end"]] == source[span["block_char_start"]:span["block_char_end"]]
