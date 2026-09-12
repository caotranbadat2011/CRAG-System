import math

from crag_ingestion.embeddings.hashing import HashingEmbedder


def test_hash_embeddings_are_deterministic_normalized_and_distinct() -> None:
    embedder = HashingEmbedder(128)
    vectors = embedder.embed(["truy xuất tài liệu", "truy xuất tài liệu", "quả chuối vàng"])
    assert vectors[0] == vectors[1]
    assert vectors[0] != vectors[2]
    assert all(abs(math.sqrt(sum(value * value for value in vector)) - 1) < 1e-9 for vector in vectors)
    assert all(len(vector) == 128 for vector in vectors)


def test_empty_text_returns_zero_vector() -> None:
    assert not any(HashingEmbedder(64).embed(["..."])[0])

