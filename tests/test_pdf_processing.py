from src.pdf_processing import chunk_pages


def test_chunk_pages_preserves_metadata():
    pages = [
        {
            "page": 1,
            "text": " ".join(
                f"word{i}" for i in range(200)
            ),
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=80,
        overlap=20,
    )

    assert len(chunks) > 1
    assert chunks[0]["page"] == 1
    assert chunks[0]["chunk_id"] == 0
    assert chunks[1]["chunk_id"] == 1


def test_chunk_overlap_must_be_smaller_than_size():
    pages = [
        {
            "page": 1,
            "text": "example text",
        }
    ]

    try:
        chunk_pages(
            pages,
            chunk_size=20,
            overlap=20,
        )
        assert False

    except ValueError:
        assert True
