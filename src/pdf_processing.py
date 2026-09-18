from io import BytesIO

from pypdf import PdfReader


def extract_pdf_pages(pdf_bytes: bytes):
    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text:
            pages.append({
                "page": page_number,
                "text": text,
            })

    return pages


def chunk_pages(
    pages,
    chunk_size=120,
    overlap=25,
):
    chunks = []
    chunk_id = 0

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap

    for page in pages:
        words = page["text"].split()

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                continue

            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": chunk_id,
                "page": page["page"],
                "text": chunk_text,
            })

            chunk_id += 1

            if start + chunk_size >= len(words):
                break

    return chunks
