from io import BytesIO
import hashlib
import re

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="DefenseAI | Document Search",
    layout="wide",
)

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PAGES = 300
CHUNK_WORDS = 220
OVERLAP_WORDS = 40


@st.cache_data(show_spinner=False)
def extract_chunks(pdf_bytes, source_name):
    reader = PdfReader(BytesIO(pdf_bytes))

    if reader.is_encrypted:
        raise ValueError(
            "Encrypted PDFs are not supported. Use an unencrypted copy."
        )

    if len(reader.pages) > MAX_PAGES:
        raise ValueError(
            f"This prototype supports at most {MAX_PAGES} pages."
        )

    document_hash = hashlib.sha256(pdf_bytes).hexdigest()

    chunks = []
    empty_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            empty_pages.append(page_number)
            continue

        words = text.split()
        step = CHUNK_WORDS - OVERLAP_WORDS

        for start in range(0, len(words), step):
            passage = " ".join(words[start:start + CHUNK_WORDS])

            chunks.append({
                "chunk_id": (
                    f"{document_hash[:12]}-p{page_number}-w{start}"
                ),
                "source": source_name,
                "pdf_page": page_number,
                "text": passage,
            })

            if start + CHUNK_WORDS >= len(words):
                break

    if not chunks:
        raise ValueError(
            "No extractable text found. The PDF may require OCR."
        )

    return {
        "chunks": pd.DataFrame(chunks),
        "page_count": len(reader.pages),
        "empty_pages": empty_pages,
        "document_sha256": document_hash,
    }


@st.cache_resource(show_spinner=False)
def build_index(passages):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(passages)

    return vectorizer, matrix


st.title("DefenseAI — Document Search")
st.caption("PDF ingestion and source-linked passage retrieval.")

st.warning(
    "Research document search only. Retrieved passages are not "
    "verified maintenance instructions or proof of a model diagnosis. "
    "Upload only public or otherwise authorized, non-sensitive documents."
)

st.info(
    "This version uses keyword-based TF-IDF retrieval. "
    "It does not generate answers, use an LLM, or search the internet."
)

uploaded = st.file_uploader(
    "Upload a text-based PDF",
    type=["pdf"],
)

if uploaded is None:
    st.write(
        "Start with a public research paper, dataset description, "
        "or educational engine-maintenance document."
    )
    st.stop()

if uploaded.size > MAX_FILE_BYTES:
    st.error("Please upload a PDF smaller than 20 MB.")
    st.stop()

try:
    with st.spinner("Extracting PDF text and building the search index..."):
        document = extract_chunks(
            uploaded.getvalue(),
            uploaded.name,
        )

        chunks = document["chunks"]

        vectorizer, matrix = build_index(
            tuple(chunks["text"].tolist())
        )

except Exception as exc:
    st.error(f"Could not index this document: {exc}")
    st.stop()

a, b, c = st.columns(3)
a.metric("PDF pages", document["page_count"])
b.metric("Searchable passages", len(chunks))
c.metric("Pages without extracted text", len(document["empty_pages"]))

if document["empty_pages"]:
    st.warning(
        "Some pages produced no text and were excluded. "
        "This search may therefore omit important content."
    )
    with st.expander("Pages without extracted text"):
        st.write(document["empty_pages"])

st.caption(
    "Page numbers refer to PDF viewer positions starting at 1. "
    "They may differ from the document's printed page labels."
)

with st.expander("Inspect extracted passages"):
    st.dataframe(
        chunks.head(20),
        hide_index=True,
        width="stretch",
    )

with st.form("document_search"):
    query = st.text_input(
        "Search question or keywords",
        placeholder="Example: remaining useful life prediction",
    )

    top_k = st.slider(
        "Maximum passages",
        min_value=1,
        max_value=10,
        value=5,
    )

    submitted = st.form_submit_button("Search document")

if submitted:
    query = query.strip()

    if not query:
        st.error("Enter a search question or keywords.")
        st.stop()

    query_vector = vectorizer.transform([query])

    if query_vector.nnz == 0:
        st.info(
            "No indexed terms match your query. "
            "Try terminology used in the document."
        )
        st.stop()

    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked_indices = np.argsort(-scores, kind="stable")
    selected_indices = [
        int(index)
        for index in ranked_indices
        if scores[index] > 0
    ][:top_k]

    if not selected_indices:
        st.info("No passages matched the query.")
        st.stop()

    results = chunks.iloc[selected_indices].copy()
    results.insert(
        0,
        "similarity_score",
        scores[selected_indices],
    )

    st.subheader("Retrieved passages")
    st.caption(
        "Scores indicate keyword-vector similarity, not confidence, "
        "factual correctness, or maintenance applicability. "
        "Even a nonzero match can be irrelevant."
    )

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        st.markdown(
            f"### Match {rank} — PDF page {int(row['pdf_page'])}"
        )
        st.caption(
            f"Source: {row['source']} | "
            f"Similarity: {row['similarity_score']:.3f} | "
            f"Chunk: {row['chunk_id']}"
        )

        # Render extracted document content as literal text.
        st.text(row["text"])
        st.divider()

    st.download_button(
        "Download retrieved passages",
        results.to_csv(index=False),
        file_name="defenseai_retrieved_passages.csv",
        mime="text/csv",
        on_click="ignore",
    )

st.caption(
    "The uploaded PDF is processed by this local application. "
    "This code does not send it to an external LLM or write it to disk. "
    "Extracted text and the search index are cached in application memory."
)