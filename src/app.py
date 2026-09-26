# app.py
# Purpose: The main user-facing interface, tying together search, upload,
# document versioning, analytics, and result downloads into a Streamlit app.

import streamlit as st
import os
import re
import pandas as pd
from qdrant_client import QdrantClient

from ingest import extract_text_from_pdf
from chunk import process_pages_into_chunks
from embed_store import (
    setup_collection, store_chunks, delete_existing_file_chunks, COLLECTION_NAME,
    get_file_hash, get_stored_file_hash,
    setup_search_log_collection, log_search_query, get_top_searches
)
from search import semantic_search, generate_summary

st.set_page_config(page_title="Policy Document Search", layout="wide")

client = QdrantClient(host="localhost", port=6333)
setup_search_log_collection()

st.title("📄 Company Policy Document Search")


def highlight_query_terms(text, query):
    """
    Wraps any word from the user's query that appears in the excerpt text
    with a highlighted <mark> tag, so matching terms visually stand out.
    """
    query_words = [w for w in query.split() if len(w) > 2]
    highlighted = text
    for word in query_words:
        pattern = re.compile(rf'\b({re.escape(word)})\b', re.IGNORECASE)
        highlighted = pattern.sub(r'<mark style="background-color: #ffe066; padding: 1px 2px;">\1</mark>', highlighted)
    return highlighted


def process_and_store(save_path, filename, file_hash, replace_existing=True):
    st.info(f"Processing {filename}... this may take a few minutes for scanned PDFs.")
    try:
        setup_collection()
        if replace_existing:
            delete_existing_file_chunks(filename)
        pages = extract_text_from_pdf(save_path)
        chunks = process_pages_into_chunks(pages, filename)
        store_chunks(chunks, file_hash=file_hash)
        st.success(f"{filename} processed! ({len(chunks)} chunks stored)")
    except Exception as e:
        st.error(f"Something went wrong while processing this file: {e}")


# --- Sidebar: Upload + Manage DB ---
with st.sidebar:
    st.header("Manage Documents")

    uploaded_file = st.file_uploader("Upload a new policy PDF", type=["pdf"])
    if uploaded_file is not None:
        file_bytes = bytes(uploaded_file.getbuffer())
        new_hash = get_file_hash(file_bytes)
        existing_hash = get_stored_file_hash(uploaded_file.name)

        if existing_hash == new_hash:
            st.warning(f"A copy of '{uploaded_file.name}' already exists in the database — no changes needed.")

        elif existing_hash is not None:
            st.warning(f"'{uploaded_file.name}' already exists but its content appears to have changed.")
            choice = st.radio(
                "How do you want to proceed?",
                ["Replace the existing version", "Keep both as separate copies"],
                key=f"choice_{uploaded_file.name}"
            )
            if st.button("Confirm", key=f"confirm_{uploaded_file.name}"):
                save_path = os.path.join("data/raw_pdfs", uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(file_bytes)

                if choice == "Replace the existing version":
                    process_and_store(save_path, uploaded_file.name, new_hash, replace_existing=True)
                else:
                    versioned_name = uploaded_file.name.replace(".pdf", "_v2.pdf")
                    versioned_path = os.path.join("data/raw_pdfs", versioned_name)
                    os.rename(save_path, versioned_path)
                    process_and_store(versioned_path, versioned_name, new_hash, replace_existing=False)

        else:
            save_path = os.path.join("data/raw_pdfs", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(file_bytes)
            process_and_store(save_path, uploaded_file.name, new_hash, replace_existing=False)

    st.divider()

    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        st.metric("Total chunks in database", collection_info.points_count)
    except Exception:
        st.metric("Total chunks in database", 0)

    st.divider()

    if st.button("🗑️ Clear entire database", type="secondary"):
        client.delete_collection(COLLECTION_NAME)
        setup_collection()
        st.success("Database cleared.")
        st.rerun()

    st.divider()
    st.subheader("Documents in database")

    try:
        all_points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=1000, with_payload=True)
        filenames = sorted(set(p.payload["filename"] for p in all_points))
    except Exception:
        filenames = []

    for fname in filenames:
        col_a, col_b = st.columns([4, 1])
        col_a.write(fname)
        if col_b.button("🗑️", key=f"delete_{fname}"):
            delete_existing_file_chunks(fname)
            st.success(f"Removed {fname}")
            st.rerun()

    st.divider()
    st.subheader("📊 Top Searches")

    top_searches = get_top_searches(5)
    if top_searches:
        # Build a small table Streamlit's bar_chart can plot directly:
        # query text as the index (row labels), search count as the value (bar length)
        chart_data = pd.DataFrame(top_searches, columns=["Query", "Searches"]).set_index("Query")
        st.bar_chart(chart_data, horizontal=True)
    else:
        st.caption("No searches yet.")
# --- Main area: Search ---

if "recent_searches" not in st.session_state:
    st.session_state.recent_searches = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "trigger_search" not in st.session_state:
    st.session_state.trigger_search = False

# Suggested/recent searches — combine recent + popular
suggestions = st.session_state.recent_searches[:3]
top_searches_list = [q for q, _ in get_top_searches(3)]
for s in top_searches_list:
    if s not in suggestions:
        suggestions.append(s)
suggestions = suggestions[:5]

# --- Handle chip clicks and Clear BEFORE creating the text_input widget ---
# (Streamlit doesn't allow changing a widget's value AFTER it's been drawn,
# so any changes to query_input must happen here, first.)
chip_clicked_value = None
if suggestions:
    st.caption("Recent / suggested searches:")
    chip_cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        with chip_cols[i]:
            if st.button(s, key=f"chip_{i}_{s}", use_container_width=True):
                chip_clicked_value = s

if chip_clicked_value is not None:
    st.session_state.query_input = chip_clicked_value
    st.session_state.trigger_search = True

# Search bar + buttons, all on one row
col_input, col_search, col_clear = st.columns([6, 1, 1])
with col_input:
    query = st.text_input(
        "Search company policies:",
        placeholder="e.g. job classification for scientific personnel",
        key="query_input",
        label_visibility="collapsed"
    )
with col_search:
    search_clicked = st.button("Search", use_container_width=True)
with col_clear:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state.query_input = ""
    st.rerun()

run_search = search_clicked or st.session_state.trigger_search
st.session_state.trigger_search = False

if run_search:
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        log_search_query(query)

        if query not in st.session_state.recent_searches:
            st.session_state.recent_searches.insert(0, query)
            st.session_state.recent_searches = st.session_state.recent_searches[:5]

        with st.spinner("Searching..."):
            matches = semantic_search(query, top_k=5)

        if not matches:
            st.warning("No relevant results found.")
        else:
            with st.spinner("Generating summary..."):
                summary = generate_summary(query, matches)

            with st.expander("Summary", expanded=True):
                st.write(summary)

            st.subheader("Matched Excerpts (with citations)")
            for i, m in enumerate(matches, 1):
                with st.expander(f"{i}. {m['filename']} — Page {m['page']} (relevance: {m['score']:.2f})"):
                    highlighted_text = highlight_query_terms(m['text'], query)
                    st.markdown(highlighted_text, unsafe_allow_html=True)

            download_text = f"Query: {query}\n\nSummary:\n{summary}\n\nMatched Excerpts:\n"
            for i, m in enumerate(matches, 1):
                download_text += f"\n{i}. {m['filename']} - Page {m['page']} (relevance: {m['score']:.2f})\n{m['text']}\n"

            st.download_button(
                label="📥 Download these results",
                data=download_text,
                file_name=f"search_results_{query[:30].replace(' ', '_')}.txt",
                mime="text/plain"
            )