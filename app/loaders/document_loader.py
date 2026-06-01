"""
DOCX & PDF Document Loader
Loads and extracts text from DOCX and PDF files with heading/sub-heading awareness for RAG applications.
"""

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_docx(file_path: str) -> List[Document]:
    """
    Load a DOCX file and return a list of Document objects.
    Detects Heading 1 (main headings) and Heading 2/3 (sub-headings) 
    and attaches them as metadata AND prefixes them into chunk content
    so the LLM always knows which section a chunk belongs to.
    """
    try:
        from docx import Document as DocxDocument
    except ImportError:
        raise ImportError(
            "python-docx is required to load DOCX files. "
            "Install it with: pip install python-docx"
        )

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() != ".docx":
        raise ValueError(f"File must be a DOCX file, got: {file_path.suffix}")

    try:
        doc = DocxDocument(str(file_path))
    except Exception as e:
        raise ValueError(f"Failed to read DOCX file: {e}")

    documents = []
    current_heading = "General"
    current_subheading = ""

    for para in doc.paragraphs:
        style_name = para.style.name
        text = para.text.strip()

        if not text:
            continue

        # Detect Heading 1 → main section
        if style_name == "Heading 1":
            current_heading = text
            current_subheading = ""
            continue  # Don't add heading itself as a standalone chunk

        # Detect Heading 2 or Heading 3 → sub-section
        elif style_name in ("Heading 2", "Heading 3"):
            current_subheading = text
            continue  # Don't add heading itself as a standalone chunk

        # Build section breadcrumb prefix so LLM sees structure
        if current_subheading:
            section_prefix = f"[Section: {current_heading} > {current_subheading}]\n"
        else:
            section_prefix = f"[Section: {current_heading}]\n"

        enriched_content = section_prefix + text

        doc_obj = Document(
            page_content=enriched_content,
            metadata={
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": "docx",
                "heading": current_heading,
                "subheading": current_subheading,
                "section": f"{current_heading} > {current_subheading}" if current_subheading else current_heading,
            }
        )
        documents.append(doc_obj)

    # Extract text from tables (with section context)
    for table_idx, table in enumerate(doc.tables):
        for row_idx, row in enumerate(table.rows):
            row_text = " | ".join(
                cell.text.strip() for cell in row.cells
            )
            if row_text.strip():
                doc_obj = Document(
                    page_content=f"[Section: {current_heading}]\n{row_text}",
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": "docx",
                        "heading": current_heading,
                        "subheading": current_subheading,
                        "table": table_idx,
                        "row": row_idx,
                    }
                )
                documents.append(doc_obj)

    return documents


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file with basic heading detection based on line patterns.
    Attaches heading metadata and prefixes section info into each chunk.
    """
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()

    enriched_docs = []
    current_heading = "General"
    current_subheading = ""

    for doc in raw_docs:
        lines = doc.page_content.split("\n")
        page_num = doc.metadata.get("page", 0)
        enriched_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Heuristic: short ALL-CAPS or title-case short lines are likely headings
            is_heading = (
                len(stripped) < 80 and
                (stripped.isupper() or stripped.istitle()) and
                not stripped.endswith(".")
            )

            if is_heading:
                # Guess level by length: shorter = more likely a main heading
                if len(stripped) <= 40:
                    current_heading = stripped
                    current_subheading = ""
                else:
                    current_subheading = stripped
                # Don't skip — still add the heading line as part of context
                enriched_lines.append(f"[Heading: {stripped}]")
            else:
                enriched_lines.append(stripped)

        if enriched_lines:
            if current_subheading:
                section_prefix = f"[Section: {current_heading} > {current_subheading}]\n"
            else:
                section_prefix = f"[Section: {current_heading}]\n"

            enriched_content = section_prefix + "\n".join(enriched_lines)

            enriched_docs.append(Document(
                page_content=enriched_content,
                metadata={
                    "source": doc.metadata.get("source", file_path),
                    "file_name": Path(file_path).name,
                    "file_type": "pdf",
                    "page": page_num,
                    "heading": current_heading,
                    "subheading": current_subheading,
                    "section": f"{current_heading} > {current_subheading}" if current_subheading else current_heading,
                }
            ))

    return enriched_docs


def load_and_split(
    file_path: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Load a PDF or DOCX file, enrich with heading/sub-heading context,
    and split into chunks for RAG retrieval.
    """
    file_path_obj = Path(file_path)
    extension = file_path_obj.suffix.lower()

    if extension == ".pdf":
        docs = load_pdf(file_path)
    elif extension == ".docx":
        docs = load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(docs)

    # Ensure section metadata is preserved on every split chunk
    for chunk in chunks:
        if "heading" not in chunk.metadata:
            chunk.metadata["heading"] = "General"
        if "subheading" not in chunk.metadata:
            chunk.metadata["subheading"] = ""

    return chunks
