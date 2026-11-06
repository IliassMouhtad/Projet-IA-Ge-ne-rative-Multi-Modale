import os
import re
import hashlib
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Tuple
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import SentenceSplitter
from tabulate import tabulate
import logging

logger = logging.getLogger(__name__)

def compute_sha256(file_path: str | Path) -> str:
    """
    Calculate SHA-256 hash of a file to manage deduplication and reindexing.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

class MultimodalPDFExtractor:
    """
    Extracts text, tables, and images from a PDF document
    and converts them into LlamaIndex Nodes with rich metadata.
    """

    def __init__(
        self,
        pdf_path: str | Path,
        output_image_dir: str | Path = "extracted_images",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.pdf_path = Path(pdf_path)
        self.output_image_dir = Path(output_image_dir)
        self.output_image_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.doc_hash = compute_sha256(self.pdf_path) if self.pdf_path.exists() else None
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def extract_images(self) -> List[Dict[str, Any]]:
        """
        Extract raster images and rendered diagrams from PDF pages.
        Saves images to output_image_dir and returns metadata for each image.
        """
        extracted = []
        doc = fitz.open(self.pdf_path)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            image_list = page.get_images(full=True)

            # 1. Extract embedded raster images
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_filename = f"page_{page_num}_img_{img_idx + 1}.{image_ext}"
                image_filepath = self.output_image_dir / image_filename

                with open(image_filepath, "wb") as f:
                    f.write(image_bytes)

                extracted.append({
                    "image_id": f"img_p{page_num}_{img_idx+1}",
                    "image_path": str(image_filepath.resolve()),
                    "filename": image_filename,
                    "page": page_num,
                    "type": "image",
                    "width": base_image.get("width"),
                    "height": base_image.get("height"),
                    "source": self.pdf_path.name,
                })

            # 2. In technical papers like "Attention Is All You Need", architectures
            # (Figure 1 on page 3, Figure 2 on page 4) are often vector drawings.
            # Detect figure text captions and render high-resolution crops or visual snapshots.
            text = page.get_text()
            figures_found = re.findall(r"(Figure\s+\d+:\s*([^\n\.]+))", text)
            for fig_full, fig_title in figures_found:
                fig_slug = re.sub(r"[^\w\-]", "_", fig_full[:25]).strip("_").lower()
                render_filename = f"page_{page_num}_{fig_slug}.png"
                render_filepath = self.output_image_dir / render_filename

                # Render page at high resolution (150 DPI) for visual inspection
                pix = page.get_pixmap(dpi=150)
                pix.save(render_filepath)

                extracted.append({
                    "image_id": f"fig_p{page_num}_{fig_slug}",
                    "image_path": str(render_filepath.resolve()),
                    "filename": render_filename,
                    "page": page_num,
                    "type": "figure_diagram",
                    "caption": fig_full.strip(),
                    "source": self.pdf_path.name,
                })

        doc.close()
        logger.info(f"Extracted {len(extracted)} image/figure items to {self.output_image_dir}")
        return extracted

    def extract_tables(self) -> List[Dict[str, Any]]:
        """
        Extract tables using pdfplumber, converting tabular data to Markdown format.
        """
        extracted_tables = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                tables = page.extract_tables()

                for table_idx, raw_table in enumerate(tables):
                    if not raw_table or len(raw_table) < 2:
                        continue

                    # Clean nulls
                    cleaned_rows = [
                        [str(cell).strip() if cell is not None else "" for cell in row]
                        for row in raw_table
                    ]

                    # Convert to Markdown table
                    headers = cleaned_rows[0]
                    data_rows = cleaned_rows[1:]
                    markdown_table = tabulate(data_rows, headers=headers, tablefmt="github")

                    table_text = (
                        f"### Table on Page {page_num} (Table {table_idx + 1})\n\n"
                        f"{markdown_table}\n"
                    )

                    extracted_tables.append({
                        "table_id": f"table_p{page_num}_{table_idx+1}",
                        "page": page_num,
                        "type": "table",
                        "markdown": markdown_table,
                        "content": table_text,
                        "rows_count": len(data_rows),
                        "cols_count": len(headers),
                        "source": self.pdf_path.name,
                    })

        logger.info(f"Extracted {len(extracted_tables)} tables from {self.pdf_path.name}")
        return extracted_tables

    def extract_text(self) -> List[Dict[str, Any]]:
        """
        Extract textual content page by page using PyMuPDF.
        """
        doc = fitz.open(self.pdf_path)
        pages_text = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            raw_text = page.get_text("text")

            # Basic clean-up
            lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
            cleaned_text = "\n".join(lines)

            if cleaned_text:
                pages_text.append({
                    "page": page_num,
                    "text": cleaned_text,
                    "source": self.pdf_path.name,
                })

        doc.close()
        logger.info(f"Extracted text from {len(pages_text)} pages.")
        return pages_text

    def extract_all_nodes(self) -> Tuple[List[TextNode], List[Dict[str, Any]]]:
        """
        Runs complete extraction and creates LlamaIndex TextNodes for text and tables,
        and returns image metadata items for downstream multimodal processing.
        """
        logger.info(f"Starting extraction for document: {self.pdf_path}")
        
        # 1. Images
        images_meta = self.extract_images()

        # 2. Tables
        tables_data = self.extract_tables()
        table_nodes = []
        for t in tables_data:
            node = TextNode(
                text=t["content"],
                metadata={
                    "doc_name": t["source"],
                    "doc_hash": self.doc_hash,
                    "page": t["page"],
                    "type": "table",
                    "table_id": t["table_id"],
                    "rows": t["rows_count"],
                    "cols": t["cols_count"],
                },
            )
            table_nodes.append(node)

        # 3. Text
        pages_text = self.extract_text()
        text_nodes = []
        for p in pages_text:
            chunks = self.splitter.split_text(p["text"])
            for chunk_idx, chunk in enumerate(chunks):
                node = TextNode(
                    text=chunk,
                    metadata={
                        "doc_name": p["source"],
                        "doc_hash": self.doc_hash,
                        "page": p["page"],
                        "type": "text",
                        "chunk_index": chunk_idx,
                    },
                )
                text_nodes.append(node)

        # 4. Image Reference Nodes (so that image references exist in the vector index)
        image_ref_nodes = []
        for img in images_meta:
            caption = img.get("caption", f"Image {img['filename']} from page {img['page']}")
            desc = (
                f"[Image Reference: {img['filename']}]\n"
                f"Page: {img['page']}\n"
                f"Caption/Description: {caption}\n"
                f"File: {img['image_path']}"
            )
            node = TextNode(
                text=desc,
                metadata={
                    "doc_name": img["source"],
                    "doc_hash": self.doc_hash,
                    "page": img["page"],
                    "type": "image",
                    "image_id": img["image_id"],
                    "image_path": img["image_path"],
                    "filename": img["filename"],
                    "caption": caption,
                },
            )
            image_ref_nodes.append(node)

        all_nodes = text_nodes + table_nodes + image_ref_nodes
        logger.info(
            f"Extraction complete: {len(text_nodes)} text chunks, "
            f"{len(table_nodes)} tables, {len(image_ref_nodes)} image reference nodes."
        )

        return all_nodes, images_meta
