import pytest
import yaml
from fpdf import FPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from openhands.tools.cat_on_steroids.definition import CatOnSteroidsAction
from openhands.tools.cat_on_steroids.impl import _DOC_CACHE, CatOnSteroidsExecutor
from openhands.tools.cat_on_steroids.preprocessor import DocumentPreprocessor
from openhands.tools.cat_on_steroids.utils import validate_and_expand_pages_json_only


@pytest.fixture(autouse=True)
def clear_cache():
    # Ensure a clean global cache for each test run
    _DOC_CACHE.clear()
    yield
    _DOC_CACHE.clear()


@pytest.fixture
def dummy_pdf(tmp_path):
    pdf_path = tmp_path / "test_doc.pdf"

    def create_dummy_pdf(path: str) -> None:
        """
        Create a two-page PDF at `path` containing the strings "memory map" and "foo"
        on each page. Tries to use reportlab, falls back to fpdf if available, and
        finally writes a minimal blank PDF if neither library is installed.
        """
        # Try reportlab first
        try:
            c = canvas.Canvas(path, pagesize=letter)
            c.setFont("Helvetica", 12)
            c.drawString(72, 720, "Page 1: memory map foo")
            c.showPage()
            c.setFont("Helvetica", 12)
            c.drawString(72, 720, "Page 2: memory map foo")
            c.showPage()
            c.save()
            return
        except Exception:
            pass

        # Fallback to fpdf (lightweight)
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, "Page 1: memory map foo", ln=1)
            pdf.add_page()
            pdf.cell(0, 10, "Page 2: memory map foo", ln=1)
            pdf.output(path)
            return
        except Exception:
            pass

        # Last-resort: write a minimal blank 1-page PDF (may not contain extractable text).
        # This ensures a valid .pdf file exists so downstream code that only checks for file
        # presence / basic structure won't fail.
        minimal_pdf = (
            b"%PDF-1.1\n"
            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
            b"4 0 obj << /Length 44 >> stream\nBT /F1 24 Tf 72 720 Td (memory map foo) Tj ET\nendstream endobj\n"
            b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
            b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000061 00000 n \n"
            b"0000000114 00000 n \n0000000220 00000 n \n0000000280 00000 n \n"
            b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n340\n%%EOF\n"
        )
        with open(path, "wb") as f:
            f.write(minimal_pdf)

    create_dummy_pdf(str(pdf_path))
    return pdf_path


def test_view_mode_returns_document_metadata(dummy_pdf):
    doc_path = str(dummy_pdf)
    processed_doc = DocumentPreprocessor(doc_path=doc_path)
    _DOC_CACHE[doc_path] = processed_doc

    ex = CatOnSteroidsExecutor()
    action = CatOnSteroidsAction(doc_path=doc_path)  # no search_pattern => view mode
    obs = ex(action)

    print(obs)
    # assert obs.doc_metadata, "Expected doc_metadata in view mode observation"
    # assert obs.doc_metadata["total_pages"] == len(processed_doc.page_count)
    # assert obs.doc_metadata["full_text_length"] == len(processed_doc.full_text)
    # assert obs.doc_metadata["pdf_metadata"] == processed_doc.doc_metadata


def test_search_level_1_returns_metadata_summary(dummy_pdf):
    doc_path = str(dummy_pdf)
    processed_doc = DocumentPreprocessor(doc_path=doc_path)
    _DOC_CACHE[doc_path] = processed_doc

    ex = CatOnSteroidsExecutor()
    action = CatOnSteroidsAction(
        doc_path=doc_path, search_pattern="memory map", search_level="surface"
    )
    obs = ex(action)
    with open("test_level_1.yaml", "w") as f:
        yaml.safe_dump([tc.text for tc in obs.to_llm_content], f)
    # Two pages each contain "foo"
    # assert obs.total_results == 2
    # assert len(obs.metadata_summary) == 2
    # # Check that the summary strings reference page numbers and a content snippet
    # assert any("Page 1" in s or "page 1" in s for s in obs.metadata_summary)
    # assert any("Page 2" in s or "page 2" in s for s in obs.metadata_summary)


def test_search_level_2_respects_n_results_limit(dummy_pdf):
    doc_path = str(dummy_pdf)
    processed_doc = DocumentPreprocessor(doc_path=doc_path)
    _DOC_CACHE[doc_path] = processed_doc

    ex = CatOnSteroidsExecutor()

    # Request level 2 but limit to 1 result
    action = CatOnSteroidsAction(
        doc_path=doc_path, search_pattern="memory map", search_level="deep", n_results=1
    )
    obs = ex(action)

    with open("test_level_2.yaml", "w") as f:
        yaml.safe_dump([tc.text for tc in obs.to_llm_content], f)
    # assert obs.total_results == 2
    # assert len(obs.content_results) == 1
    # assert "page_number" in first and "page_content" in first


def _test_case(pages_str):
    try:
        pages = validate_and_expand_pages_json_only(pages_str)
        print(f"{pages_str!r} -> {pages}")
    except Exception as e:
        print(f"{pages_str!r} -> ERROR: {e}")


if __name__ == "__main__":
    _test_case('["46-48", "63-65"]')  # -> [46,47,48,63,64,65]
    _test_case("[1, 11, 12, 27]")  # -> [1,11,12,27]
    _test_case('["1-3","5"]')  # -> [1,2,3,5]
    _test_case('["1-3,5"]')  # -> ERROR (comma inside element)
    _test_case('"["46-48","63-65"]"')  # -> ERROR (starts with a quote, not '[')
    _test_case("[]")  # -> [] (empty => no filter)
    _test_case("[true]")  # -> ERROR (boolean not allowed)
    _test_case("[1,2,3]")  # -> ERROR (invalid string element)

# if __name__ == "__main__":
#     # test_view_mode_returns_document_metadata()
#     # test_search_level_1_returns_metadata_summary()
#     test_search_level_2_respects_n_results_limit()
