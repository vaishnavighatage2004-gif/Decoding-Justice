import io
import re
from typing import List

from PyPDF2 import PdfReader

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False


SECTION_PATTERN = re.compile(
    r'(?:Section\s+\d+[A-Za-z()]*|IPC\s+\d+|CrPC\s+\d+|Article\s+\d+)',
    re.IGNORECASE
)


def extract_text_from_pdf_bytes(
    pdf_bytes,
    allow_ocr=False
):

    text_chunks = []

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))

        for p in reader.pages:

            t = p.extract_text()

            if t:
                text_chunks.append(t)

    except:
        pass

    full_text = "\n".join(text_chunks)

    if full_text.strip():
        return full_text

    if allow_ocr and OCR_AVAILABLE:

        images = convert_from_bytes(pdf_bytes)

        ocr_parts = [
            pytesseract.image_to_string(img)
            for img in images
        ]

        return "\n".join(ocr_parts)

    return ""


def extract_sections(text: str) -> List[str]:

    found = SECTION_PATTERN.findall(text)

    return list(dict.fromkeys(found))
FIR_PATTERN = re.compile(
    r'FIR\s*No\.?\s*[:\-]?\s*[\w\/\-]+',
    re.IGNORECASE
)

PS_PATTERN = re.compile(
    r'Police Station\s*[:\-]?\s*[A-Za-z ]+',
    re.IGNORECASE
)

DATE_PATTERN = re.compile(
    r'(\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4})',
    re.IGNORECASE
)


def extract_basic_metadata(text):

    firs = [
        m.group(0)
        for m in FIR_PATTERN.finditer(text)
    ]

    ps = [
        m.group(0)
        for m in PS_PATTERN.finditer(text)
    ]

    dates = [
        m.group(0)
        for m in DATE_PATTERN.finditer(text)
    ]

    return {
        "firs": firs,
        "police_stations": ps,
        "dates": dates
    }