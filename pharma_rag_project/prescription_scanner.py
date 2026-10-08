from __future__ import annotations
import io, re, unicodedata
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError

try:
    import easyocr as _easyocr_mod
    # Probe: try registering to surface torchvision fake_impl errors at import
    # time so _EASYOCR_AVAILABLE is reliable.
    _EASYOCR_AVAILABLE = True
except Exception:
    _easyocr_mod = None
    _EASYOCR_AVAILABLE = False

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    _TESSERACT_AVAILABLE = False

_easyocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            _easyocr_reader = _easyocr_mod.Reader(['en'], gpu=False, verbose=False)
        except Exception as exc:
            raise RuntimeError(
                f"EasyOCR model initialisation failed: {exc}\n"
                "This is usually a torchvision version mismatch. "
                "Try: pip install --upgrade torchvision"
            ) from exc
    return _easyocr_reader


SUPPORTED_FORMATS = {'png', 'jpg', 'jpeg'}
CONFIDENCE_WARN_THRESHOLD = 60.0
MIN_MEDICINE_WORD_LEN = 3

_STOP_WORDS = {
    'the', 'and', 'for', 'use', 'uses', 'take', 'daily', 'twice', 'once',
    'tablet', 'tablets', 'capsule', 'capsules', 'syrup', 'injection',
    'mg', 'ml', 'mcg', 'dose', 'doses', 'dosage', 'morning', 'evening',
    'night', 'before', 'after', 'with', 'without', 'food', 'water',
    'weeks', 'days', 'month', 'months', 'years', 'year', 'week', 'day',
    'prescribed', 'prescription', 'doctor', 'patient', 'name', 'age',
    'date', 'sig', 'rx', 'ref', 'dr', 'hospital', 'clinic',
    'one', 'two', 'three', 'four', 'five', 'six', 'half', 'each',
    'this', 'that', 'when', 'needed', 'prn', 'bid', 'tid', 'qid',
    'as', 'at', 'by', 'in', 'of', 'on', 'or', 'to',
}


class ScanStatus(Enum):
    SUCCESS = auto()
    OCR_UNCERTAIN = auto()
    OCR_EMPTY = auto()
    INVALID_FILE = auto()
    BACKEND_UNAVAILABLE = auto()


@dataclass
class PrescriptionScanResult:
    status: ScanStatus
    extracted_text: str = ''
    medicine_candidates: List[str] = field(default_factory=list)
    mean_confidence: float = -1.0
    error_message: str = ''
    low_confidence_tokens: List[str] = field(default_factory=list)
    ocr_backend: str = 'unknown'


def _preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert('L')
    w, h = img.size
    if max(w, h) < 1000:
        scale = 1000 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def _extract_medicine_candidates(text: str) -> List[str]:
    text = unicodedata.normalize('NFKD', text)
    tokens = re.split(r'[\s,;:()\[\]./\\|"\']+', text)
    candidates: List[str] = []
    seen: set = set()
    for token in tokens:
        token = token.strip('.-_*#@!?0123456789')
        lower = token.lower()
        if (
            len(token) >= MIN_MEDICINE_WORD_LEN
            and lower not in _STOP_WORDS
            and not token.isdigit()
            and token[0].isalpha()
            and lower not in seen
        ):
            candidates.append(token)
            seen.add(lower)
    return candidates


def _run_easyocr(img: Image.Image) -> Tuple[str, float, List[str]]:
    reader = _get_easyocr_reader()
    img_array = np.array(img)
    results = reader.readtext(img_array, detail=1, paragraph=False)
    words: List[str] = []
    confs: List[float] = []
    low_conf: List[str] = []
    for (_bbox, text, conf) in results:
        text = text.strip()
        if not text:
            continue
        conf_pct = conf * 100.0
        words.append(text)
        confs.append(conf_pct)
        if conf_pct < CONFIDENCE_WARN_THRESHOLD:
            low_conf.append(text)
    full_text = ' '.join(words)
    mean_conf = (sum(confs) / len(confs)) if confs else -1.0
    return full_text, mean_conf, low_conf


def _run_tesseract(img: Image.Image) -> Tuple[str, float, List[str]]:
    data = pytesseract.image_to_data(
        img, output_type=pytesseract.Output.DICT, CONFIG='--psm 6'
    )
    words_out: List[str] = []
    confs_out: List[float] = []
    low_conf: List[str] = []
    for w, c in zip(data['text'], data['conf']):
        w = w.strip()
        if not w:
            continue
        try:
            c_f = float(c)
        except (ValueError, TypeError):
            continue
        if c_f < 0:
            continue
        words_out.append(w)
        confs_out.append(c_f)
        if c_f < CONFIDENCE_WARN_THRESHOLD:
            low_conf.append(w)
    text = ' '.join(words_out)
    mean_conf = (sum(confs_out) / len(confs_out)) if confs_out else -1.0
    return text, mean_conf, low_conf


def validate_file_extension(filename: str) -> bool:
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in SUPPORTED_FORMATS


def scan_prescription_image(image_bytes: bytes, filename: str = 'upload.jpg') -> PrescriptionScanResult:
    if not validate_file_extension(filename):
        return PrescriptionScanResult(
            status=ScanStatus.INVALID_FILE,
            error_message='Unsupported file type. Please upload a PNG, JPG, or JPEG image.',
        )
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        img = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        return PrescriptionScanResult(
            status=ScanStatus.INVALID_FILE,
            error_message='The uploaded file could not be read as an image. It may be corrupt.',
        )
    except Exception as exc:
        return PrescriptionScanResult(
            status=ScanStatus.INVALID_FILE,
            error_message=f'Image loading failed: {exc}',
        )
    try:
        processed = _preprocess_image(img)
    except Exception as exc:
        return PrescriptionScanResult(
            status=ScanStatus.INVALID_FILE,
            error_message=f'Image pre-processing failed: {exc}',
        )
    text = ''
    mean_conf = -1.0
    low_conf_tokens: List[str] = []
    backend_used = 'none'
    if _EASYOCR_AVAILABLE:
        try:
            text, mean_conf, low_conf_tokens = _run_easyocr(processed)
            backend_used = 'EasyOCR'
        except Exception as exc:
            # EasyOCR failed at runtime (e.g. torchvision::nms mismatch).
            # Fall through to Tesseract if available, otherwise surface error.
            if not _TESSERACT_AVAILABLE:
                return PrescriptionScanResult(
                    status=ScanStatus.BACKEND_UNAVAILABLE,
                    error_message=(
                        f'EasyOCR runtime error: {exc}\n\n'
                        'This is caused by a torch/torchvision version mismatch.\n'
                        'Fix with:\n'
                        '  pip install easyocr --upgrade\n'
                        'Or downgrade torch to a stable release:\n'
                        '  pip install torch==2.3.0 torchvision==0.18.0 --index-url '
                        'https://download.pytorch.org/whl/cpu'
                    ),
                )
    if backend_used == 'none' and _TESSERACT_AVAILABLE:
        try:
            text, mean_conf, low_conf_tokens = _run_tesseract(processed)
            backend_used = 'Tesseract'
        except Exception as exc:
            return PrescriptionScanResult(
                status=ScanStatus.BACKEND_UNAVAILABLE,
                error_message=f'OCR failed: {exc}',
            )
    if backend_used == 'none':
        return PrescriptionScanResult(
            status=ScanStatus.BACKEND_UNAVAILABLE,
            error_message='No OCR backend available. Run: pip install easyocr',
        )
    clean_text = text.strip()
    if not clean_text or not any(c.isalnum() for c in clean_text):
        return PrescriptionScanResult(
            status=ScanStatus.OCR_EMPTY,
            extracted_text=clean_text,
            mean_confidence=mean_conf,
            ocr_backend=backend_used,
            error_message=(
                'No readable text was found in the image. '
                'The image may be too blurry or contain no text. '
                'Please try a clearer photo.'
            ),
        )
    candidates = _extract_medicine_candidates(clean_text)
    if mean_conf >= 0 and mean_conf < CONFIDENCE_WARN_THRESHOLD:
        return PrescriptionScanResult(
            status=ScanStatus.OCR_UNCERTAIN,
            extracted_text=clean_text,
            medicine_candidates=candidates,
            mean_confidence=round(mean_conf, 1),
            low_confidence_tokens=low_conf_tokens,
            ocr_backend=backend_used,
            error_message=(
                f'OCR confidence is low ({mean_conf:.1f}%). '
                'Handwriting may be unclear. '
                'Please verify medicine names before proceeding.'
            ),
        )
    return PrescriptionScanResult(
        status=ScanStatus.SUCCESS,
        extracted_text=clean_text,
        medicine_candidates=candidates,
        mean_confidence=round(mean_conf, 1),
        low_confidence_tokens=low_conf_tokens,
        ocr_backend=backend_used,
    )


def format_rag_drug_card(doc: dict) -> str:
    NO_DATA = '[NO DATA]'

    def _field(label: str, value: str, icon: str) -> str:
        if not value or str(value).strip() == NO_DATA:
            return f'**{icon} {label}:** *Not available in dataset*\n'
        return f'**{icon} {label}:** {value}\n'

    name = doc.get('name', 'Unknown')
    return (
        f'### {name}\n\n'
        + _field('Composition', doc.get('composition', NO_DATA), '\U0001F9EA')
        + '\n'
        + _field('Common Uses / Indications', doc.get('indications', NO_DATA), '\U0001F48A')
        + '\n'
        + _field('Dosage', doc.get('dosage', NO_DATA), '\U0001F4CB')
        + '\n'
        + _field('Contraindications', doc.get('contraindications', NO_DATA), '\U000026A0')
        + '\n'
        + _field('Possible Side Effects', doc.get('side_effects', NO_DATA), '\U0001F52C')
    )
