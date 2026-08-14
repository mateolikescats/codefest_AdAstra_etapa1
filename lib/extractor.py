import os
import re
import json
import unicodedata
import openpyxl
import pymupdf as fitz  # PyMuPDF
import pandas as pd
from typing import Dict, List, Any

class CorpusExtractor:
    def __init__(self, base_dir: str, excel_index_path: str):
        self.base_dir = base_dir
        self.excel_index_path = excel_index_path
        self.doc_index = self._load_excel_index()

    def _load_excel_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads Indice_Datos_Codefest.xlsx ('Inventario de Archivos' sheet)
        and builds a lookup map from relative file path -> metadata dict with official DOC_ID.
        """
        wb = openpyxl.load_workbook(self.excel_index_path, data_only=True)
        sheet = wb['Inventario de Archivos']
        index_map = {}
        
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if not row or not row[3]:
                continue
            fenomeno_str = str(row[0]) if row[0] else ""
            # Extract phenomenon integer (F1 -> 1, F2 -> 2, F3 -> 3)
            fen_match = re.search(r'F(\d)', fenomeno_str)
            fenomeno_int = int(fen_match.group(1)) if fen_match else 1
            
            doc_id = str(row[3]).strip()
            filename = str(row[4]).strip() if row[4] else ""
            folder = str(row[5]).strip() if row[5] else ""
            doc_type = str(row[6]).strip().lower() if row[6] else "txt"
            
            # Construct normalized relative path key
            rel_path = os.path.normpath(os.path.join(folder, filename))
            
            index_map[rel_path] = {
                "doc_id": doc_id,
                "fenomeno": fenomeno_int,
                "fuente": rel_path.replace("\\", "/"),
                "formato": doc_type,
                "observatorio": str(row[1]) if row[1] else "",
                "filename": filename,
                "folder": folder
            }
            
            # Also map by filename alone for fallback matching (only if not already mapped)
            if filename not in index_map:
                index_map[filename] = index_map[rel_path]
            
        wb.close()
        return index_map

    def extract_document(self, full_filepath: str, rel_filepath: str) -> Dict[str, Any]:
        """
        Extracts raw text and metadata from a document file.
        """
        norm_rel = os.path.normpath(rel_filepath)
        filename = os.path.basename(full_filepath)
        
        meta = self.doc_index.get(norm_rel) or self.doc_index.get(filename)
        if not meta:
            ext = os.path.splitext(filename)[1].replace(".", "").lower()
            meta = {
                "doc_id": f"DOC-{abs(hash(rel_filepath)) % 100000:05d}",
                "fenomeno": 1,
                "fuente": rel_filepath.replace("\\", "/"),
                "formato": ext,
                "observatorio": "Desconocido",
                "filename": filename
            }
        
        text = ""
        ext = meta["formato"].lower()
        
        try:
            if ext == "pdf":
                text = self._extract_pdf(full_filepath)
            elif ext in ["html", "htm"]:
                text = self._extract_html(full_filepath)
            elif ext in ["txt", "md", "markdown"]:
                text = self._extract_text_file(full_filepath)
            elif ext == "json":
                text = self._extract_json(full_filepath)
            elif ext in ["csv", "xlsx", "xls"]:
                text = self._extract_tabular(full_filepath, ext)
            else:
                text = self._extract_text_file(full_filepath)
        except Exception as e:
            text = f"Error al extraer archivo {filename}: {str(e)}"
            
        cleaned_text = self._clean_text(text)
        
        return {
            "doc_id": meta["doc_id"],
            "fenomeno": meta["fenomeno"],
            "fuente": meta["fuente"],
            "formato": meta["formato"],
            "observatorio": meta["observatorio"],
            "filename": meta["filename"],
            "text": cleaned_text
        }

    def _extract_pdf(self, path: str) -> str:
        import pypdf
        pages_text = []
        try:
            reader = pypdf.PdfReader(path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    # Limpiar caracteres surrogate para evitar fallos internos en el tokenizador de HuggingFace
                    text_clean = text.encode('utf-8', 'surrogatepass').decode('utf-8', 'ignore')
                    pages_text.append(text_clean)
        except Exception as e:
            # Fallback a PyMuPDF en caso de error inesperado
            pages_text = []
            with fitz.open(path) as doc:
                for page in doc:
                    p_text = page.get_text("text")
                    if p_text and p_text.strip():
                        pages_text.append(p_text.strip())
        return "\n\n".join(pages_text)

    def _extract_html(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        content = re.sub(r'<(h[1-6]|p|div|br|li)[^>]*>', '\n', content, flags=re.IGNORECASE)
        content = re.sub(r'<[^>]+>', ' ', content)
        return content

    def _extract_text_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _extract_json(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        
        parts = []
        target_keys = {
            "title", "heading", "name", "subject", "summary", "body_text",
            "body_paragraphs", "content", "description", "text", "titulo",
            "resumen", "contenido", "descripcion", "texto", "parrafos"
        }
        
        def _recurse_json(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k.lower() in target_keys:
                        if isinstance(v, str):
                            parts.append(v)
                        elif isinstance(v, list):
                            parts.extend([str(item) for item in v if item])
                    else:
                        _recurse_json(v)
            elif isinstance(obj, list):
                for item in obj:
                    _recurse_json(item)
            elif isinstance(obj, str):
                parts.append(obj)

        _recurse_json(data)
        return "\n\n".join(parts) if parts else json.dumps(data, ensure_ascii=False)

    def _extract_tabular(self, path: str, ext: str) -> str:
        if ext == "csv":
            df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        else:
            df = pd.read_excel(path)
            
        lines = []
        headers = list(df.columns)
        for _, row in df.iterrows():
            row_str = " | ".join([f"{h}: {row[h]}" for h in headers if pd.notna(row[h])])
            if row_str.strip():
                lines.append(row_str)
        return "\n".join(lines)

    def _clean_text(self, text: str) -> str:
        """Normalizes Unicode text to NFC, removes control chars and sanitizes secrets."""
        if not text:
            return ""
        # Unicode normalization
        text = unicodedata.normalize("NFC", text)
        # Remove control/non-printable characters except newline and tab
        text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
        # Sanitize API token patterns (e.g. Mapbox public tokens)
        text = re.sub(r'pk\.ey[a-zA-Z0-9_\-\.]+', 'pk.ey_REDACTED_TOKEN', text)
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
