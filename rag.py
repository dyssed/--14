"""Поиск по документам и базе «вопрос-ответ»: загрузка, нарезка на фрагменты, BM25."""
import json
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

log = logging.getLogger("rag")

CHUNK_SIZE = 1000      # примерный размер фрагмента, символов
SUPPORTED = {".txt", ".md", ".pdf", ".docx"}
QA_SOURCE = "Вопросы и ответы"
QA_BOOST = 1.5         # пары «вопрос-ответ» ищутся с повышенным весом
MIN_MATCH_LONG = 2     # для вопросов из 3+ значимых слов во фрагменте должно найтись минимум 2
QA_MATCH = 0.6         # порог «вопрос совпал с сохранённым» для прямого ответа
TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Слова в нижнем регистре, обрезанные до 6 символов (грубая замена стемминга,
    чтобы «трансформатор» и «трансформатора» считались одним словом)."""
    out = []
    for t in TOKEN_RE.findall(text.lower().replace("ё", "е")):
        out.append(t[:6])
    return out


# Служебные слова, которые не учитываем при сравнении вопроса с сохранёнными
_STOP_WORDS = ("что такое это как какой какая какое какие каков каком каких какую какому для при или про и в во на по а "
               "из у с со о об от до за не ни бы же ли так то да нет но либо тоже также чтобы если когда где кто чем "
               "почему зачем сколько есть быть был была было были все всё его её их мне нам вам вы ты я мы он она оно они "
               "этот эта эти этом этого тот такой такая можно нужно надо ещё еще очень уже только "
               "расскажи скажи объясни подскажи пожалуйста дай покажи напиши посоветуй хочу хотел хотела")
STOP = {t for w in _STOP_WORDS.split() for t in tokenize(w)}


def _match_tokens(text: str) -> frozenset:
    return frozenset(t for t in tokenize(text) if t not in STOP)


@dataclass
class Chunk:
    source: str
    page: int | None
    text: str
    tf: Counter
    length: int
    kind: str = "doc"          # "doc" — фрагмент документа, "qa" — пара «вопрос-ответ»

    @property
    def label(self) -> str:
        return f"{self.source}, стр. {self.page}" if self.page else self.source


# ---------- база «вопрос-ответ» (JSON-файл) ----------

def load_qa(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [d for d in data if d.get("question") and d.get("answer")]
    except Exception:
        log.exception("Не удалось прочитать %s", p)
        return []


def save_qa(path, items: list[dict]) -> None:
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


# ---------- чтение файлов: возвращают список (страница|None, текст) ----------

def _read_txt(path: Path):
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1251"):
        try:
            return [(None, raw.decode(enc))]
        except UnicodeDecodeError:
            continue
    return [(None, raw.decode("utf-8", errors="replace"))]


def _read_pdf(path: Path):
    from pypdf import PdfReader
    pages = []
    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        pages.append((i, page.extract_text() or ""))
    if not any(t.strip() for _, t in pages):
        log.warning("%s: текста нет (скан?). Нужен OCR.", path.name)
    return pages


def _read_docx(path: Path):
    from docx import Document
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return [(None, "\n".join(parts))]


READERS = {".txt": _read_txt, ".md": _read_txt, ".pdf": _read_pdf, ".docx": _read_docx}


def _split(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Режем по абзацам, склеивая их до нужного размера."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paragraphs:
        while len(p) > size:          # очень длинный абзац режем жёстко
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:size])
            p = p[size:]
        if len(cur) + len(p) + 1 > size and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


class Index:
    def __init__(self, folder: str | Path, qa_file: str | Path = "qa.json"):
        self.folder = Path(folder)
        self.qa_file = Path(qa_file)
        self._s = SimpleNamespace(chunks=[], files=[], df=Counter(), avg_len=1.0, qa=[], qa_count=0)

    # Состояние подменяется целиком в конце build(), поэтому поиск во время
    # пересборки не видит «половину» индекса.
    @property
    def chunks(self) -> list[Chunk]:
        return self._s.chunks

    @property
    def files(self) -> list[str]:
        return self._s.files

    @property
    def qa_count(self) -> int:
        return self._s.qa_count

    def build(self) -> None:
        chunks: list[Chunk] = []
        files: list[str] = []
        self.folder.mkdir(exist_ok=True)

        for path in sorted(self.folder.rglob("*")):
            if path.suffix.lower() not in SUPPORTED or not path.is_file():
                continue
            try:
                pages = READERS[path.suffix.lower()](path)
            except Exception:
                log.exception("Не удалось прочитать %s", path.name)
                continue
            added = 0
            for page, text in pages:
                for piece in _split(text):
                    tokens = tokenize(piece)
                    if not tokens:
                        continue
                    chunks.append(Chunk(path.name, page, piece, Counter(tokens), len(tokens)))
                    added += 1
            if added:
                files.append(path.name)
            log.info("%s: %d фрагментов", path.name, added)

        qa_items = load_qa(self.qa_file)
        qa_keys = []
        for item in qa_items:
            text = f"Вопрос: {item['question']}\nОтвет: {item['answer']}"
            tokens = tokenize(text)
            chunks.append(Chunk(QA_SOURCE, None, text, Counter(tokens), len(tokens), "qa"))
            qa_keys.append((_match_tokens(item["question"]), item["answer"]))
        log.info("Пар «вопрос-ответ»: %d", len(qa_items))

        df: Counter = Counter()
        for c in chunks:
            df.update(c.tf.keys())
        avg_len = (sum(c.length for c in chunks) / len(chunks)) if chunks else 1.0
        self._s = SimpleNamespace(chunks=chunks, files=files, df=df, avg_len=avg_len,
                                  qa=qa_keys, qa_count=len(qa_items))

    def match_qa(self, question: str, threshold: float = QA_MATCH) -> str | None:
        """Если вопрос почти совпадает с сохранённым — вернуть сохранённый ответ."""
        q = _match_tokens(question)
        if not q:
            return None
        best_answer, best_score = None, 0.0
        for keys, answer in self._s.qa:
            if not keys:
                continue
            score = len(q & keys) / len(q | keys)
            if score > best_score:
                best_answer, best_score = answer, score
        return best_answer if best_score >= threshold else None

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        s = self._s
        q = {t for t in tokenize(query) if t not in STOP}
        if not q or not s.chunks:
            return []
        need = MIN_MATCH_LONG if len(q) >= 3 else 1   # сколько слов вопроса должно найтись во фрагменте
        n, k1, b = len(s.chunks), 1.5, 0.75
        scored = []
        for c in s.chunks:
            score = 0.0
            matched = 0
            for t in q:
                f = c.tf.get(t)
                if not f:
                    continue
                matched += 1
                idf = math.log(1 + (n - s.df[t] + 0.5) / (s.df[t] + 0.5))
                score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * c.length / s.avg_len))
            if score > 0 and matched >= need:
                scored.append((score * (QA_BOOST if c.kind == "qa" else 1.0), c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]
