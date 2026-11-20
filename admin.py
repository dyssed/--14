"""Веб-панель администратора: вопросы-ответы, документы, неизвестные вопросы,
режим ответов, статистика и кнопка «Переобучить».
Работает в том же процессе, что и бот, поэтому после нажатия кнопки бот сразу
использует обновлённую базу."""
import asyncio
import base64
import html
import json
import logging
import os
import re
import secrets
import time
import uuid
from collections import Counter, deque
from pathlib import Path
from urllib.parse import quote

from aiohttp import web

from rag import SUPPORTED, load_qa, save_qa

log = logging.getLogger("admin")

# ---------------------------------------------------------------- состояние

recent: deque = deque(maxlen=50)   # последние вопросы к боту (очищается при перезапуске)
stats: Counter = Counter()         # счётчики с момента запуска
state = {"dirty": False}           # есть изменения, ещё не применённые кнопкой «Переобучить»

KIND_LABELS = {
    "qa": "из базы вопросов-ответов",
    "docs": "по документам",
    "unknown": "нет в базе",
    "strict_unknown": "нет в базе (строгий режим)",
    "off_topic": "вне темы",
    "rate_limit": "лимит запросов",
    "error": "ошибка",
    "empty": "пустой ответ модели",
}
UNKNOWN_KINDS = ("unknown", "strict_unknown", "off_topic")
UNKNOWN_MAX = 500


def record(question: str, answer: str, kind: str = "docs") -> None:
    recent.appendleft({"q": question, "a": answer, "kind": kind})
    stats[kind] += 1


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


# ------------------------------------------------- неизвестные вопросы (файл)

def _upath() -> Path:
    return Path(os.getenv("UNKNOWN_FILE", "unknown.json"))


def load_unknown() -> list[dict]:
    p = _upath()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        log.exception("Не удалось прочитать %s", p)
        return []


def save_unknown(items: list[dict]) -> None:
    p = _upath()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def add_unknown(question: str, kind: str) -> None:
    """Запоминаем вопрос, на который в базе нет ответа (или он вне темы)."""
    key = _norm(question)[:300]
    if not key:
        return
    items = load_unknown()
    now = time.time()
    for it in items:
        if it["key"] == key:
            it["count"] += 1
            it["last"] = now
            it["kind"] = kind
            break
    else:
        items.append({"key": key, "question": question.strip()[:500], "kind": kind, "count": 1, "last": now})
    items.sort(key=lambda i: i["last"], reverse=True)
    save_unknown(items[:UNKNOWN_MAX])


# ------------------------------------------------------- настройки (файл)

def _spath() -> Path:
    return Path(os.getenv("SETTINGS_FILE", "settings.json"))


def load_settings() -> dict:
    p = _spath()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_strict() -> bool:
    """True — отвечать только по контексту (документы и пары «вопрос-ответ»)."""
    s = load_settings()
    if "strict_docs" in s:
        return bool(s["strict_docs"])
    return os.getenv("STRICT_DOCS", "0") == "1"


def set_strict(value: bool) -> None:
    s = load_settings()
    s["strict_docs"] = bool(value)
    p = _spath()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


# ------------------------------------------------ защита паролем (HTTP Basic)

def _make_auth(user: str, password: str):
    @web.middleware
    async def auth(request, handler):
        if password:
            ok = False
            header = request.headers.get("Authorization", "")
            if header.startswith("Basic "):
                try:
                    u, _, p = base64.b64decode(header[6:]).decode("utf-8").partition(":")
                    ok = (secrets.compare_digest(u.encode(), user.encode())
                          and secrets.compare_digest(p.encode(), password.encode()))
                except Exception:
                    ok = False
            if not ok:
                return web.Response(status=401, text="Нужен логин и пароль",
                                    headers={"WWW-Authenticate": 'Basic realm="Bot admin"'})
        return await handler(request)
    return auth


# ------------------------------------------------------------------ страница

CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#1d1d1f;background:#f6f7f9}
h1{font-size:22px} h2{font-size:18px;margin-top:0}
section{background:#fff;border:1px solid #e1e3e8;border-radius:10px;padding:16px 18px;margin-top:16px}
textarea,input[type=text]{width:100%;box-sizing:border-box;padding:8px;border:1px solid #c9ccd3;border-radius:6px;font:inherit}
textarea{min-height:80px}
button{padding:8px 14px;border:0;border-radius:6px;background:#2f6fed;color:#fff;font:inherit;cursor:pointer}
button.del{background:#d64545;padding:4px 10px;font-size:13px}
button.big{background:#1a8f4a;font-size:16px;padding:10px 18px}
.msg{background:#e7f5ec;border:1px solid #b8dfc6;padding:10px 14px;border-radius:8px;margin-top:12px}
.warn{background:#fff4dd;border:1px solid #f0d28a;padding:10px 14px;border-radius:8px;margin-top:12px}
table{width:100%;border-collapse:collapse;margin-top:10px}
td,th{border-top:1px solid #e1e3e8;padding:8px;text-align:left;vertical-align:top;font-size:14px}
.muted{color:#6b6f78;font-size:13px} .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
a.fix{font-size:13px} label.opt{display:block;margin:8px 0}
"""


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _short(s: str, n: int = 300) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def _fmt_time(ts: float) -> str:
    return time.strftime("%d.%m %H:%M", time.localtime(ts))


def _label(kind: str) -> str:
    return KIND_LABELS.get(kind, kind)


def render(index, msg: str = "", prefill_q: str = "") -> str:
    qa = load_qa(index.qa_file)
    docs = sorted(p for p in index.folder.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    unknown = sorted(load_unknown(), key=lambda i: (-i["count"], -i["last"]))
    strict = is_strict()

    banner = ""
    if msg:
        banner += f'<div class="msg">{_esc(msg)}</div>'
    if state["dirty"]:
        banner += ('<div class="warn">Есть изменения, которые бот ещё не использует. '
                   'Нажмите «Переобучить», чтобы применить их.</div>')

    # статистика
    total = sum(stats.values())
    unknown_n = sum(stats[k] for k in UNKNOWN_KINDS)
    bad_n = stats["error"] + stats["rate_limit"] + stats["empty"]
    stats_line = (f"Вопросов с момента запуска: <b>{total}</b>. По документам: {stats['docs']}, "
                  f"из базы пар: {stats['qa']}, неизвестных: {unknown_n} (из них вне темы: {stats['off_topic']}), "
                  f"ошибок и лимитов: {bad_n}.")

    qa_rows = "".join(
        f"<tr><td><b>{_esc(i['question'])}</b><br>{_esc(_short(i['answer'], 400))}</td>"
        f'<td style="width:90px"><form method="post" action="/qa/delete" onsubmit="return confirm(\'Удалить пару?\')">'
        f'<input type="hidden" name="id" value="{_esc(i["id"])}"><button class="del">Удалить</button></form></td></tr>'
        for i in qa
    ) or '<tr><td class="muted">Пока пусто.</td></tr>'

    doc_rows = "".join(
        f"<tr><td>{_esc(p.name)} <span class='muted'>({p.stat().st_size // 1024 + 1} КБ)</span></td>"
        f'<td style="width:90px"><form method="post" action="/doc/delete" onsubmit="return confirm(\'Удалить файл?\')">'
        f'<input type="hidden" name="name" value="{_esc(p.name)}"><button class="del">Удалить</button></form></td></tr>'
        for p in docs
    ) or '<tr><td class="muted">Документов нет.</td></tr>'

    unknown_rows = "".join(
        f"<tr><td><b>{_esc(i['question'])}</b><br><span class='muted'>{_esc(_label(i['kind']))} · "
        f"спрашивали: {i['count']} · {_fmt_time(i['last'])}</span></td>"
        f'<td style="width:190px"><a class="fix" href="/?q={quote(i["question"])}#add">Добавить ответ</a>'
        f'<form method="post" action="/unknown/delete" style="display:inline;margin-left:8px">'
        f'<input type="hidden" name="key" value="{_esc(i["key"])}"><button class="del">Скрыть</button></form></td></tr>'
        for i in unknown
    ) or '<tr><td class="muted">Неизвестных вопросов пока нет.</td></tr>'

    recent_rows = "".join(
        f"<tr><td><b>{_esc(r['q'])}</b> <span class='muted'>[{_esc(_label(r['kind']))}]</span><br>"
        f"<span class='muted'>{_esc(_short(r['a'], 250))}</span></td>"
        f'<td style="width:110px"><a class="fix" href="/?q={quote(r["q"])}#add">Исправить</a></td></tr>'
        for r in recent
    ) or '<tr><td class="muted">Бот пока не получал вопросов с момента запуска.</td></tr>'

    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Админка бота</title><style>{CSS}</style></head><body>
<h1>Админка бота</h1>
{banner}

<section>
  <form method="post" action="/rebuild" class="row">
    <button class="big">Переобучить</button>
    <span class="muted">Перечитывает документы и пары «вопрос-ответ». Модель при этом не меняется:
    бот просто начинает использовать обновлённую базу знаний (занимает несколько секунд).</span>
  </form>
  <p class="muted">Сейчас в базе: {len(index.files)} документов, {index.qa_count} пар «вопрос-ответ», {len(index.chunks)} фрагментов.</p>
  <p class="muted">{stats_line}</p>
</section>

<section>
  <h2>Режим ответов</h2>
  <form method="post" action="/settings">
    <label class="opt"><input type="radio" name="mode" value="strict" {"checked" if strict else ""}>
      <b>Только по контексту</b> — документы и пары «вопрос-ответ». Если в базе ничего нет, бот так и скажет
      и не обращается к модели.</label>
    <label class="opt"><input type="radio" name="mode" value="context_plus" {"" if strict else "checked"}>
      <b>Контекст + общие знания</b> — если в базе ответа нет, модель отвечает по энергетике из общих знаний
      (с пометкой). Вопросы вне темы модель отклоняет по правилам из системного промпта.</label>
    <button>Сохранить режим</button> <span class="muted">Действует сразу, «Переобучить» не нужно.</span>
  </form>
</section>

<section>
  <h2>Неизвестные вопросы</h2>
  <p class="muted">Сюда попадают вопросы, на которые в базе нет ответа, и вопросы не по теме бота (например,
  «какая погода в Новой Зеландии»). Если вопрос по делу, нажмите «Добавить ответ» и внесите правильный ответ.
  Если он не по теме, нажмите «Скрыть». Приветствия и короткие фразы (меньше трёх слов) не учитываются.</p>
  <table>{unknown_rows}</table>
  <form method="post" action="/unknown/clear" onsubmit="return confirm('Очистить весь список?')" style="margin-top:10px">
    <button class="del">Очистить список</button>
  </form>
</section>

<section id="add">
  <h2>Вопросы и ответы</h2>
  <p class="muted">Если бот ответил плохо, добавьте сюда вопрос и правильный ответ. Когда новый вопрос почти совпадает
  с сохранённым, бот отвечает сохранённым текстом; на похожие вопросы использует его как главный источник.</p>
  <form method="post" action="/qa/add">
    <p><label>Вопрос<br><input type="text" name="question" value="{_esc(prefill_q)}" required maxlength="500"></label></p>
    <p><label>Правильный ответ<br><textarea name="answer" required maxlength="3000"></textarea></label></p>
    <button>Добавить пару</button>
  </form>
  <table>{qa_rows}</table>
</section>

<section>
  <h2>Документы</h2>
  <form method="post" action="/doc/upload" enctype="multipart/form-data" class="row">
    <input type="file" name="file" multiple accept=".txt,.md,.pdf,.docx" required>
    <button>Загрузить</button>
    <span class="muted">TXT, MD, PDF, DOCX</span>
  </form>
  <table>{doc_rows}</table>
</section>

<section>
  <h2>Последние вопросы к боту</h2>
  <p class="muted">Список очищается при перезапуске бота. Нажмите «Исправить», чтобы перенести вопрос в форму выше.</p>
  <table>{recent_rows}</table>
</section>
</body></html>"""


# ---------------------------------------------------------------- обработчики

def _back(msg: str = ""):
    raise web.HTTPSeeOther("/?msg=" + quote(msg) if msg else "/")


async def page(request: web.Request):
    index = request.app["index"]
    return web.Response(text=render(index, request.query.get("msg", ""), request.query.get("q", "")),
                        content_type="text/html")


async def qa_add(request: web.Request):
    index = request.app["index"]
    data = await request.post()
    q = (data.get("question") or "").strip()[:500]
    a = (data.get("answer") or "").strip()[:3000]
    if not q or not a:
        _back("Заполните и вопрос, и ответ.")
    items = load_qa(index.qa_file)
    items.append({"id": uuid.uuid4().hex[:8], "question": q, "answer": a})
    save_qa(index.qa_file, items)
    # вопрос больше не «неизвестный»
    key = _norm(q)[:300]
    save_unknown([i for i in load_unknown() if i["key"] != key])
    state["dirty"] = True
    _back("Пара добавлена. Нажмите «Переобучить», чтобы бот её использовал.")


async def qa_delete(request: web.Request):
    index = request.app["index"]
    data = await request.post()
    qid = data.get("id")
    save_qa(index.qa_file, [i for i in load_qa(index.qa_file) if i["id"] != qid])
    state["dirty"] = True
    _back("Пара удалена. Нажмите «Переобучить», чтобы применить.")


async def doc_upload(request: web.Request):
    index = request.app["index"]
    data = await request.post()
    saved, skipped = [], []
    index.folder.mkdir(exist_ok=True)
    for f in data.getall("file", []):
        name = Path(getattr(f, "filename", "") or "").name
        if not name or name.startswith(".") or Path(name).suffix.lower() not in SUPPORTED:
            if name:
                skipped.append(name)
            continue
        (index.folder / name).write_bytes(f.file.read())
        saved.append(name)
    if saved:
        state["dirty"] = True
    msg = f"Загружено: {', '.join(saved) or 'ничего'}."
    if skipped:
        msg += f" Пропущены (неподдерживаемый формат): {', '.join(skipped)}."
    if saved:
        msg += " Нажмите «Переобучить», чтобы бот их использовал."
    _back(msg)


async def doc_delete(request: web.Request):
    index = request.app["index"]
    data = await request.post()
    path = index.folder / Path(data.get("name") or "").name
    if path.is_file() and path.suffix.lower() in SUPPORTED:
        path.unlink()
        state["dirty"] = True
        _back(f"Файл {path.name} удалён. Нажмите «Переобучить», чтобы применить.")
    _back("Файл не найден.")


async def settings_save(request: web.Request):
    data = await request.post()
    set_strict(data.get("mode") == "strict")
    _back("Режим сохранён и уже действует.")


async def unknown_delete(request: web.Request):
    data = await request.post()
    key = data.get("key")
    save_unknown([i for i in load_unknown() if i["key"] != key])
    _back("Вопрос скрыт.")


async def unknown_clear(request: web.Request):
    save_unknown([])
    _back("Список неизвестных вопросов очищен.")


async def rebuild(request: web.Request):
    index = request.app["index"]
    await asyncio.to_thread(index.build)
    state["dirty"] = False
    _back(f"Готово: {len(index.files)} документов, {index.qa_count} пар «вопрос-ответ», "
          f"{len(index.chunks)} фрагментов. Бот уже использует обновлённую базу.")


async def start_admin(index, host: str = "127.0.0.1", port: int = 8080,
                      user: str = "admin", password: str = "") -> web.AppRunner:
    app = web.Application(middlewares=[_make_auth(user, password)], client_max_size=50 * 1024 * 1024)
    app["index"] = index
    app.add_routes([
        web.get("/", page),
        web.post("/qa/add", qa_add), web.post("/qa/delete", qa_delete),
        web.post("/doc/upload", doc_upload), web.post("/doc/delete", doc_delete),
        web.post("/settings", settings_save),
        web.post("/unknown/delete", unknown_delete), web.post("/unknown/clear", unknown_clear),
        web.post("/rebuild", rebuild),
    ])
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    if host not in ("127.0.0.1", "localhost") and not password:
        log.warning("Админка открыта в сеть без пароля! Задайте ADMIN_PASSWORD в .env.")
    log.info("Админка: http://%s:%s", "localhost" if host == "127.0.0.1" else host, port)
    return runner
