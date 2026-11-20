"""Проверка ключа GigaChat без бота и без Docker.
Запуск (из папки проекта, где лежит .env):   python check_gigachat.py
Ключ и полученный токен на экран НЕ выводятся."""
import sys
import uuid

import httpx
from dotenv import dotenv_values

env = dotenv_values(".env")
key = (env.get("GIGACHAT_CREDENTIALS") or "")
scope = (env.get("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS").strip()
ca = (env.get("GIGACHAT_CA_BUNDLE_FILE") or "").strip() or None
verify_off = (env.get("GIGACHAT_VERIFY_SSL_CERTS") or "").strip().lower() == "false"

if not key:
    sys.exit("В .env нет строки GIGACHAT_CREDENTIALS")

print("--- что лежит в .env (сам ключ не показываю) ---")
print(f"длина ключа: {len(key)}")
print(f"заканчивается на '==': {key.endswith('==')}")
print(f"лишние пробелы по краям: {key != key.strip()}")
print(f"пробелы или переносы внутри: {any(c.isspace() for c in key.strip())}")
print(f"начинается с 'Basic ': {key.strip().lower().startswith('basic ')}  (должно быть False, слово Basic вписывать не нужно)")
print(f"в кавычках: {key.strip()[:1] in ('\"', chr(39))}")
print(f"scope: {scope}")

key = key.strip()
if key.lower().startswith("basic "):
    key = key[6:].strip()

print("\n--- запрос токена доступа ---")
try:
    r = httpx.post(
        "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        headers={"Authorization": "Basic " + key, "RqUID": str(uuid.uuid4()),
                 "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        data={"scope": scope},
        verify=False if verify_off else (ca or True),
        timeout=20,
    )
except Exception as e:
    sys.exit(f"Не удалось соединиться: {type(e).__name__}: {str(e)[:200]}")

if r.status_code == 200:
    print("ОК: ключ рабочий, токен доступа получен.")
else:
    print(f"Ошибка {r.status_code}: {r.text[:200]}")
    if r.status_code == 401:
        print("Сервер не узнал ключ: нужен новый ключ авторизации из личного кабинета (или scope не тот).")
