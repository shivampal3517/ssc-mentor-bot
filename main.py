import html
import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.schedulers.background import BackgroundScheduler
import telebot
from google import genai
from google.genai import types

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def required_env(name: str) -> str:
    try:
        value = os.environ[name].strip()
    except KeyError as exc:
        raise RuntimeError(f"Missing required environment variable: {name}") from exc
    if not value:
        raise RuntimeError(f"Environment variable {name} is empty")
    return value

TELEGRAM_BOT_TOKEN = required_env("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = required_env("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"
DATABASE_PATH = os.getenv("DATABASE_PATH", "ssc_mentor.db")
BOT_TIMEZONE_NAME = os.getenv("BOT_TIMEZONE", "Asia/Kolkata")
MAX_MEMORY_MESSAGES = 8
UTC = timezone.utc

try:
    BOT_TIMEZONE = ZoneInfo(BOT_TIMEZONE_NAME)
except ZoneInfoNotFoundError as exc:
    raise RuntimeError(f"Unknown BOT_TIMEZONE: {BOT_TIMEZONE_NAME}") from exc

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
client = genai.Client(api_key=GEMINI_API_KEY)

SSC_MENTOR_SYSTEM_INSTRUCTION = """
You are a 100x SSC CGL AI Mentor and TCS exam strategist.

Tone and style:
- Reply in casual, crisp, natural Hinglish.
- Be direct, beginner-friendly, and exam-focused. No generic motivational filler.
- Understand follow-up questions from the conversation context.

Mandatory answer format:
1. Give ONLY ONE fastest TCS exam shortcut first. Choose the best single approach: Ratio, Digital Sum, Divisibility, or Option Elimination.
2. Explain that shortcut in only 3–4 short lines or bullets.
3. Use strictly plain-text math. NEVER use LaTeX, "$", "$$", "\\(", "\\)", or math code blocks. Write examples like: Required value = (60 / 20) x 35.
4. Put the final answer in bold using **Final Answer: ...**.
5. End with exactly one short line beginning with "Common Trap:".
6. Do not provide multiple methods unless the student explicitly asks for another.

For English, Reasoning, and GS questions, keep the same concise structure:
give the fastest exam insight first, then the answer, then one Common Trap line.
Never invent facts. If an image or question is unclear, say what is unclear.
"""

SSC_VISION_INSTRUCTION = """
Solve this SSC CGL question from the attached screenshot.
Read the image carefully and do not guess text that is not legible.
Identify the subject and question, then use the single fastest TCS shortcut.
Follow the mentor system rules exactly: casual Hinglish, plain-text math only,
no LaTeX or dollar-sign math, 3–4 concise lines or bullets, bold final answer,
and one Common Trap line.
"""

# ----------------------------- Rolling memory ----------------------------- #

conversation_memory: dict[int, list[dict[str, str]]] = {}
memory_lock = threading.RLock()

def get_memory(chat_id: int) -> list[dict[str, str]]:
    with memory_lock:
        return list(conversation_memory.get(chat_id, []))

def remember(chat_id: int, role: str, content: str) -> None:
    with memory_lock:
        history = conversation_memory.setdefault(chat_id, [])
        history.append({"role": role, "content": content})
        del history[:-MAX_MEMORY_MESSAGES]

def clear_memory(chat_id: int) -> None:
    with memory_lock:
        conversation_memory.pop(chat_id, None)

def memory_contents(chat_id: int) -> list[types.Content]:
    return [
        types.Content(
            role=item["role"],
            parts=[types.Part.from_text(text=item["content"])],
        )
        for item in get_memory(chat_id)
    ]

def model_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SSC_MENTOR_SYSTEM_INSTRUCTION,
        temperature=0.45,
        max_output_tokens=1200,
    )

def generate_reply(
    chat_id: int,
    user_text: str,
    *,
    image_bytes: bytes | None = None,
    mime_type: str = "image/jpeg",
    instruction: str | None = None,
    save_to_memory: bool = True,
) -> str:
    contents = memory_contents(chat_id)
    prompt = instruction or user_text
    if image_bytes is not None:
        parts = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            types.Part.from_text(text=prompt),
        ]
    else:
        parts = [types.Part.from_text(text=prompt)]
    contents.append(types.Content(role="user", parts=parts))

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=model_config(),
    )
    reply = (response.text or "").strip()
    if not reply:
        raise RuntimeError("Gemini returned an empty response")
    if save_to_memory:
        remember(chat_id, "user", user_text)
        remember(chat_id, "model", reply)
    return reply

def generate_scheduled_text(instruction: str, fallback: str) -> str:
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=instruction,
            config=model_config(),
        )
        return (response.text or "").strip() or fallback
    except Exception:
        logger.exception("Gemini generation failed for scheduled content")
        return fallback

def send_ai_reply(message: telebot.types.Message, text: str) -> None:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped, flags=re.DOTALL)
    try:
        bot.reply_to(message, escaped, parse_mode="HTML")
    except Exception:
        logger.exception("HTML reply failed; sending plain text")
        bot.reply_to(message, text)

# -------------------------------- SQLite --------------------------------- #

SUBJECT_ALIASES = {
    "quant": "Quant",
    "math": "Quant",
    "mathematics": "Quant",
    "english": "English",
    "vocab": "English",
    "vocabulary": "English",
    "reasoning": "Reasoning",
    "logic": "Reasoning",
    "gs": "GS",
    "gk": "GS",
    "general": "GS",
    "general studies": "GS",
}
REVIEW_INTERVALS_DAYS = (1, 3, 7)
REVIEW_STAGE_NAMES = ("Day 1", "Day 3", "Day 7", "Complete")

def utc_now() -> datetime:
    return datetime.now(UTC)

def iso_now() -> str:
    return utc_now().isoformat()

def get_db() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection

def init_db() -> None:
    with get_db() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                chat_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                awaiting_audit INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                mistake_details TEXT NOT NULL,
                review_stage INTEGER NOT NULL DEFAULT 0,
                next_review_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_reviewed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS study_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                report_text TEXT NOT NULL,
                mock_score REAL,
                created_at TEXT NOT NULL
            );
            """
        )

def normalize_subject(subject: str) -> str:
    cleaned = " ".join(subject.strip().lower().split())
    return SUBJECT_ALIASES.get(cleaned, subject.strip().title())

def register_user(message: telebot.types.Message) -> None:
    user = message.from_user
    now = iso_now()
    with get_db() as connection:
        connection.execute(
            """
            INSERT INTO users (chat_id, username, first_name, created_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                username = excluded.username,
                first_name = excluded.first_name,
                last_seen_at = excluded.last_seen_at
            """,
            (
                message.chat.id,
                user.username if user else None,
                user.first_name if user else None,
                now,
                now,
            ),
        )

def log_error(chat_id: int, subject: str, topic: str, details: str) -> None:
    with get_db() as connection:
        connection.execute(
            """
            INSERT INTO errors
                (chat_id, subject, topic, mistake_details, next_review_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                normalize_subject(subject),
                topic.strip(),
                details.strip(),
                (utc_now() + timedelta(days=1)).isoformat(),
                iso_now(),
            ),
        )

def format_errors(chat_id: int) -> str:
    with get_db() as connection:
        rows = connection.execute(
            """
            SELECT subject, topic, COUNT(*) AS total,
                   MIN(next_review_at) AS next_review_at
            FROM errors WHERE chat_id = ?
            GROUP BY subject, topic ORDER BY total DESC
            """,
            (chat_id,),
        ).fetchall()
    if not rows:
        return "Error Notebook empty hai.\nUse: /logerror Quant Percentage silly mistake"
    lines = ["<b>Your Error Notebook</b>"]
    for row in rows:
        lines.append(
            f"• <b>{html.escape(row['subject'])}</b> — "
            f"{html.escape(row['topic'])} ({row['total']} error(s), "
            f"review: {row['next_review_at'][:10]})"
        )
    return "\n".join(lines)

def save_audit(chat_id: int, score: float, report: str) -> None:
    with get_db() as connection:
        connection.execute(
            """
            INSERT INTO study_reports (chat_id, report_text, mock_score, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (chat_id, report.strip(), score, iso_now()),
        )

def format_report(chat_id: int) -> str:
    with get_db() as connection:
        errors = connection.execute(
            "SELECT COUNT(*) AS total FROM errors WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()["total"]
        latest = connection.execute(
            """
            SELECT mock_score, report_text, created_at FROM study_reports
            WHERE chat_id = ? ORDER BY created_at DESC LIMIT 1
            """,
            (chat_id,),
        ).fetchone()
    result = f"<b>SSC Performance Radar</b>\nError notebook entries: {errors}"
    if latest:
        result += (
            f"\nLatest mock score: {latest['mock_score']}"
            f"\nAudit: {html.escape(latest['report_text'])}"
        )
    else:
        result += "\nNo audit saved yet. Use /audit <score> <report>."
    return result

init_db()

# ------------------------------ Scheduled work --------------------------- #

def registered_chat_ids() -> list[int]:
    with get_db() as connection:
        return [row["chat_id"] for row in connection.execute(
            "SELECT chat_id FROM users"
        ).fetchall()]

def broadcast(text: str) -> None:
    for chat_id in registered_chat_ids():
        try:
            bot.send_message(chat_id, text)
        except Exception:
            logger.exception("Failed proactive message to chat %s", chat_id)

def send_due_reviews() -> None:
    now = iso_now()
    with get_db() as connection:
        rows = connection.execute(
            """
            SELECT * FROM errors
            WHERE next_review_at <= ? AND review_stage < 3
            ORDER BY next_review_at
            """,
            (now,),
        ).fetchall()
    for row in rows:
        prompt = (
            "Create one short SSC CGL revision MCQ in Hinglish for this error. "
            "Give four options and do not reveal the answer before the attempt.\n"
            f"Subject: {row['subject']}\nTopic: {row['topic']}\n"
            f"Previous mistake: {row['mistake_details']}"
        )
        question = generate_scheduled_text(prompt, f"Revision: {row['topic']} ko revise karo.")
        try:
            bot.send_message(row["chat_id"], f"Spaced Revision — {question}")
            next_stage = row["review_stage"] + 1
            next_due = (
                (utc_now() + timedelta(days=REVIEW_INTERVALS_DAYS[next_stage])).isoformat()
                if next_stage < 3 else iso_now()
            )
            with get_db() as connection:
                connection.execute(
                    """
                    UPDATE errors
                    SET review_stage = ?, next_review_at = ?, last_reviewed_at = ?
                    WHERE id = ?
                    """,
                    (next_stage, next_due, iso_now(), row["id"]),
                )
        except Exception:
            logger.exception("Failed spaced review for error %s", row["id"])

def send_morning_drill() -> None:
    text = generate_scheduled_text(
        "Create a concise morning SSC CGL drill in Hinglish: three vocabulary "
        "items and one static GS MCQ with four options.",
        "Morning Drill: three vocabulary words revise karo aur ek GS MCQ solve karo.",
    )
    broadcast(f"Morning Drill — {BOT_TIMEZONE_NAME}\n\n{text}")

def send_afternoon_drill() -> None:
    text = generate_scheduled_text(
        "Create exactly three SSC CGL mental-math MCQs in Hinglish on percentage, "
        "fraction, simplification, ratio, or speed. Add a compact answer key.",
        "Afternoon Drill: 25% of 360 = 90. Aaj percentage speed practice karo.",
    )
    broadcast(f"Afternoon Drill — {BOT_TIMEZONE_NAME}\n\n{text}")

def send_night_audit() -> None:
    for chat_id in registered_chat_ids():
        try:
            bot.send_message(
                chat_id,
                "Night Audit: mock score aur aaj ka study report bhejo:\n"
                "/audit <score> <study report>",
            )
        except Exception:
            logger.exception("Failed night audit to chat %s", chat_id)

def start_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=BOT_TIMEZONE)
    scheduler.add_job(send_due_reviews, "interval", minutes=1, id="spaced-reviews",
                      replace_existing=True, max_instances=1, coalesce=True)
    scheduler.add_job(send_morning_drill, "cron", hour=8, minute=0,
                      id="morning-drill", replace_existing=True)
    scheduler.add_job(send_afternoon_drill, "cron", hour=14, minute=0,
                      id="afternoon-drill", replace_existing=True)
    scheduler.add_job(send_night_audit, "cron", hour=22, minute=0,
                      id="night-audit", replace_existing=True)
    scheduler.start()
    logger.info("Scheduler started in %s", BOT_TIMEZONE_NAME)
    return scheduler

# -------------------------------- Commands -------------------------------- #

@bot.message_handler(commands=["start", "help"])
def handle_start(message: telebot.types.Message) -> None:
    register_user(message)
    bot.reply_to(
        message,
        "Namaste! Main aapka 100x SSC CGL AI Mentor hoon.\n\n"
        "Text doubt ya screenshot bhejo — main fastest TCS shortcut dunga.\n"
        "/quiz [topic] — 3 topic-wise TCS MCQs\n"
        "/clear or /reset — conversation memory wipe\n"
        "/logerror <subject> <topic> <mistake>\n"
        "/errors — Error Notebook\n"
        "/audit <score> <report> — daily report\n"
        "/report — performance radar",
    )

@bot.message_handler(commands=["clear", "reset"])
def handle_clear_memory(message: telebot.types.Message) -> None:
    register_user(message)
    clear_memory(message.chat.id)
    bot.reply_to(message, "Conversation memory clear ho gayi. Fresh start karte hain.")

@bot.message_handler(commands=["quiz"])
def handle_quiz(message: telebot.types.Message) -> None:
    register_user(message)
    topic = message.text.partition(" ")[2].strip() or "mixed SSC CGL Quant, Reasoning, English and GS"
    prompt = (
        f"Generate exactly 3 standard TCS-style SSC CGL MCQs on: {topic}.\n"
        "Each must have four options A-D. Give the answer key after all three "
        "questions and one-line solutions. Keep them exam-realistic and concise."
    )
    try:
        reply = generate_reply(message.chat.id, f"/quiz {topic}", instruction=prompt)
        send_ai_reply(message, reply)
    except Exception as exc:
        logger.exception("Quiz generation failed")
        bot.reply_to(message, f"Quiz Error: {exc!r}")

@bot.message_handler(commands=["logerror"])
def handle_logerror(message: telebot.types.Message) -> None:
    register_user(message)
    payload = message.text.partition(" ")[2].strip()
    parts = [part.strip() for part in payload.split("|", 2)]
    if len(parts) != 3:
        parts = payload.split(maxsplit=2)
    if len(parts) != 3 or not all(parts):
        bot.reply_to(message, "Format: /logerror <subject> <topic> <mistake details>")
        return
    log_error(message.chat.id, parts[0], parts[1], parts[2])
    bot.reply_to(message, f"Saved: {normalize_subject(parts[0])} — {parts[1]}")

@bot.message_handler(commands=["errors"])
def handle_errors(message: telebot.types.Message) -> None:
    register_user(message)
    bot.reply_to(message, format_errors(message.chat.id), parse_mode="HTML")

@bot.message_handler(commands=["audit"])
def handle_audit(message: telebot.types.Message) -> None:
    register_user(message)
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s+(.+)$", message.text.partition(" ")[2])
    if not match:
        bot.reply_to(message, "Format: /audit <score> <study report>")
        return
    save_audit(message.chat.id, float(match.group(1)), match.group(2))
    bot.reply_to(message, "Daily audit saved. Kal aur disciplined preparation.")

@bot.message_handler(commands=["report"])
def handle_report(message: telebot.types.Message) -> None:
    register_user(message)
    bot.reply_to(message, format_report(message.chat.id), parse_mode="HTML")

# -------------------------------- Messages -------------------------------- #

@bot.message_handler(content_types=["photo"])
def handle_photo(message: telebot.types.Message) -> None:
    register_user(message)
    try:
        photo = message.photo[-1]
        file_info = bot.get_file(photo.file_id)
        image_bytes = bot.download_file(file_info.file_path)
        caption = (message.caption or "").strip()
        user_text = f"Screenshot question. Student note: {caption}".strip()
        reply = generate_reply(
            message.chat.id,
            user_text,
            image_bytes=image_bytes,
            mime_type="image/jpeg",
            instruction=SSC_VISION_INSTRUCTION + (f"\nStudent note: {caption}" if caption else ""),
        )
        send_ai_reply(message, reply)
    except Exception as exc:
        logger.exception("Photo vision request failed")
        bot.reply_to(message, f"Vision Error: {exc!r}")

@bot.message_handler(content_types=["text"])
def handle_text(message: telebot.types.Message) -> None:
    register_user(message)
    try:
        reply = generate_reply(message.chat.id, message.text or "")
        send_ai_reply(message, reply)
    except Exception as exc:
        logger.exception("Text Gemini request failed")
        bot.reply_to(message, f"REAL TIME ERROR: {exc!r}")

if __name__ == "__main__":
    logger.info("Starting Telegram bot with Gemini model %s", GEMINI_MODEL)
    scheduler = start_scheduler()
    try:
        bot.infinity_polling(skip_pending=True)
    finally:
        scheduler.shutdown(wait=False)
