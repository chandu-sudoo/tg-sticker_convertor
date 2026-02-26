#!/usr/bin/env python3
"""
Telegram Sticker/Emoji Converter Bot
=====================================
Converts any video or GIF into a Telegram-ready sticker or emoji:

Sticker rules:
  • One side must be exactly 512 px; other side ≤ 512 px
  • WEBM/VP9, no audio, ≤ 3 s, ≤ 256 KB, ≤ 30 fps, looped

Emoji rules:
  • Exactly 100×100 px
  • Same codec/format constraints

Background removal is supported via rembg (AI-based).

Usage
-----
  export BOT_TOKEN="123456:ABC-DEF..."
  python bot.py

Dependencies
------------
  pip install "python-telegram-bot[all]" rembg onnxruntime pillow
  # System: ffmpeg  (sudo apt install ffmpeg  /  brew install ffmpeg)
"""

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN: str = os.environ.get("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")


# ═══════════════════════════ VIDEO UTILITIES ═════════════════════════════════


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    logger.info("CMD: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def probe(path: str) -> dict:
    """Return basic stream info for the first video stream."""
    result = _run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1",
        path,
    ])
    info: dict = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    if "r_frame_rate" in info:
        n, d = info["r_frame_rate"].split("/")
        info["fps"] = float(n) / float(d)
    if "duration" in info:
        info["duration"] = float(info["duration"])
    for k in ("width", "height"):
        if k in info:
            info[k] = int(info[k])
    return info


def _scale_filter(tw: int, th: int) -> str:
    """
    FFmpeg filter: scale preserving aspect ratio → pad to tw×th with
    transparent (black) pixels so the canvas is exactly tw×th.
    """
    return (
        f"scale=w={tw}:h={th}:force_original_aspect_ratio=decrease,"
        f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black@0"
    )


def extract_frames(src: str, frames_dir: Path, fps: float, duration: float, tw: int, th: int):
    frames_dir.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-t", str(duration),
        "-vf", f"fps={fps},{_scale_filter(tw, th)}",
        "-pix_fmt", "rgba",
        str(frames_dir / "f%04d.png"),
    ])


def frames_to_webm(frames_dir: Path, out: str, fps: float, crf: int = 33):
    """Encode PNG frames (with alpha) → VP9 WEBM, no audio, looped."""
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / "f%04d.png"),
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",   # preserve alpha
        "-b:v", "0",
        "-crf", str(crf),
        "-an",                    # no audio
        "-loop", "0",             # infinite loop in WEBM
        out,
    ])


# ═══════════════════════════ BACKGROUND REMOVAL ══════════════════════════════


def remove_bg_frames(src_dir: Path, dst_dir: Path):
    """Remove background from every frame using rembg."""
    try:
        from rembg import remove  # type: ignore
    except ImportError:
        raise RuntimeError(
            "rembg is not installed.\n"
            "Run: pip install rembg onnxruntime"
        )

    dst_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(src_dir.glob("f*.png"))
    logger.info("Removing background from %d frame(s)…", len(frames))
    for i, fp in enumerate(frames, 1):
        data = remove(fp.read_bytes())
        (dst_dir / fp.name).write_bytes(data)
        if i % 15 == 0:
            logger.info("  %d / %d done", i, len(frames))


# ═══════════════════════════ MAIN CONVERSION ═════════════════════════════════


def convert(
    src: str,
    dst: str,
    mode: str,        # "sticker" | "emoji"
    remove_bg: bool,
) -> float:
    """
    Full pipeline.  Returns output file size in KB.
    mode="sticker"  → 512×512, ≤256 KB
    mode="emoji"    → 100×100, ≤256 KB
    """
    tw = th = 512 if mode == "sticker" else 100
    max_kb   = 256
    max_dur  = 3.0
    max_fps  = 30.0

    info = probe(src)
    duration = min(info.get("duration", 3.0), max_dur)
    fps      = min(info.get("fps", 24.0), max_fps)
    fps      = round(fps, 3)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path  = Path(tmp)
        raw_dir   = tmp_path / "raw"
        clean_dir = tmp_path / "clean"
        webm_tmp  = str(tmp_path / "out.webm")

        # 1. Extract frames
        extract_frames(src, raw_dir, fps, duration, tw, th)

        # 2. (Optional) background removal
        frames_dir = raw_dir
        if remove_bg:
            remove_bg_frames(raw_dir, clean_dir)
            frames_dir = clean_dir

        # 3. Encode with auto quality adjustment to hit ≤256 KB
        crf = 33
        for attempt in range(4):
            frames_to_webm(frames_dir, webm_tmp, fps, crf)
            size_kb = Path(webm_tmp).stat().st_size / 1024
            logger.info("Encode attempt %d: %.1f KB (crf=%d)", attempt + 1, size_kb, crf)
            if size_kb <= max_kb:
                break
            crf = min(crf + 10, 63)

        shutil.copy2(webm_tmp, dst)
        return Path(dst).stat().st_size / 1024


# ═══════════════════════════ BOT HANDLERS ════════════════════════════════════


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Telegram Sticker / Emoji Bot*\n\n"
        "Send me any *video* or *GIF* and I'll convert it to a "
        "Telegram‑ready sticker or emoji (WEBM/VP9, transparent background).\n\n"
        "Use /help for details.",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *How to use*\n\n"
        "1. Send a video or GIF (up to 20 MB).\n"
        "2. Pick a format:\n"
        "   • *Sticker* — 512×512 px\n"
        "   • *Emoji*   — 100×100 px\n"
        "3. Choose whether to remove the background (AI‑powered).\n"
        "4. Receive your `.webm` file!\n\n"
        "📐 *Output specs*\n"
        "• Codec: VP9 (WEBM)\n"
        "• Duration: ≤ 3 s (auto‑trimmed)\n"
        "• FPS: ≤ 30\n"
        "• File size: ≤ 256 KB\n"
        "• Audio: removed\n"
        "• Loop: ✅\n"
        "• Alpha/transparency: ✅ (when BG removed)",
        parse_mode="Markdown",
    )


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if msg.video:
        fobj = msg.video
    elif msg.animation:
        fobj = msg.animation
    elif msg.document and (msg.document.mime_type or "").startswith("video"):
        fobj = msg.document
    else:
        await msg.reply_text("⚠️ Please send a *video* or *GIF*.", parse_mode="Markdown")
        return

    context.user_data["file_id"] = fobj.file_id

    kb = [
        [
            InlineKeyboardButton("🖼 Sticker (512px)",          callback_data="sticker"),
            InlineKeyboardButton("😀 Emoji (100px)",            callback_data="emoji"),
        ],
        [
            InlineKeyboardButton("🖼 Sticker + Remove BG 🪄",  callback_data="sticker_nobg"),
            InlineKeyboardButton("😀 Emoji + Remove BG 🪄",    callback_data="emoji_nobg"),
        ],
    ]
    await msg.reply_text("✅ Got it! Choose output format:", reply_markup=InlineKeyboardMarkup(kb))


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    action    = query.data
    file_id   = context.user_data.get("file_id")
    if not file_id:
        await query.edit_message_text("❌ Session expired — please resend your file.")
        return

    remove_bg = "_nobg" in action
    mode      = "emoji" if action.startswith("emoji") else "sticker"
    size_px   = "100×100" if mode == "emoji" else "512×512"
    bg_note   = " + background removed 🪄" if remove_bg else ""

    await query.edit_message_text(f"⏳ Converting to {mode}{bg_note}… (up to ~60 s)")

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "input.mp4")
        dst = os.path.join(tmp, f"{mode}.webm")

        # Download
        try:
            tg_file = await context.bot.get_file(file_id)
            await tg_file.download_to_drive(src)
        except Exception as exc:
            await query.edit_message_text(f"❌ Download failed: {exc}")
            return

        # Convert (blocking → run in thread pool)
        try:
            loop = asyncio.get_event_loop()
            size_kb = await loop.run_in_executor(
                None, lambda: convert(src, dst, mode, remove_bg)
            )
        except Exception as exc:
            logger.exception("Conversion error")
            await query.edit_message_text(f"❌ Conversion failed:\n`{exc}`", parse_mode="Markdown")
            return

        # Send
        caption = (
            f"✅ *{mode.capitalize()}* ready!\n"
            f"📐 {size_px} px  |  🗜 {size_kb:.1f} KB  |  VP9 WEBM"
            + (f"\n🪄 Background removed" if remove_bg else "")
        )
        try:
            with open(dst, "rb") as f:
                await context.bot.send_document(
                    chat_id=query.message.chat_id,
                    document=f,
                    filename=f"telegram_{mode}.webm",
                    caption=caption,
                    parse_mode="Markdown",
                )
            await query.edit_message_text(f"✅ {mode.capitalize()} sent ⬇️")
        except Exception as exc:
            await query.edit_message_text(f"❌ Could not send file: {exc}")


# ═══════════════════════════ ENTRY POINT ═════════════════════════════════════


def main():
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        raise SystemExit(
            "❌  Bot token not set.\n"
            "    export BOT_TOKEN='123456:ABC-DEF...'\n"
            "    python bot.py"
        )

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(
        MessageHandler(
            filters.VIDEO | filters.ANIMATION | filters.Document.VIDEO,
            handle_media,
        )
    )
    app.add_handler(CallbackQueryHandler(handle_callback))

    logger.info("Bot is running…  Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()