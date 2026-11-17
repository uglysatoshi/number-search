"""Saved prediction images and an offline gallery; source photos remain untouched."""

import hashlib
import math
import re
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for name in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def render_prediction(image, result):
    """Draw the predicted polygon only, with truth shown separately as text."""
    from .geometry import crop_plate

    photo = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    original_size = photo.size
    photo.thumbnail((1000, 650))
    width = 1000
    height = max(300, photo.height)
    canvas = Image.new("RGB", (width, height + 170), "#eef2f6")
    offset = ((width - photo.width) // 2, (height - photo.height) // 2)
    canvas.paste(photo, offset)
    draw = ImageDraw.Draw(canvas)
    correct = result.get("correct")
    color = "#168046" if correct is True else "#c93030" if correct is False else "#1565c0"
    polygon = result.get("polygon")
    crop = None
    if polygon is not None:
        points = np.asarray(polygon, dtype=np.float32)
        scaled = [
            (
                float(x * photo.width / original_size[0] + offset[0]),
                float(y * photo.height / original_size[1] + offset[1]),
            )
            for x, y in points
        ]
        draw.line(scaled + scaled[:1], fill=color, width=4)
        crop = Image.fromarray(
            crop_plate(image, points, margin=result.get("crop_margin", 1.06))
        ).convert("RGB")
        enlarged = crop.resize((320, 64), Image.Resampling.LANCZOS)
        canvas.paste(enlarged, (660, height + 70))
    draw.rectangle((0, height, width, height + 6), fill=color)
    label = Path(result.get("file", "Изображение")).name
    draw.text((20, height + 15), label, font=font(20), fill="#283444")
    draw.text(
        (20, height + 47),
        "Ответ: " + (result.get("text") or "номер не прочитан"),
        font=font(27),
        fill=color,
    )
    if "expected" in result:
        draw.text((20, height + 88), "Эталон: " + result["expected"], font=font(25), fill="#283444")
        status = "Полное совпадение" if correct else "Ошибка распознавания"
        draw.text((20, height + 127), status, font=font(21), fill=color)
    if crop is not None:
        draw.text((660, height + 40), "Область, выбранная моделью", font=font(18), fill="#283444")
    return canvas, crop


def save_prediction(image, result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = Path(result["file"]).name
    if "id" in result:
        identifier = str(result["id"])
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", identifier)[:50]
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:12]
        name = f"{safe}_{digest}__{name[:120]}"
    canvas, crop = render_prediction(image, result)
    output = directory / (name + ".result.jpg")
    canvas.save(output, quality=94)
    links = {"image": str(output)}
    if crop is not None:
        crop_path = directory / (name + ".crop.png")
        crop.save(crop_path)
        links["crop"] = str(crop_path)
    return links


def save_gallery(results, directory, accuracy, cer):
    directory = Path(directory)
    cards = []
    thumbnails = []
    for result in results:
        path = Path(result["visualization"]["image"])
        status = "ok" if result["correct"] else "error"
        cards.append(
            f'<article class="{status}"><a href="{escape(path.name, quote=True)}">'
            f'<img src="{escape(path.name, quote=True)}" alt="{escape(Path(result["file"]).name, quote=True)}">'
            "</a></article>"
        )
        with Image.open(path) as image:
            thumbnail = image.copy()
        thumbnail.thumbnail((400, 360))
        thumbnails.append(thumbnail)
    cell_height = max(im.height for im in thumbnails) + 12
    sheet = Image.new("RGB", (1200, 76 + math.ceil(len(thumbnails) / 3) * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    heading = f"Полное совпадение: {accuracy:.0%} ({sum(r['correct'] for r in results)}/{len(results)}) · CER: {cer:.2%}"
    draw.text((20, 20), heading, font=font(27), fill="#283444")
    for i, thumbnail in enumerate(thumbnails):
        sheet.paste(thumbnail, (i % 3 * 400, 76 + i // 3 * cell_height))
    sheet.save(directory / "summary.jpg", quality=95)
    page = (
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Результаты распознавания номеров</title><style>"
        "body{font:18px system-ui;margin:24px;background:#eef2f6;color:#283444}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}"
        "article{border:3px solid;border-radius:8px;overflow:hidden;background:white}"
        ".ok{border-color:#168046}.error{border-color:#c93030}img{display:block;width:100%}"
        "</style><h1>Результаты распознавания</h1><p>" + escape(heading) + "</p>"
        "<p>Зелёный — полное совпадение, красный — ошибка. Контур предсказан моделью; "
        "эталонный текст взят из проверочного набора после предсказания. Нажмите на изображение для увеличения.</p>"
        '<p><a href="summary.jpg">Общая таблица изображений</a></p><main>'
        + "".join(cards)
        + "</main></html>"
    )
    (directory / "index.html").write_text(page, encoding="utf-8")
    return {
        "directory": str(directory),
        "gallery": str(directory / "index.html"),
        "summary": str(directory / "summary.jpg"),
    }
