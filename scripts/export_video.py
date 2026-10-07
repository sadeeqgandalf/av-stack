"""Export recorded replays to MP4: plays the viewer headlessly and encodes every frame.

    PYTHONPATH=. .venv/bin/python scripts/export_video.py viewer/urban_vru.html viewer/road_debris.html
    PYTHONPATH=. .venv/bin/python scripts/export_video.py viewer/highway.html --view top --size 1920x1080

Needs: pip install playwright && python -m playwright install chromium, plus ffmpeg on PATH
(or `pip install imageio-ffmpeg`). Frames are recorded at 10 Hz, so --fps 10 is real time.
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
from pathlib import Path

from playwright.async_api import async_playwright


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


async def export(html: Path, out: Path, width: int, height: int, fps: int, view: str, camera: str) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                                "--ignore-gpu-blocklist"])
        page = await browser.new_page(viewport={"width": width, "height": height}, color_scheme="dark")
        await page.goto(html.resolve().as_uri())
        await page.wait_for_function("window.__viewerReady === true", timeout=60000)
        if view != "chase":
            await page.click(f"[data-view={view}]")
        if camera:
            await page.click(f"[data-cam={camera}]")
        n = await page.evaluate("+document.getElementById('scrub').max + 1")
        out.parent.mkdir(parents=True, exist_ok=True)
        enc = subprocess.Popen(
            [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-movflags", "+faststart", str(out)],
            stdin=subprocess.PIPE)
        for i in range(n):
            await page.evaluate(
                "(i) => { const s = document.getElementById('scrub'); s.value = i;"
                " s.dispatchEvent(new Event('input')); }", i)
            await page.evaluate("() => new Promise((r) => requestAnimationFrame(() => r()))")
            enc.stdin.write(await page.screenshot(type="png"))
            if i % 100 == 0:
                print(f"  {html.name}: frame {i}/{n}", flush=True)
        enc.stdin.close()
        enc.wait()
        await browser.close()
    print(f"{out}  ({n} frames, {n / fps:.0f} s at {fps} fps, {out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("videos"))
    ap.add_argument("--size", default="1600x1000")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--view", default="chase", choices=["chase", "top", "orbit"])
    ap.add_argument("--camera", default="", help="camera tab to show, e.g. front_tele")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
    for html in a.html:
        asyncio.run(export(html, a.out_dir / (html.stem + ".mp4"), w, h, a.fps, a.view, a.camera))


if __name__ == "__main__":
    main()
