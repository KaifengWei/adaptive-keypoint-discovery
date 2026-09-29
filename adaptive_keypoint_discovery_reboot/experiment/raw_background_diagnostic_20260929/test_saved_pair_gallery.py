"""Offline asset/browser acceptance for the exploratory train-pair gallery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def check(directory: Path) -> None:
    directory = directory.resolve()
    ledger = json.loads((directory / "gallery_manifest.json").read_text(encoding="utf-8"))
    assert ledger["plants"] == 20 and ledger["panels"] == 160
    assert len(ledger["file_sha256"]) == 163
    for relative, expected in ledger["file_sha256"].items():
        assert hashlib.sha256((directory / relative).read_bytes()).hexdigest() == expected, relative
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        )
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.goto((directory / "index.html").as_uri())
        assert page.locator("#sample option").count() == 20
        for index in range(20):
            page.locator("#sample").select_option(str(index))
            assert page.locator(".card").count() == 8
            page.wait_for_function(
                "() => [...document.querySelectorAll('.viewport img')].length === 8 && "
                "[...document.querySelectorAll('.viewport img')].every(img => img.complete && img.naturalWidth > 0 && img.naturalHeight > 0)",
                timeout=30000,
            )
        page.locator("#sample").select_option("0")
        page.locator("#next").click()
        assert "2 / 20" in page.locator("#counter").inner_text()
        page.locator("#zoom").select_option("2")
        assert page.locator(".viewport img").first.evaluate("img => img.style.maxWidth") == "none"
        browser.close()
    assert not errors, errors
    print("train-pair gallery checks: PASS; 20 plants, 160 panels, 0 page/console errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    check(parser.parse_args().directory)
