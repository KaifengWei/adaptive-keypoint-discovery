"""Offline browser/asset acceptance for the saved-prediction visualization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(directory: Path) -> None:
    directory = directory.resolve()
    ledger = json.loads((directory / "gallery_manifest.json").read_text(encoding="utf-8"))
    assert ledger["plants"] == 40 and ledger["methods"] == ["Teacher-direct", "Student-B", "Student-D"]
    assert len(ledger["file_sha256"]) == 164  # 40 x (raw + 3) + 3 quicklooks + HTML
    for relative, expected in ledger["file_sha256"].items():
        assert sha(directory / relative) == expected, relative
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        )
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.goto((directory / "index.html").as_uri())
        assert page.locator("#sample option").count() == 40
        for index in range(40):
            page.locator("#sample").select_option(str(index))
            assert page.locator(".card").count() == 4
            assert page.locator(".viewport img").evaluate_all(
                "images => images.every(img => img.complete && img.naturalWidth > 0 && img.naturalHeight > 0)"
            ), index
        page.locator("#typical").click()
        assert "v4_test_0022" in page.locator("#counter").inner_text()
        page.locator("#zoom").select_option("2")
        assert page.locator(".viewport img").first.evaluate("img => img.style.maxWidth") == "none"
        browser.close()
    assert not errors, errors
    print("gallery browser/asset checks: PASS; 40 samples, 160 panels, 0 page/console errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    check(parser.parse_args().directory)
