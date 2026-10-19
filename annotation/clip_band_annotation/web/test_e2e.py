#!/usr/bin/env python3
"""End-to-end test of the object-check page on the synthetic demo bundle (headless Chromium).

Checks: no requests leave the local server, no console/CSP errors, two-step flow,
undo, zoom, resume after reload, export contents, import into a fresh browser.

  python make_demo_bundle.py
  python test_e2e.py [--shots DIR]      # needs: pip install playwright==1.47.0
"""
import argparse
import csv
import functools
import http.server
import io
import threading
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

HERE = Path(__file__).resolve().parent
DEMO = HERE / "demo_bundle"


def serve():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    h = functools.partial(Quiet, directory=str(HERE))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/index.html"


def watch(page, base, errors, requests):
    page.on("console", lambda m: m.type in ("error", "warning") and errors.append(m.text))
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("request", lambda r: requests.append(r.url))


def open_bundle(page):
    page.set_input_files("#folder", str(DEMO))
    expect(page.locator("#confirm")).to_be_visible()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=Path)
    args = ap.parse_args()
    srv, url = serve()
    base = url.rsplit("/", 1)[0]
    errors, requests = [], []
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1280, "height": 900}, accept_downloads=True)
        page = ctx.new_page()
        watch(page, base, errors, requests)
        page.goto(url)
        open_bundle(page)
        expect(page.locator("#who")).to_have_text("DEMO")
        page.click("#go")

        # page 1, crop step: 7 crops, mark two then undo one
        cards = page.locator("#grid .card")
        expect(cards).to_have_count(7)
        crop_order = [cards.nth(i).get_attribute("data-id") for i in range(7)]
        cards.nth(0).click(); cards.nth(1).click(); cards.nth(1).click()
        expect(page.locator("#grid .card.marked")).to_have_count(1)
        if args.shots:
            page.screenshot(path=args.shots / "crop_step.png", full_page=True)
        page.click("#next")

        # frame step: same items, reshuffled, unmarked; zoom; mark two
        expect(page.locator("#grid.scene .card .frame img")).to_have_count(7)
        frame_order = [cards.nth(i).get_attribute("data-id") for i in range(7)]
        assert sorted(frame_order) == sorted(crop_order) and frame_order != crop_order
        expect(page.locator("#grid .card.marked")).to_have_count(0)
        cards.nth(2).locator(".zoom-btn").click()
        expect(page.locator("#zoom")).to_be_visible()
        expect(page.locator("#grid .card.marked")).to_have_count(0)  # zoom does not mark
        page.keyboard.press("Escape")
        expect(page.locator("#zoom")).to_be_hidden()
        cards.nth(3).click(); cards.nth(4).click()
        if args.shots:
            page.screenshot(path=args.shots / "frame_step.png", full_page=True)
        page.click("#next")
        expect(page.locator("#where")).to_have_text("Page 2 / 2")

        # reload: progress kept, resumes on page 2
        page.reload()
        open_bundle(page)
        expect(page.locator("#confirm-text")).to_contain_text("finished 1 page")
        page.click("#go")
        expect(page.locator("#where")).to_have_text("Page 2 / 2")
        expect(cards).to_have_count(6)
        page.click("#next")                       # nothing marked on the crop step
        cards.nth(0).press("Enter")               # keyboard marking
        expect(page.locator("#next")).to_have_text("Finish")
        page.click("#next")
        expect(page.locator("#done")).to_be_visible()

        with page.expect_download() as d:
            page.click("#export-end")
        rows = list(csv.DictReader(io.StringIO(Path(d.value.path()).read_text())))
        assert len(rows) == 26, len(rows)
        marked = {}
        for r in rows:
            marked[(r["page_index"], r["phase"])] = marked.get((r["page_index"], r["phase"]), 0) + int(r["marked_not_object"])
        assert marked == {("0", "crop"): 1, ("0", "frame"): 2, ("1", "crop"): 0, ("1", "frame"): 1}, marked
        assert all(r["step_submitted"] and float(r["step_seconds"]) >= 0 for r in rows)
        assert {r["rater"] for r in rows} == {"DEMO"}
        backup = Path(d.value.path())

        # import into a fresh browser profile
        ctx2 = b.new_context(viewport={"width": 390, "height": 844}, color_scheme="dark")
        p2 = ctx2.new_page()
        watch(p2, base, errors, requests)
        p2.goto(url)
        p2.locator("details summary").click()
        p2.set_input_files("#import", str(backup))
        expect(p2.locator("#start-msg")).to_contain_text("Backup loaded (26 rows)")
        open_bundle(p2)
        expect(p2.locator("#confirm-text")).to_contain_text("finished 2 pages")
        if args.shots:
            p2.screenshot(path=args.shots / "phone_dark_confirm.png", full_page=True)
        b.close()
    srv.shutdown()

    outside = [r for r in requests if not r.startswith(base) and not r.startswith("blob:")]
    assert not outside, f"requests left the page: {outside}"
    assert not errors, f"console errors: {errors}"
    print(f"PASS: {len(requests)} requests (all local files or blob images), 0 console errors")


if __name__ == "__main__":
    main()
