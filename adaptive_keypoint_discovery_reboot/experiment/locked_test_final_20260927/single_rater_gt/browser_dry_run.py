"""Edge acceptance on synthetic fixtures; optional real-package asset-only opening.

The asset-only mode never clicks an annotation tool or inspects image content.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from playwright.sync_api import sync_playwright
from package_tools import HERE, sha, require, code_hashes, inventory, audit_public_package, validate_exports, digest_text
from build_package import AMENDMENT

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def snapshot(page):
    return page.evaluate("JSON.parse(localStorage.getItem('blind-annotation-v1-'+window.MANIFEST_SHA))")


def current(page):
    state = snapshot(page)
    ident = page.locator("#blind").inner_text()
    return state["records"][ident]


def click_xy(page, x, y):
    b = page.locator("#overlay").bounding_box()
    s = page.evaluate("JSON.parse(document.getElementById('manifest').textContent).samples[JSON.parse(localStorage.getItem('blind-annotation-v1-'+window.MANIFEST_SHA)).position]")
    page.mouse.click(b["x"] + x / s["width"] * b["width"], b["y"] + y / s["height"] * b["height"])


def wait_locked(page):
    page.wait_for_function("document.getElementById('lock').textContent.startsWith('已提交锁定')")


def export_files(page, root):
    page.locator("#export").click()
    page.wait_for_selector("#downloads a[download='SHA256_LEDGER.json']")
    root.mkdir(exist_ok=True)
    for name in ("raw_annotations.json", "sessions.csv", "traces.csv", "SHA256_LEDGER.json"):
        with page.expect_download() as task:
            page.locator(f"#downloads a[download='{name}']").click()
        task.value.save_as(root / name)


def synthetic(root):
    package = root / "public"
    report_path = root / "synthetic_browser_acceptance.json"
    errors = []
    subprocess.run(["node", "--check", str(HERE / "app.js")], check=True, capture_output=True)
    unit = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_package"], cwd=HERE, capture_output=True, text=True)
    require(unit.returncode == 0, unit.stdout + unit.stderr)
    audit_public_package(package, 2)
    require(sha(package / "app.js") == sha(HERE / "app.js"), "Stale fixture script")
    checks = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=EDGE, headless=True)
        context = browser.new_context(accept_downloads=True, viewport={"width": 1450, "height": 1000})
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda e: errors.append(e.text) if e.type == "error" else None)
        page.goto((package / "index.html").as_uri())
        page.wait_for_function("document.getElementById('photo').complete && document.getElementById('photo').naturalWidth>0")
        require(not current(page)["items"], "Initial structure leak")
        checks.append("initial_blank_dynamic_structure_addition")
        page.locator("#base").click(); click_xy(page, 760, 210)
        page.locator("#add").click()
        require(current(page)["items"][0]["visibility_status"] == "", "Existence prefilled")
        page.locator("#state").select_option("measurable")
        page.locator("#confidence").select_option("high")
        page.locator("#trace").click()
        for pt in ((420, 190), (80, 120), (70, 120)): click_xy(page, *pt)
        page.locator("#undo").click()
        require(len(current(page)["items"][0]["points_px"]) == 3, "Undo failed")
        # Real mouse drag on a synthetic control point.
        b = page.locator("#overlay circle[data-point='1']").bounding_box()
        page.mouse.move(b["x"]+b["width"]/2, b["y"]+b["height"]/2)
        page.mouse.down(); page.mouse.move(b["x"]+b["width"]/2+3, b["y"]+b["height"]/2); page.mouse.up()
        checks += ["shared_above_ground_base", "geometry_trace_tip_drag_undo"]
        w = page.locator("#stage").bounding_box()["width"]
        page.locator("#zoom-in").click()
        require(page.locator("#stage").bounding_box()["width"] > w, "Zoom failed")
        page.locator("#fit").click()
        page.locator("#viewport").hover(); page.mouse.wheel(0, -100)
        require(page.locator("#stage").bounding_box()["width"] > w, "Wheel zoom failed")
        page.locator("#fit").click()
        page.locator("#toggle").click()
        require(page.locator("#overlay circle").count() == 0, "Overlay toggle failed")
        page.locator("#toggle").click()
        checks.append("wheel_zoom_fit_overlay_toggle")
        for state in ("visible_unmeasurable", "uncertain", "non_target_structure"):
            page.locator("#add").click(); page.locator("#state").select_option(state)
            page.locator("#confidence").select_option("medium")
            page.locator("#item-note").fill("Synthetic fixture reason; no real-image judgment.")
            page.locator("#tip").click(); click_xy(page, 270, 250)
        require(all(not t["points_px"] for t in current(page)["items"][1:]), "Nonmeasurable geometry fabricated")
        checks.append("all_four_states_nonmeasurable_tip_only")
        # Invalid incomplete form must not submit.
        page.locator("#submit").click()
        require(not current(page)["submitted"], "Incomplete record submitted")
        page.locator("#checked").check()
        with page.expect_download() as task: page.locator("#backup").click()
        draft = root / "draft_backup.json"; task.value.save_as(draft)
        draft_data = json.loads(json.loads(draft.read_text(encoding="utf-8"))["payload_json"])
        require(len(next(iter(draft_data["records"].values()))["items"]) == 4, "Draft lost from backup")
        page.reload();page.wait_for_selector("#items button")
        require(len(current(page)["items"]) == 4, "Refresh lost draft")
        checks += ["schema_incomplete_submit_rejected", "draft_backup", "refresh_recovery"]
        page.locator("#submit").click();wait_locked(page)
        require(page.locator("#add").is_disabled() and page.locator("#state").is_disabled(), "Submission is editable")
        require(len(current(page)["history"]) == 1 and len(current(page)["history"][0]["sha256"]) == 64, "Missing immediate SHA snapshot")
        checks.append("formal_submit_sha_lock")
        page.locator("#technical").click()
        page.locator("#revision-reason").fill("Synthetic only: documented accidental coordinate click.")
        page.locator("#begin-revision").click()
        require(page.locator("#add").is_disabled() and page.locator("#state").is_disabled(), "Revision permits judgment changes")
        page.locator("#items button").first.click()
        page.locator("#trace").click();click_xy(page, 75, 121)
        page.locator("#undo").click()
        page.locator("#submit").click();wait_locked(page)
        require(len(current(page)["history"]) == 2, "Revision overwrote original")
        checks.append("technical_revision_full_hash_chain_and_semantic_lock")
        page.locator("#next").click()
        page.locator("#base").click();click_xy(page,760,210)
        page.locator("#add").click();page.locator("#state").select_option("measurable");page.locator("#confidence").select_option("high")
        page.locator("#occlusion").select_option("minor");page.locator("#interpolation").select_option("yes")
        page.locator("#trace").click();click_xy(page,420,190);click_xy(page,80,130)
        page.locator("#checked").check();page.locator("#submit").click();wait_locked(page)
        export_files(page, root / "synthetic_exports")
        result = validate_exports(root / "synthetic_exports", package)
        require(result["n"] == 2 and result["snapshots"] == 3, "Export lost records/versions")
        checks += ["minor_occlusion_interpolation", "JSON_CSV_file_SHA_consistency", "independent_raw_schema_validation"]
        with page.expect_download() as task: page.locator("#backup").click()
        backup = root / "complete_backup.json";task.value.save_as(backup)
        fresh = browser.new_context(accept_downloads=True, viewport={"width":1450,"height":1000})
        restored = fresh.new_page()
        restored.on("pageerror",lambda e:errors.append(str(e)))
        restored.goto((package / "index.html").as_uri())
        restored.locator("#restore-file").set_input_files(str(backup))
        restored.wait_for_function("document.getElementById('message').textContent.includes('恢复成功')")
        require(current(restored)["submitted"], "Restore lost locked state")
        restored.reload();restored.wait_for_function("document.getElementById('lock').textContent.startsWith('已提交锁定')")
        export_files(restored, root / "restored_exports")
        validate_exports(root / "restored_exports", package)
        require(sha(root / "restored_exports/raw_annotations.json") == sha(root / "synthetic_exports/raw_annotations.json"), "Restore changed raw snapshot bytes")
        checks += ["new_browser_restore", "refresh_locked_recovery", "export_after_restore_byte_identical"]
        restored.locator("#restore-file").set_input_files(str(draft))
        restored.wait_for_function("document.getElementById('message').textContent.includes('不得丢弃')")
        checks.append("restore_cannot_discard_formal_revision")
        invalid = json.loads(backup.read_text(encoding="utf-8"));invalid["sha256"]="0"*64
        bad = root / "invalid_backup.json";bad.write_text(json.dumps(invalid),encoding="utf-8")
        restored.locator("#restore-file").set_input_files(str(bad))
        restored.wait_for_function("document.getElementById('message').textContent.includes('SHA 校验失败')")
        require(current(restored)["submitted"], "Invalid restore mutated state")
        checks.append("corrupt_restore_rejected_without_mutation")
        require(not errors, "Browser errors: " + repr(errors))
        browser.close()
    report = {"status":"PASS", "synthetic_only":True, "fixture_n":2, "browser":"headless Microsoft Edge via Python Playwright",
              "code_sha256":code_hashes(), "amendment_sha256":sha(AMENDMENT), "checks":checks,
              "schema_unit_tests":12, "unit_test_output":unit.stdout+unit.stderr,
              "javascript_syntax":"PASS", "browser_console_errors":errors, "test_model_reads":0,
              "real_test_image_reads":0, "inference_gate":"CLOSED"}
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS", "checks":len(checks), "report":str(report_path)},ensure_ascii=False))


def assets_only(package):
    require(package.name == "public", "Public directory required")
    report = audit_public_package(package, 40)
    before = inventory(package)
    manifest = json.loads((package / "measurement_manifest.json").read_text(encoding="utf-8"))
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=EDGE,headless=True)
        context=browser.new_context(viewport={"width":1450,"height":1000})
        page=context.new_page();page.on("pageerror",lambda e:errors.append(str(e)))
        page.on("console",lambda e:errors.append(e.text) if e.type=="error" else None)
        page.goto((package / "index.html").as_uri())
        for index,s in enumerate(manifest["samples"]):
            page.wait_for_function("([w,h])=>{const x=document.getElementById('photo');return x.complete&&x.naturalWidth===w&&x.naturalHeight===h}",arg=[s["width"],s["height"]])
            require(page.locator("#blind").inner_text()==s["blind_id"],"Page identity differs")
            r=current(page)
            require(not r["items"] and not r["submitted"] and not r["history"] and r["base_xy"] is None,"Real package contains annotations")
            if index<39:page.locator("#next").click()
        require(not errors,"Asset browser errors: "+repr(errors))
        browser.close()
    require(inventory(package)==before,"Asset-only opening mutated package")
    require(not any((package.parent / "results").iterdir()),"Results directory not empty")
    report.update(status="PASS",browser_asset_opening="40/40",browser_console_errors=errors,
                  annotations_created=0,model_reads=0,content_QC=False,inference_gate="CLOSED")
    (package.parent / "admin/formal_asset_browser_audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    g=parser.add_mutually_exclusive_group(required=True);g.add_argument("--synthetic-root",type=Path);g.add_argument("--assets-only",type=Path)
    args=parser.parse_args()
    if args.synthetic_root:synthetic(args.synthetic_root.resolve())
    else:assets_only(args.assets_only.resolve())
