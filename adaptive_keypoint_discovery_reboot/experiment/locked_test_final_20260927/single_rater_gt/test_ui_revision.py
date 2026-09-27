"""Synthetic regression for full-image fitting and explicit tracing prerequisites.

Never opens real test images or an actual user browser profile.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from build_package import build_public
from browser_dry_run import EDGE, click_xy, current, snapshot, wait_locked, export_files
from package_tools import HERE, sha, require, code_hashes, validate_exports, inventory


def compatibility(browser, root, old_fixture):
    """Use genuine old-version SYNTHETIC snapshots, never a user profile/GT."""
    destination=root/"compatibility_public"
    shutil.copytree(old_fixture/"public",destination)
    manifest_path=destination/"measurement_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    before=inventory(destination)
    ctx=browser.new_context(accept_downloads=True,viewport={"width":1450,"height":1000})
    page=ctx.new_page();errors=[];page.on("pageerror",lambda e:errors.append(str(e)))
    page.goto((destination/"index.html").as_uri())
    page.locator("#restore-file").set_input_files(str(old_fixture/"complete_backup.json"))
    page.wait_for_function("document.getElementById('message').textContent.includes('恢复成功')")
    saved=snapshot(page)
    export_files(page,root/"compat_old_exports")
    template=(HERE/"index.html").read_text(encoding="utf-8")
    html=template.replace("__MANIFEST__",json.dumps(manifest,separators=(",",":"))).replace("__MANIFEST_SHA__",sha(manifest_path))
    (destination/"index.html").write_text(html,encoding="utf-8",newline="\n")
    shutil.copyfile(HERE/"app.js",destination/"app.js")
    page.reload();page.wait_for_function("document.getElementById('lock').textContent.startsWith('已提交锁定')")
    require(snapshot(page)==saved,"UI revision changed old local records")
    export_files(page,root/"compat_new_exports")
    validate_exports(root/"compat_new_exports",destination)
    require(sha(root/"compat_old_exports/raw_annotations.json")==sha(root/"compat_new_exports/raw_annotations.json"),"UI revision changed old raw snapshots/export bytes")
    old_assets={r["path"]:r["sha256"] for r in before["files"] if r["path"] not in {"app.js","index.html"}}
    new_assets={r["path"]:r["sha256"] for r in inventory(destination)["files"] if r["path"] not in {"app.js","index.html"}}
    require(old_assets==new_assets,"UI revision changed synthetic identity/assets")
    require(not errors,"Compatibility browser errors: "+repr(errors));ctx.close()


def fit_check(page):
    b=page.locator("#stage").bounding_box()
    v=page.locator("#viewport").bounding_box()
    require(b["width"] <= v["width"]-1 and b["height"] <= v["height"]-1,"Full image does not fit both dimensions")
    require(b["x"]>=v["x"]-.1 and b["y"]>=v["y"]-.1 and b["x"]+b["width"]<=v["x"]+v["width"]+.1 and b["y"]+b["height"]<=v["y"]+v["height"]+.1,"Full image is clipped")
    scroll=page.locator("#viewport").evaluate("e=>[e.scrollLeft,e.scrollTop]")
    require(scroll==[0,0],"Fit/change-image retained scroll position")


def run(root,old_fixture):
    require(not root.exists(),"Synthetic revision fixture destination already exists")
    src=root/"source";src.mkdir(parents=True)
    rows=[]
    for name,size in (("wide",(3400,400)),("portrait",(400,3400))):
        im=Image.new("RGB",size,"white");d=ImageDraw.Draw(im)
        d.line([(size[0]*.2,size[1]*.2),(size[0]*.8,size[1]*.8)],fill="green",width=5)
        p=src/(name+".png");im.save(p)
        rows.append({"path":p,"expected_sha256":sha(p),"fixture":True})
    build_public(root/"public",rows)
    errors=[];checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=EDGE,headless=True)
        ctx=browser.new_context(accept_downloads=True,viewport={"width":1450,"height":1000})
        page=ctx.new_page();page.on("pageerror",lambda e:errors.append(str(e)))
        page.on("console",lambda e:errors.append(e.text) if e.type=="error" else None)
        page.goto((root/"public/index.html").as_uri())
        page.wait_for_function("document.getElementById('photo').naturalWidth>0")
        for n in range(2):
            fit_check(page)
            s=page.evaluate("JSON.parse(document.getElementById('manifest').textContent).samples[JSON.parse(localStorage.getItem('blind-annotation-v1-'+window.MANIFEST_SHA)).position]")
            checks.append("full_image_fit_"+("portrait" if s["height"]>s["width"] else "wide"))
            page.locator("#trace").click()
            require(not current(page)["items"] and current(page)["base_xy"] is None,"Tracing prompt invented annotations")
            require("基点" in page.locator("#message").inner_text(),"Missing base prerequisite prompt")
            base=[s["width"]*.8,s["height"]*.8]
            click_xy(page,*base)
            page.locator("#trace").click()
            require("添加结构" in page.locator("#message").inner_text(),"Missing add-structure prompt")
            page.locator("#add").click()
            require(current(page)["items"][0]["visibility_status"]=="","State prefilled")
            page.locator("#trace").click()
            require("尚未选择状态" in page.locator("#message").inner_text(),"Missing state prompt")
            require(page.locator("#state").evaluate("e=>document.activeElement===e"),"State selector not focused")
            msg=page.locator("#message").bounding_box();vp=page.locator("#viewport").bounding_box()
            require(msg["y"]+msg["height"]<=vp["y"]+1,"Prompt remains below viewport")
            require(not current(page)["items"][0]["points_px"],"Blocked trace wrote geometry")
            page.locator("#state").select_option("measurable")
            require("当前：描迹" in page.locator("#mode").inner_text(),"Explicit measurable selection did not open tracing")
            tip=[s["width"]*.2,s["height"]*.2]
            bounds=page.locator("#overlay").bounding_box()
            page.locator("#overlay").evaluate("e=>e.addEventListener('pointerdown',x=>window.lastSyntheticClick=[x.clientX,x.clientY],{capture:true,once:true})")
            click_xy(page,*tip)
            r=current(page);t=r["items"][0]
            cx,cy=page.evaluate("window.lastSyntheticClick")
            expected=[(cx-bounds["x"])*s["width"]/bounds["width"],(cy-bounds["y"])*s["height"]/bounds["height"]]
            require(len(t["points_px"])==2 and all(abs(a-b)<1e-6 for a,b in zip(t["tip_xy"],expected)),"Raw coordinates differ from native-image inverse mapping")
            page.locator("#confidence").select_option("high")
            page.locator("#checked").check();page.locator("#submit").click();wait_locked(page)
            before=json.dumps(current(page),sort_keys=True)
            for _ in range(10):page.locator("#zoom-in").click()
            page.locator("#viewport").evaluate("e=>{e.scrollLeft=1000;e.scrollTop=1000}")
            page.locator("#fit").click();fit_check(page)
            require(before==json.dumps(current(page),sort_keys=True),"Fit/zoom mutated frozen record")
            if n==0:
                page.locator("#next").click()
                page.wait_for_function("document.getElementById('photo').complete&&document.getElementById('photo').naturalWidth>0")
        checks += ["missing_prerequisites_visible", "no_auto_existence_or_measurability", "explicit_measurable_starts_trace", "raw_pixel_coordinates_unchanged", "zoom_fit_do_not_mutate_records", "page_change_resets_zoom_scroll"]
        export_files(page,root/"exports");validate_exports(root/"exports",root/"public")
        saved=snapshot(page)
        # Resize must also fit both axes without touching existing submissions.
        page.set_viewport_size({"width":1100,"height":850})
        page.wait_for_timeout(100);fit_check(page)
        require(saved==snapshot(page),"Resize modified annotations")
        checks += ["export_sha_schema_unchanged", "resize_fit_no_record_changes"]
        require(not errors,"Browser errors: "+repr(errors))
        compatibility(browser,root,old_fixture)
        checks += ["old_localStorage_and_revision_chains_unchanged", "old_raw_JSON_export_byte_identical", "same_manifest_images_ID_order_and_storage_key"]
        browser.close()
    report={"status":"PASS","synthetic_only":True,"checks":checks,"code_sha256":code_hashes(),"browser_console_errors":errors,"real_test_image_reads":0,"test_model_reads":0,"inference_gate":"CLOSED"}
    (root/"ui_revision_acceptance.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","checks":len(checks),"report":str(root/"ui_revision_acceptance.json")},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,required=True);parser.add_argument("--old-synthetic-fixture",type=Path,required=True)
    args=parser.parse_args();run(args.root.resolve(),args.old_synthetic_fixture.resolve())
