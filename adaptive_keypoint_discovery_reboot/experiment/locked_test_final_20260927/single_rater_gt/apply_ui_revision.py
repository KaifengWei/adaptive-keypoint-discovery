"""Apply an accepted UI-only revision to the SAME formal package, retaining history.

Does not rebuild the manifest, images, private mapping, or any GT. Does not access
the user's browser profile. Only public HTML/JS bytes are revised.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from build_package import FORMAL, AMENDMENT
from package_tools import HERE, sha, require, code_hashes, inventory, audit_public_package


def apply(synthetic_path,revision_path):
    synthetic=json.loads(synthetic_path.read_text(encoding="utf-8"))
    revision=json.loads(revision_path.read_text(encoding="utf-8"))
    for report in (synthetic,revision):
        require(report["status"]=="PASS" and report["synthetic_only"] is True,"Synthetic acceptance absent")
        require(report["code_sha256"]==code_hashes(),"Code changed after UI acceptance")
    require(synthetic["amendment_sha256"]==sha(AMENDMENT),"Amendment changed")
    public=FORMAL/"public";admin=FORMAL/"admin"
    audit_public_package(public,40)
    before=inventory(public)
    old_ledger=json.loads((admin/"SHA256_LEDGER.json").read_text(encoding="utf-8"))
    require(before["package_sha256"]==old_ledger["package_sha256"],"Public package has unregistered changes")
    destination=admin/"ui_revision_01"
    require(not destination.exists(),"This UI revision has already been applied")
    manifest_path=public/"measurement_manifest.json"
    manifest_sha=sha(manifest_path)
    mapping_sha=sha(admin/"mapping.json")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    template=(HERE/"index.html").read_text(encoding="utf-8")
    html=template.replace("__MANIFEST__",json.dumps(manifest,separators=(",",":"))).replace("__MANIFEST_SHA__",manifest_sha)
    # Preserve the exact previous public code and administrative audit bytes.
    old=destination/"previous";old.mkdir(parents=True)
    for name in ("index.html","app.js"):shutil.copyfile(public/name,old/name)
    for name in ("SHA256_LEDGER.json","synthetic_browser_acceptance.json","formal_asset_browser_audit.json"):
        shutil.copyfile(admin/name,old/name)
    (public/"index.html").write_text(html,encoding="utf-8",newline="\n")
    shutil.copyfile(HERE/"app.js",public/"app.js")
    after=inventory(public)
    unchanged=lambda data:{r["path"]:(r["bytes"],r["sha256"]) for r in data["files"] if r["path"] not in {"index.html","app.js"}}
    require(unchanged(before)==unchanged(after),"UI update changed assets/manifest")
    require(sha(manifest_path)==manifest_sha and sha(admin/"mapping.json")==mapping_sha,"Frozen identities changed")
    audit=audit_public_package(public,40)
    commit=subprocess.run(["git","-C",str(HERE.parents[3]),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
    record={"schema_version":"blind-package-ui-revision-v1","revision":1,"reason":"Width-only fit clipped portrait images; tracing prerequisites lacked visible guidance",
            "implementation_git":commit,"previous_package_sha256":before["package_sha256"],"package_sha256":after["package_sha256"],
            "measurement_manifest_sha256":manifest_sha,"private_mapping_sha256":mapping_sha,"code_sha256":code_hashes(),
            "image_assets_unchanged":40,"ID_order_unchanged":True,"localStorage_key_unchanged":True,"raw_schema_unchanged":True,
            "test_model_reads":0,"inference_gate":"CLOSED","public_asset_audit":audit}
    (destination/"revision_record.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    shutil.copyfile(synthetic_path,destination/"synthetic_browser_acceptance.json")
    shutil.copyfile(revision_path,destination/"ui_revision_acceptance.json")
    # Advance current inventory, preserving the complete previous ledger above.
    ledger={**old_ledger,**after,"previous_package_sha256":before["package_sha256"],"ui_revision":1,"code_sha256":code_hashes(),"asset_audit":audit}
    (admin/"SHA256_LEDGER.json").write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(record,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-report",type=Path,required=True);parser.add_argument("--ui-revision-report",type=Path,required=True)
    args=parser.parse_args();apply(args.synthetic_report.resolve(),args.ui_revision_report.resolve())
