// Synthetic browser records are created only in an OS temporary runtime.
const fs = require('fs');
const os = require('os');
const path = require('path');
const {pathToFileURL} = require('url');
const {spawnSync} = require('child_process');
const {chromium} = require('playwright');
const assert = require('assert/strict');

(async () => {
  const protocol = path.resolve(__dirname, '..');
  const sourceRuntime = path.join(protocol, 'runtime');
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'gt-v2-dryrun-'));
  const runtime = path.join(tmp, 'runtime');
  const stem = 'adjudication/semantic_first_20260906';
  for (const rel of ['adjudication/method_blind_20260905','admin/semantic_first_20260906', `${stem}/semantic`]) {
    fs.cpSync(path.join(sourceRuntime,rel),path.join(runtime,rel),{recursive:true});
  }
  fs.copyFileSync(path.join(sourceRuntime,'admin/blind_id_admin_mapping.csv'),path.join(runtime,'admin/blind_id_admin_mapping.csv'));
  const semanticDir = path.join(runtime,stem,'semantic');
  const browser = await chromium.launch({headless:true,executablePath:process.env.EDGE_PATH || 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'});
  const context = await browser.newContext({acceptDownloads:true,viewport:{width:1440,height:1100}});
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror',error=>errors.push(String(error)));
  page.on('dialog',dialog=>dialog.accept());
  await page.goto(pathToFileURL(path.join(semanticDir,'index.html')).href);
  await page.waitForFunction(()=>document.querySelector('canvas')?.width>100);
  assert.equal(await page.locator('canvas').count(),1);
  assert.equal(await page.locator('.leaf').count(),0);
  assert.equal(await page.evaluate(()=>window.REVIEW_DATA.items.some(i=>'sets' in i || 'trace_count' in i || 'reason_categories' in i)),false);
  const initialShot = path.join(tmp,'semantic_initial.png');
  await page.screenshot({path:initialShot,fullPage:true});
  await page.click('#submit');
  assert.match(await page.locator('#message').textContent(),/确认/);
  for (let n=0;n<13;n++) {
    await page.click('#add');
    const viewport=page.locator('.viewport').first();
    await page.waitForFunction(()=>document.querySelector('.viewport')?.dataset.ready==='true');
    const vb=await viewport.boundingBox();
    await viewport.click({position:{x:vb.width/2,y:vb.height/2}});
    const selectors=page.locator('.leaf').last().locator('select');
    await selectors.nth(0).selectOption('measurable');
    await selectors.nth(1).selectOption(n===0?'cropped_tip':'none');
    await selectors.nth(2).selectOption('high');
    await page.check('#reviewComplete');
    if(n===0){
      await page.click('#submit');
      assert.match(await page.locator('#message').textContent(),/裁剪/);
      await selectors.nth(0).selectOption('visible_unmeasurable');
      await page.locator('.leaf textarea').fill('SYNTHETIC TEST: cropped endpoint; no complete geometry.');
    }
    await page.click('#submit');
    assert.match(await page.locator('#status').textContent(),/已提交/,await page.locator('#message').textContent());
    if(n===0){
      await page.click('#revise');
      await page.locator('#notes').fill('SYNTHETIC TEST revision.');
      await page.click('#submit');
      await page.reload();
      assert.match(await page.locator('#status').textContent(),/第 2 版/);
      await viewport.dispatchEvent('wheel',{deltaY:-100,clientX:600,clientY:300});
      await page.click('#backup');
      const downloadPromise=page.waitForEvent('download');
      await page.locator('#downloads a').click();
      const backup=await downloadPromise;
      const backupPath=path.join(tmp,backup.suggestedFilename());await backup.saveAs(backupPath);
      await page.locator('#restoreFile').setInputFiles(backupPath);
      await page.waitForFunction(()=>document.querySelector('#message').textContent.includes('进度已恢复'));
    }
    if(n<12) await page.click('#next');
  }
  async function exportsInto(dir){
    await page.click('#export');
    assert.equal(await page.locator('#downloads a').count(),2);
    for(const a of await page.locator('#downloads a').all()){
      const pending=page.waitForEvent('download');await a.click();const d=await pending;await d.saveAs(path.join(dir,d.suggestedFilename()));
    }
  }
  await exportsInto(tmp);
  const semPath=path.join(tmp,'semantic_decisions.json');
  const exported=JSON.parse(fs.readFileSync(semPath,'utf8'));
  assert.equal(exported.records.length,13);
  assert.equal(exported.records[0].history.length,1);
  assert.equal(exported.records[0].history[0].revision,1);
  const run=spawnSync(process.env.PYTHON_EXE||'python',[path.join(__dirname,'build.py'),'geometry','--runtime',runtime,'--semantic-export',semPath],{encoding:'utf8'});
  if(run.status!==0)throw Error(run.stdout+run.stderr);
  const geometryDir=path.join(runtime,stem,'geometry');
  await page.goto(pathToFileURL(path.join(geometryDir,'index.html')).href);
  await page.waitForFunction(()=>document.querySelectorAll('canvas').length===4);
  assert.equal(await page.locator('#add').isVisible(),false);
  const geometryShot=path.join(tmp,'geometry_initial.png');
  await page.screenshot({path:geometryShot,fullPage:true});
  for(let n=0;n<13;n++){
    const selectors=page.locator('.leaf').first().locator('select');
    await selectors.nth(0).selectOption(n===0?'no_geometry':'A:T01');
    await selectors.nth(1).selectOption('high');
    await page.check('#reviewComplete');await page.click('#submit');
    assert.match(await page.locator('#status').textContent(),/已提交/);
    if(n<12)await page.click('#next');
  }
  await exportsInto(tmp);
  const geometry = JSON.parse(fs.readFileSync(path.join(tmp,'geometry_decisions.json'),'utf8'));
  assert.equal(geometry.records.length,13);
  for(const r of geometry.records){const before=exported.records.find(s=>s.blind_id===r.blind_id);for(const [n,l] of before.leaves.entries())for(const [k,v] of Object.entries(l))assert.deepEqual(r.leaves[n][k],v);}
  await browser.close();assert.deepEqual(errors,[]);
  const report={status:'pass',synthetic_only:true,temp_directory:tmp,semantic_initial_zero_structures:true,semantic_candidate_data_absent:true,semantic_submissions:13,geometry_submissions:13,cropped_measurable_rejected:true,revision_history_preserved:true,backup_restore_verified:true,csv_json_downloads:4,semantic_decisions_preserved_in_geometry:true,browser_errors:errors.length,semantic_screenshot:initialShot,geometry_screenshot:geometryShot};
  fs.writeFileSync(path.join(tmp,'dry_run_report.json'),JSON.stringify(report,null,2));
  process.stdout.write(JSON.stringify(report,null,2));
})().catch(error=>{console.error(error);process.exit(1);});
