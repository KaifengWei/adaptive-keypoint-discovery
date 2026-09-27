"use strict";
(() => {
  const manifest=JSON.parse(document.getElementById("manifest").textContent), samples=manifest.samples;
  const key="blind-annotation-v1-"+window.MANIFEST_SHA, $=id=>document.getElementById(id);
  const states=["measurable","visible_unmeasurable","uncertain","non_target_structure"];
  const clone=o=>JSON.parse(JSON.stringify(o)), uuid=()=>crypto.randomUUID();
  let position=0,selected=-1,mode="browse",zoom=1,show=true,undo=[],drag=null,records={},busy=false;
  const bytes=s=>new TextEncoder().encode(s);
  const sha=async s=>Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",bytes(s)))).map(x=>x.toString(16).padStart(2,"0")).join("");
  function say(text,error=false){$("message").textContent=text;$("message").className=error?"error":"ok";}
  function sample(){return samples[position];}
  function blank(s){return {schema_version:"blind-session-v1",manifest_sha256:window.MANIFEST_SHA,session_id:uuid(),rater_id:"S1",round:1,blind_id:s.blind_id,image_sha256:s.image_sha256,width:s.width,height:s.height,base_xy:null,items:[],image_note:"",whole_plant_checked:false,submitted:false,revision:1,technical_revision:null,history:[]};}
  function rec(){if(!records[sample().blind_id])records[sample().blind_id]=blank(sample());return records[sample().blind_id];}
  function locked(){return rec().submitted||busy;}
  function technical(){return !!rec().technical_revision;}
  function item(){return rec().items[selected];}
  function save(){try{localStorage.setItem(key,JSON.stringify({position,records}));}catch(e){say("本地存储失败，请立即备份进度："+e.message,true);throw e;}}
  function remember(){undo.push(clone(rec()));if(undo.length>100)undo.shift();}
  function pointOK(p,s){return Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)&&p[0]>=0&&p[1]>=0&&p[0]<=s.width&&p[1]<=s.height;}
  function validate(r,s,formal=true){
    if(r.schema_version!=="blind-session-v1"||r.manifest_sha256!==window.MANIFEST_SHA||r.rater_id!=="S1"||r.round!==1||!r.session_id)throw Error("记录 schema/测量轮次非法");
    if(r.blind_id!==s.blind_id||r.image_sha256!==s.image_sha256||r.width!==s.width||r.height!==s.height)throw Error("图像身份不一致");
    if(r.base_xy!==null&&!pointOK(r.base_xy,s))throw Error("共同基点坐标非法");
    if(!Array.isArray(r.items))throw Error("结构列表非法");
    if(!formal)return;
    if(!r.whole_plant_checked)throw Error("请先确认已检查整株");
    if(!r.items.length&&!r.image_note.trim())throw Error("未记录结构时必须解释整株判断");
    const ids=new Set();
    for(const t of r.items){
      if(!t.trace_uuid||ids.has(t.trace_uuid))throw Error("结构身份非法或重复");ids.add(t.trace_uuid);
      if(!states.includes(t.visibility_status))throw Error("每条结构都须选择状态");
      if(!["high","medium","low"].includes(t.confidence))throw Error("每条结构都须选择信心");
      if(!["none","minor","major"].includes(t.occlusion)||!["yes","no"].includes(t.interpolation_used))throw Error("遮挡/插值字段非法");
      if(t.tip_xy!==null&&!pointOK(t.tip_xy,s))throw Error("尖端坐标非法");
      if(t.visibility_status==="measurable"){
        if(!r.base_xy||t.points_px.length<2||!t.points_px.every(p=>pointOK(p,s)))throw Error("可测结构需要共同基点及完整合法描迹");
        if(JSON.stringify(t.points_px[0])!==JSON.stringify(r.base_xy)||JSON.stringify(t.points_px.at(-1))!==JSON.stringify(t.tip_xy))throw Error("路径须从共同基点到记录尖端");
        if(!t.points_px.some(p=>Math.hypot(p[0]-r.base_xy[0],p[1]-r.base_xy[1])>1e-6))throw Error("路径长度为零");
        if(t.occlusion==="major")throw Error("严重遮挡不得声明完整可靠几何");
      }else{
        if(t.points_px.length)throw Error("非可测结构不能导出几何描迹");
        if(!t.note.trim())throw Error("非可测结构须说明原因");
      }
      if(t.interpolation_used==="yes"&&(t.occlusion!=="minor"||t.visibility_status!=="measurable"))throw Error("插值仅用于符合冻结规则的轻微遮挡可测结构");
    }
  }
  async function checkHistory(r,s){
    let prev=null,n=0,first=null;
    for(const h of r.history){
      if(await sha(h.snapshot_json)!==h.sha256)throw Error("正式记录 SHA 校验失败");
      const v=JSON.parse(h.snapshot_json);validate(v,s);n++;
      if(v.revision!==n||v.previous_submission_sha256!==prev)throw Error("修订链断裂");
      if(n===1){if(v.technical_revision!==null)throw Error("首轮不能是技术修订");first=v;}
      else{
        if(!v.technical_revision||!v.technical_revision.reason.trim())throw Error("缺少技术修订原因");
        if(v.items.length!==first.items.length)throw Error("技术修订改变结构数量");
        for(let i=0;i<v.items.length;i++)for(const f of ["trace_uuid","gt_id","visibility_status","occlusion","interpolation_used","confidence","note"]){if(v.items[i][f]!==first.items[i][f])throw Error("技术修订改变冻结判断");}
        if(v.image_note!==first.image_note||v.whole_plant_checked!==first.whole_plant_checked)throw Error("技术修订改变整株判断");
        if(v.session_id!==first.session_id)throw Error("技术修订改变会话身份");
      }prev=h.sha256;
    }
    if(r.submitted){if(!n)throw Error("锁定记录缺少快照");const current=clone(r);delete current.history;delete current.submitted;if(JSON.stringify(current)!==r.history.at(-1).snapshot_json)throw Error("锁定记录与快照不一致");}
  }
  async function verifyStore(payload){
    if(payload.schema_version!=="blind-progress-v1"||payload.manifest_sha256!==window.MANIFEST_SHA)throw Error("备份不属于本测量包");
    if(!Number.isInteger(payload.position)||payload.position<0||payload.position>=samples.length)throw Error("备份页码非法");
    for(const [id,r] of Object.entries(payload.records)){const s=samples.find(x=>x.blind_id===id);if(!s)throw Error("备份含陌生样本");validate(r,s,false);await checkHistory(r,s);}
  }
  function draw(){
    const svg=$("overlay"),s=sample(),r=rec();svg.setAttribute("viewBox",`0 0 ${s.width} ${s.height}`);svg.innerHTML="";
    const radius=Math.max(s.width,s.height)*.0035,colors=["#e24b4b","#048cd5","#179855","#9052cc","#d37b13"];
    const add=(tag,attrs)=>{const e=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);svg.appendChild(e);return e;};
    if(show){r.items.forEach((t,i)=>{const c=colors[i%colors.length];if(t.points_px.length){add("polyline",{points:t.points_px.map(p=>p.join(",")).join(" "),fill:"none",stroke:c,"stroke-width":radius*.65});t.points_px.forEach((p,j)=>{if(j===0)return;add("circle",{cx:p[0],cy:p[1],r:radius,fill:j===t.points_px.length-1?"white":c,stroke:c,"stroke-width":radius*.4,"data-item":i,"data-point":j});});}else if(t.tip_xy)add("circle",{cx:t.tip_xy[0],cy:t.tip_xy[1],r:radius,fill:"white",stroke:c,"stroke-width":radius*.5,"data-item":i,"data-point":"tip"});});
      if(r.base_xy)add("circle",{cx:r.base_xy[0],cy:r.base_xy[1],r:radius*1.25,fill:"#333",stroke:"white","stroke-width":radius*.35});}
  }
  function sizing(){const s=sample(),w=Math.max(200,$("viewport").clientWidth-2)*zoom;$("stage").style.width=w+"px";$("stage").style.height=w*s.height/s.width+"px";}
  function render(){
    const r=rec(),t=item();$("blind").textContent=sample().blind_id;$("counter").textContent=`${position+1} / ${samples.length}`;$("status").textContent=`已锁定 ${Object.values(records).filter(x=>x.submitted).length} / ${samples.length}`;$("lock").textContent=r.submitted?`已提交锁定 · revision ${r.revision}`:technical()?"技术修订草稿":"未提交";
    $("prev").disabled=position===0||busy;$("next").disabled=position===samples.length-1||busy;
    for(const id of ["base","trace","tip","undo"])$(id).disabled=locked();for(const id of ["add","remove"])$(id).disabled=locked()||technical();
    $("submit").disabled=locked();$("technical").disabled=!r.submitted||busy;
    $("mode").textContent=mode==="base"?"当前：设置共同基点":mode==="trace"?"当前：描迹":mode==="tip"?"当前：记录尖端":"当前：浏览";
    $("base-info").textContent=r.base_xy?"共同基点："+r.base_xy.map(x=>x.toFixed(1)).join(", "):"共同基点未设置；如无法可靠定位，请说明原因，不得虚构完整几何。";
    $("items").innerHTML="";r.items.forEach((x,i)=>{const b=document.createElement("button");b.textContent=(x.gt_id||`待提交结构 ${i+1}`)+" · "+(x.visibility_status||"未选择状态");b.className=i===selected?"selected":"";b.onclick=()=>{selected=i;render();};$("items").appendChild(b);});
    $("details").hidden=!t;if(t){$("state").value=t.visibility_status;$("occlusion").value=t.occlusion;$("interpolation").value=t.interpolation_used;$("confidence").value=t.confidence;$("item-note").value=t.note;$("item-summary").textContent=`控制点 ${t.points_px.length}；尖端 ${t.tip_xy?t.tip_xy.map(x=>x.toFixed(1)).join(", "):"未记录"}`;}
    for(const id of ["state","occlusion","interpolation","confidence","item-note","image-note","checked"])$(id).disabled=locked()||technical();$("image-note").value=r.image_note;$("checked").checked=r.whole_plant_checked;draw();
  }
  function go(n){position=n;selected=-1;mode="browse";undo=[];zoom=1;$("revision-panel").hidden=true;$("photo").src=sample().image_filename;sizing();render();save();say("");}
  function xy(ev){const b=$("overlay").getBoundingClientRect(),s=sample();return [Math.max(0,Math.min(s.width,(ev.clientX-b.left)*s.width/b.width)),Math.max(0,Math.min(s.height,(ev.clientY-b.top)*s.height/b.height))];}
  $("overlay").addEventListener("pointerdown",e=>{if(locked())return;const el=e.target;if(el.dataset.point){remember();drag={i:Number(el.dataset.item),p:el.dataset.point};selected=drag.i;$("overlay").setPointerCapture(e.pointerId);return;}
    const p=xy(e),r=rec(),t=item();if(mode==="base"){remember();r.base_xy=p;for(const x of r.items)if(x.points_px.length)x.points_px[0]=clone(p);}
    else if(mode==="trace"){if(!t||t.visibility_status!=="measurable"||!r.base_xy)return say("先选择可测结构并设置共同基点",true);remember();if(!t.points_px.length)t.points_px=[clone(r.base_xy)];t.points_px.push(p);t.tip_xy=p;}
    else if(mode==="tip"){if(!t||t.visibility_status==="measurable")return say("可测结构的尖端为描迹最后一点；此工具仅定位非可测结构",true);remember();t.tip_xy=p;}
    else return;save();render();});
  $("overlay").addEventListener("pointermove",e=>{if(!drag)return;const t=rec().items[drag.i],p=xy(e);if(drag.p==="tip")t.tip_xy=p;else{t.points_px[Number(drag.p)]=p;if(Number(drag.p)===t.points_px.length-1)t.tip_xy=p;}draw();});
  $("overlay").addEventListener("pointerup",()=>{if(drag){drag=null;save();render();}});
  $("overlay").addEventListener("pointercancel",()=>{drag=null;save();render();});
  $("base").onclick=()=>{mode="base";render();};$("trace").onclick=()=>{mode="trace";render();};$("tip").onclick=()=>{mode="tip";render();};
  $("add").onclick=()=>{if(locked()||technical())return;remember();rec().items.push({trace_uuid:uuid(),visibility_status:"",points_px:[],tip_xy:null,occlusion:"none",interpolation_used:"no",confidence:"",note:""});selected=rec().items.length-1;mode="browse";save();render();};
  $("remove").onclick=()=>{if(locked()||technical()||selected<0)return;if(!confirm("删除尚未提交的当前结构？"))return;remember();rec().items.splice(selected,1);selected=-1;save();render();};
  $("undo").onclick=()=>{if(locked()||!undo.length)return;records[sample().blind_id]=undo.pop();selected=Math.min(selected,rec().items.length-1);save();render();};
  const fields={state:"visibility_status",occlusion:"occlusion",interpolation:"interpolation_used",confidence:"confidence","item-note":"note"};
  for(const [id,f] of Object.entries(fields))$(id).addEventListener(id==="item-note"?"input":"change",()=>{if(locked()||technical()||!item())return;const t=item(),value=$(id).value;if(f==="visibility_status"&&value!=="measurable"&&t.points_px.length&&!confirm("此状态只保留存在性/定位；清除尚未提交的几何？")){render();return;}remember();t[f]=value;if(f==="visibility_status"&&value!=="measurable"){t.points_px=[];t.interpolation_used="no";}save();render();});
  $("image-note").oninput=()=>{if(locked()||technical())return;rec().image_note=$("image-note").value;save();};$("checked").onchange=()=>{if(locked()||technical())return;rec().whole_plant_checked=$("checked").checked;save();};
  $("submit").onclick=async()=>{if(locked())return;try{validate(rec(),sample());busy=true;render();const r=rec();r.items.forEach((t,i)=>t.gt_id=`GT${String(i+1).padStart(2,"0")}`);r.submitted_at=new Date().toISOString();r.previous_submission_sha256=r.history.length?r.history.at(-1).sha256:null;const snapshot=clone(r);delete snapshot.history;delete snapshot.submitted;const snapshot_json=JSON.stringify(snapshot);const h={snapshot_json,sha256:await sha(snapshot_json)};
      const candidate=clone(r);candidate.history.push(h);candidate.submitted=true;await checkHistory(candidate,sample());records[sample().blind_id]=candidate;undo=[];mode="browse";save();say("已锁定并保存 SHA-256："+h.sha256+"。请使用备份或全套导出进行持久保存。");}
    catch(e){say(e.message,true);}finally{busy=false;render();}};
  $("technical").onclick=()=>{$("revision-panel").hidden=false;};
  $("begin-revision").onclick=()=>{if(!rec().submitted)return;const reason=$("revision-reason").value.trim();if(!reason)return say("必须填写可核查的纯技术原因",true);const r=rec();r.submitted=false;r.revision++;r.technical_revision={kind:$("revision-kind").value,reason};delete r.submitted_at;delete r.previous_submission_sha256;undo=[];$("revision-panel").hidden=true;save();render();say("原快照和 hash 已保留；仅允许技术坐标修订，不允许重判状态/数量/身份。");};
  $("prev").onclick=()=>go(position-1);$("next").onclick=()=>go(position+1);
  const changeZoom=f=>{zoom=Math.min(8,Math.max(.5,zoom*f));sizing();};$("zoom-in").onclick=()=>changeZoom(1.25);$("zoom-out").onclick=()=>changeZoom(.8);$("fit").onclick=()=>{zoom=1;sizing();};$("toggle").onclick=()=>{show=!show;$("toggle").textContent=show?"隐藏人工标记":"显示人工标记";draw();};$("viewport").addEventListener("wheel",e=>{e.preventDefault();changeZoom(e.deltaY<0?1.12:1/1.12);},{passive:false});window.addEventListener("resize",sizing);
  function download(name,text,type="application/json"){const a=document.createElement("a");a.download=name;a.href=URL.createObjectURL(new Blob([text],{type}));a.textContent="下载 "+name;$("downloads").appendChild(a);return a;}
  async function backup(){const payload_json=JSON.stringify({schema_version:"blind-progress-v1",manifest_sha256:window.MANIFEST_SHA,position,records});const outer={payload_json,sha256:await sha(payload_json)};const a=download("annotation_progress.json",JSON.stringify(outer,null,2));a.click();}
  $("backup").onclick=()=>backup().catch(e=>say(e.message,true));$("restore").onclick=()=>$("restore-file").click();
  $("restore-file").onchange=async()=>{try{const file=$("restore-file").files[0];if(!file)return;const data=JSON.parse(await file.text());if(await sha(data.payload_json)!==data.sha256)throw Error("备份 SHA 校验失败");const payload=JSON.parse(data.payload_json);await verifyStore(payload);for(const [id,r] of Object.entries(records))if(r.history.length){const other=payload.records[id];if(!other||other.history.length<r.history.length||r.history.some((h,i)=>other.history[i]?.sha256!==h.sha256))throw Error("恢复不得丢弃或替换已有正式版本");}records=payload.records;position=payload.position;go(position);say("恢复成功，正式记录和修订链已校验");}catch(e){say(e.message,true);}finally{$("restore-file").value="";}};
  const csvCell=x=>'"'+String(x??"").replaceAll('"','""')+'"';const csv=(head,rows)=>"\uFEFF"+[head,...rows].map(r=>r.map(csvCell).join(",")).join("\r\n")+"\r\n";
  $("export").onclick=async()=>{try{if(samples.some(s=>!records[s.blind_id]?.submitted))throw Error("请先完成并锁定全部图片");for(const s of samples)await checkHistory(records[s.blind_id],s);const payload={schema_version:"blind-raw-v1",manifest_sha256:window.MANIFEST_SHA,rater_id:"S1",samples:samples.map(s=>({blind_id:s.blind_id,history:records[s.blind_id].history}))};const payload_json=JSON.stringify(payload);const raw=JSON.stringify({payload_json,sha256:await sha(payload_json)},null,2)+"\n";
    const sr=[],tr=[];for(const x of payload.samples){const h=x.history.at(-1),r=JSON.parse(h.snapshot_json);sr.push([x.blind_id,r.revision,h.sha256,h.snapshot_json]);for(const t of r.items)tr.push([x.blind_id,t.gt_id,t.trace_uuid,t.visibility_status,JSON.stringify(t.points_px),JSON.stringify(t.tip_xy),t.occlusion,t.interpolation_used,t.confidence,t.note,h.sha256]);}
    const files={"raw_annotations.json":raw,"sessions.csv":csv(["blind_id","revision","record_sha256","record_json"],sr),"traces.csv":csv(["blind_id","gt_id","trace_uuid","visibility_status","points_px","tip_xy","occlusion","interpolation_used","confidence","note","record_sha256"],tr)};
    const ledger={schema_version:"blind-export-sha-v1",manifest_sha256:window.MANIFEST_SHA,files:{}};for(const [name,text]of Object.entries(files))ledger.files[name]=await sha(text);files["SHA256_LEDGER.json"]=JSON.stringify(ledger,null,2)+"\n";$("downloads").innerHTML="";for(const [name,text]of Object.entries(files))download(name,text,name.endsWith(".csv")?"text/csv;charset=utf-8":"application/json");say("全部完成。请逐一保存下方四个文件；不要只保存 CSV。原始版本和修订链位于 JSON。");
    }catch(e){say(e.message,true);}};
  try{const stored=localStorage.getItem(key);if(stored){const p=JSON.parse(stored);records=p.records;position=p.position;if(!Number.isInteger(position)||position<0||position>=samples.length)throw Error("页码损坏");for(const [id,r]of Object.entries(records)){const s=samples.find(x=>x.blind_id===id);if(!s)throw Error("陌生样本");validate(r,s,false);}}go(position);(async()=>{for(const [id,r]of Object.entries(records))await checkHistory(r,samples.find(x=>x.blind_id===id));})().catch(e=>{busy=true;render();say("本地正式记录校验失败，禁止继续："+e.message,true);});}catch(e){busy=true;render();say("本地记录无法安全恢复："+e.message,true);}
})();
