"use strict";
(() => {
  const data = window.REVIEW_DATA, semantic = data.stage === "semantic";
  const key = `review:${data.build_version}:${data.package_id}:${data.stage}:${data.semantic_export_sha256 || ""}`;
  const $ = id => document.getElementById(id), clone = value => JSON.parse(JSON.stringify(value));
  const statuses = {measurable:"确定是叶片，完整路径可测",visible_unmeasurable:"确定是叶片，但完整路径不可测",uncertain:"是否为叶片尚无法确定",non_leaf:"确定非叶片（毛边／杂物等）"};
  const reasons = {none:"无此问题",cropped_tip:"叶尖被裁剪／截断",cropped_base:"基部被裁剪／截断",occlusion:"遮挡",damage:"组织破损",identity_uncertain:"叶片身份不确定",artifact:"毛边／阴影／杂物",other:"其他（写明理由）"};
  const confidence = {high:"高",medium:"中",low:"低"};
  let index = 0, selected = null, locate = false, views = [], store = {};
  try { store = JSON.parse(localStorage.getItem(key) || "{}"); } catch { $("message").textContent = "本地进度读取失败，请从备份恢复。"; }
  const item = () => data.items[index];
  const make = () => ({blind_id:item().blind_id,image_sha256:item().image_sha256,submitted:false,revision:0,history:[],notes:"",review_complete:false,candidates_seen:!semantic,leaves:semantic?[]:clone(item().semantic.leaves).map(l=>({...l,geometry_choice:"",geometry_notes:"",geometry_confidence:""}))});
  const rec = () => store[item().blind_id] ||= make();
  function persist(){try{localStorage.setItem(key,JSON.stringify(store));}catch{message("浏览器存储失败，请立即下载进度备份。");}}
  function message(text){$("message").textContent=text;}
  function element(tag,text){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;return el;}
  function selection(values,value,callback,disabled=false){const select=element("select");select.append(new Option("请选择",""));for(const [k,v] of Object.entries(values))select.append(new Option(v,k));select.value=value||"";select.disabled=disabled;select.onchange=()=>{callback(select.value);persist();};return select;}
  function field(parent,title,node){const label=element("label",title);label.append(node);parent.append(label);}
  function geometryOptions(i){const values={needs_redraw:"所有候选均不适合，需要重新描迹",hold_uncertain:"暂不能裁决，保留争议"};for(const s of i.sets||[])for(const t of s.traces)values[`${s.label}:${t.candidate_trace_id}`]=`采用 ${s.label} / ${t.candidate_trace_id}`;return values;}
  function createView(parent,i,set){
    const canvas=element("canvas"),ctx=canvas.getContext("2d"),img=new Image();parent.append(canvas);
    let zoom=1,px=0,py=0,drag=null,scale=1,ox=0,oy=0;
    function draw(){const box=parent.getBoundingClientRect(),dpr=devicePixelRatio||1;canvas.width=Math.max(1,Math.round(box.width*dpr));canvas.height=Math.max(1,Math.round(box.height*dpr));ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,box.width,box.height);if(!img.complete||!img.naturalWidth)return;scale=Math.min(box.width/img.naturalWidth,box.height/img.naturalHeight)*zoom;ox=(box.width-img.naturalWidth*scale)/2+px;oy=(box.height-img.naturalHeight*scale)/2+py;ctx.drawImage(img,ox,oy,img.naturalWidth*scale,img.naturalHeight*scale);
      if(set){ctx.save();ctx.globalAlpha=Number($("opacity").value);ctx.strokeStyle="#116eb5";ctx.fillStyle="#064274";ctx.lineWidth=1.6;ctx.font="bold 13px Arial";for(const t of set.traces){ctx.beginPath();t.points.forEach((p,n)=>n?ctx.lineTo(ox+p[0]*scale,oy+p[1]*scale):ctx.moveTo(ox+p[0]*scale,oy+p[1]*scale));ctx.stroke();const p=t.points.at(-1);ctx.fillText(t.candidate_trace_id,ox+p[0]*scale+4,oy+p[1]*scale-5);}ctx.restore();}
      for(const [n,l] of rec().leaves.entries()){if(!l.point)continue;const x=ox+l.point[0]*scale,y=oy+l.point[1]*scale;ctx.strokeStyle=l.leaf_id===selected?"#f07900":"#27629c";ctx.lineWidth=1.6;ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.stroke();ctx.fillStyle="#153b61";ctx.font="bold 13px Arial";ctx.fillText(String(n+1),x+8,y+12);}
      parent.classList.toggle("locate",semantic&&locate&&!set&&!rec().submitted);
    }
    img.onload=()=>{parent.dataset.ready="true";draw();};img.onerror=()=>message("图片加载失败，请完整复制整个文件夹。");img.src=i.image;
    const observer=new ResizeObserver(draw);observer.observe(parent);
    parent.onwheel=e=>{e.preventDefault();const box=parent.getBoundingClientRect(),cx=e.clientX-box.left,cy=e.clientY-box.top,ix=(cx-ox)/scale,iy=(cy-oy)/scale;zoom=Math.max(1,Math.min(16,zoom*(e.deltaY<0?1.2:1/1.2)));const nextScale=Math.min(box.width/img.naturalWidth,box.height/img.naturalHeight)*zoom;px=cx-ix*nextScale-(box.width-img.naturalWidth*nextScale)/2;py=cy-iy*nextScale-(box.height-img.naturalHeight*nextScale)/2;draw();};
    parent.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,px,py};parent.setPointerCapture(e.pointerId);};
    parent.onpointermove=e=>{if(!drag)return;if(!locate||set){px=drag.px+e.clientX-drag.x;py=drag.py+e.clientY-drag.y;draw();}};
    parent.onpointerup=e=>{if(!drag)return;const moved=Math.hypot(e.clientX-drag.x,e.clientY-drag.y);drag=null;if(semantic&&locate&&!set&&!rec().submitted&&moved<6){const box=parent.getBoundingClientRect(),x=(e.clientX-box.left-ox)/scale,y=(e.clientY-box.top-oy)/scale;if(x<0||y<0||x>=i.width||y>=i.height)return message("请在图像范围内标记。");const leaf=rec().leaves.find(l=>l.leaf_id===selected);if(leaf){leaf.point=[Math.round(x*100)/100,Math.round(y*100)/100];locate=false;persist();render();}}};
    parent.onpointercancel=()=>drag=null;
    const reset=()=>{zoom=1;px=py=0;draw();};parent.ondblclick=reset;
    return {draw,reset,destroy(){observer.disconnect();img.onload=null;}};
  }
  function panel(title,i,set){const box=element("article");box.className="panel";box.append(element("h3",title));const viewport=element("div");viewport.className="viewport";box.append(viewport);$("panels").append(box);views.push(createView(viewport,i,set));}
  function render(){
    views.forEach(v=>v.destroy());views=[];const i=item(),r=rec();$("title").textContent=semantic?"第1步：独立判断叶片与可测性":"第2步：逐叶核对路径几何";
    $("instructions").textContent=semantic?"请独立检查这幅白底标准化输入图。每发现一片叶或需要记录的疑似结构，点击“添加”，再在叶尖、可见截断端或疑似位置点一下定位。这里的点只帮助沟通位置，不计算长度。完整路径必须同时具有可靠的共同基点、连续走向与真实叶尖。看不清时如实选择无法确定。":"下列结构判断来自已提交的第1步，保持原记录。逐个结构检查候选路径是否对应同一片叶、沿中心并到达真实叶尖。可从不同候选中提出逐叶建议；混合来源仍需共同基点与几何复核。新的语义发现写入备注，保留第1步原判断。";
    $("sample").textContent=`匿名样本 ${i.blind_id}`;$("counter").textContent=`${index+1} / ${data.items.length}`;$("status").textContent=r.submitted?`已提交，第 ${r.revision} 版`:"待提交";
    $("prev").disabled=index===0;$("next").disabled=index===data.items.length-1;$("panels").replaceChildren();$("panels").className=semantic?"single":"";panel("标准化输入图（定位圈由您自己添加）",i,null);if(!semantic)for(const s of i.sets)panel(`候选 ${s.label}`,i,s);
    $("add").hidden=!semantic;$("add").disabled=r.submitted;$("pan").classList.toggle("active",!locate);$("opacity").parentElement.hidden=semantic;$("hint").textContent=locate?"请在图上单击定位当前结构；之后可拖动查看。":"滚轮可放大到16倍；拖动平移。点击结构下的“重新定位”可修改定位圈。";
    $("leaves").replaceChildren();r.leaves.forEach((l,n)=>{
      const box=element("article");box.className="leaf"+(selected===l.leaf_id?" selected":"");box.dataset.leafId=l.leaf_id;box.append(element("h3",`结构 ${n+1}${l.point?` · 定位 (${l.point.join(", ")})`:" · 尚未定位"}`));
      const fields=element("div");fields.className="fields";box.append(fields);
      if(semantic){field(fields,"判断",selection(statuses,l.status,v=>{l.status=v;},r.submitted));field(fields,"问题原因",selection(reasons,l.reason,v=>{l.reason=v;},r.submitted));field(fields,"判断信心",selection(confidence,l.confidence,v=>{l.confidence=v;},r.submitted));
        const note=element("textarea");note.rows=2;note.value=l.notes||"";note.disabled=r.submitted;note.placeholder="不可测、非叶片或不确定时，请写明位置和理由";note.oninput=()=>{l.notes=note.value;persist();};box.append(note);
        const position=element("button","重新定位");position.disabled=r.submitted;position.onclick=()=>{selected=l.leaf_id;locate=true;render();};box.append(position);
        const remove=element("button","删除此记录");remove.disabled=r.submitted;remove.onclick=()=>{if(!confirm("删除这条尚未冻结的结构记录？"))return;r.leaves=r.leaves.filter(x=>x!==l);persist();render();};box.append(remove);
      }else{box.append(element("p",`${statuses[l.status]}；${reasons[l.reason]}；${l.notes||"无附注"}`));
        let options=geometryOptions(i);if(l.status!=="measurable")options={no_geometry:"不采用完整路径几何",hold_uncertain:"出现新分歧，留待共同裁决"};
        field(fields,"路径建议",selection(options,l.geometry_choice,v=>{l.geometry_choice=v;},r.submitted));field(fields,"几何判断信心",selection(confidence,l.geometry_confidence,v=>{l.geometry_confidence=v;},r.submitted));
        const note=element("textarea");note.rows=2;note.value=l.geometry_notes||"";note.disabled=r.submitted;note.placeholder="重描、争议、新发现或多个候选等效时，请写明";note.oninput=()=>{l.geometry_notes=note.value;persist();};box.append(note);
      }$("leaves").append(box);
    });
    $("notes").value=r.notes;$("notes").disabled=r.submitted;$("reviewComplete").checked=r.review_complete;$("reviewComplete").disabled=r.submitted;$("submit").disabled=r.submitted;$("revise").disabled=!r.submitted;
  }
  function validate(r){if(!r.review_complete)return "请确认已检查整幅图。";if(!r.leaves.length&&!r.notes.trim())return "没有结构记录时，请说明理由。";const used=new Set();for(const l of r.leaves){if(!l.point||!statuses[l.status]||!reasons[l.reason]||!confidence[l.confidence])return "每个结构都需要定位、分类、原因和信心。";if(l.status!=="measurable"&&(l.reason==="none"||!l.notes.trim()))return "不可测、不确定和非叶片需要原因及文字说明。";if(l.status==="measurable"&&!["none","occlusion"].includes(l.reason))return "裁剪或残缺端点不能归为完整路径可测。";if(!semantic){if(!l.geometry_choice||!l.geometry_confidence)return "请填写每个结构的路径建议和信心。";if(["needs_redraw","hold_uncertain"].includes(l.geometry_choice)&&!l.geometry_notes.trim())return "重描或争议需要说明理由。";if(l.geometry_choice.includes(":")){if(used.has(l.geometry_choice))return "同一候选路径不能分配给两片叶。";used.add(l.geometry_choice);}}}return null;}
  $("prev").onclick=()=>{index--;selected=null;locate=false;render();message("");};$("next").onclick=()=>{index++;selected=null;locate=false;render();message("");};
  $("add").onclick=()=>{selected=crypto.randomUUID();rec().leaves.push({leaf_id:selected,point:null,status:"",reason:"",confidence:"",notes:""});locate=true;persist();render();};
  $("pan").onclick=()=>{locate=false;render();};$("resetView").onclick=()=>views.forEach(v=>v.reset());$("opacity").oninput=()=>views.forEach(v=>v.draw());
  $("notes").oninput=()=>{rec().notes=$("notes").value;persist();};$("reviewComplete").onchange=()=>{rec().review_complete=$("reviewComplete").checked;persist();};
  $("submit").onclick=()=>{const r=rec(),error=validate(r);if(error)return message(error);r.submitted=true;r.revision++;r.submitted_at_utc=new Date().toISOString();persist();render();message("本图已提交并保存。完成整包后请导出两个文件。");};
  $("revise").onclick=()=>{if(!confirm("创建修订将保留当前已提交版本。是否继续？"))return;const r=rec(),prior=clone(r);delete prior.history;r.history.push(prior);r.submitted=false;persist();render();};
  function payload(){return {build_version:data.build_version,package_id:data.package_id,stage:data.stage,semantic_export_sha256:data.semantic_export_sha256||null,exported_at_utc:new Date().toISOString(),records:data.items.map(i=>store[i.blind_id]).filter(Boolean)};}
  function link(name,text){const a=element("a",`下载 ${name}`);a.href=URL.createObjectURL(new Blob([text],{type:name.endsWith(".json")?"application/json":"text/csv;charset=utf-8"}));a.download=name;return a;}
  function clearDownloads(){for(const a of $("downloads").querySelectorAll("a"))URL.revokeObjectURL(a.href);$("downloads").replaceChildren();}
  $("backup").onclick=()=>{clearDownloads();$("downloads").append(link(`${data.stage}_progress.json`,JSON.stringify(payload(),null,2)));};
  $("restore").onclick=()=>$("restoreFile").click();$("restoreFile").onchange=async e=>{const f=e.target.files[0];if(!f)return;try{const p=JSON.parse(await f.text());if(p.build_version!==data.build_version||p.package_id!==data.package_id||p.stage!==data.stage||(p.semantic_export_sha256||null)!==(data.semantic_export_sha256||null)||!Array.isArray(p.records))throw Error("文件不属于本包／本阶段");const valid=new Map(data.items.map(i=>[i.blind_id,i]));const ids=new Set();for(const r of p.records){if(!valid.has(r.blind_id)||ids.has(r.blind_id)||r.image_sha256!==valid.get(r.blind_id).image_sha256||!Array.isArray(r.leaves)||!Array.isArray(r.history)||r.candidates_seen!==!semantic)throw Error("编号、图像或记录结构不符");ids.add(r.blind_id);}if(!confirm("恢复会替换此浏览器当前包的进度，请确认已备份当前进度。"))return;store=Object.fromEntries(p.records.map(r=>[r.blind_id,r]));persist();render();message("进度已恢复。");}catch(error){message(`恢复失败：${error.message}`);}finally{e.target.value="";}};
  $("export").onclick=()=>{for(const i of data.items){const r=store[i.blind_id];if(!r?.submitted||validate(r))return message("仍有未完成或字段不完整的图片，请逐张提交。");}const p=payload(),headers=["blind_id","revision","submitted_at_utc","structure_count","measurable_count","visible_unmeasurable_count","uncertain_count","non_leaf_count","notes","record_json"],cell=v=>`"${String(v??"").replaceAll('"','""')}"`;const rows=p.records.map(r=>[r.blind_id,r.revision,r.submitted_at_utc,r.leaves.length,...["measurable","visible_unmeasurable","uncertain","non_leaf"].map(s=>r.leaves.filter(l=>l.status===s).length),r.notes,JSON.stringify(r)]);const csv="\ufeff"+[headers,...rows].map(row=>row.map(cell).join(",")).join("\r\n");clearDownloads();$("downloads").append(link(`${data.stage}_decisions.json`,JSON.stringify(p,null,2)),link(`${data.stage}_decisions.csv`,csv));message("请依次点击两个下载链接，并一起保存到本包 results 文件夹。几何阶段须由管理员核验后另行发放。");};
  render();
})();
