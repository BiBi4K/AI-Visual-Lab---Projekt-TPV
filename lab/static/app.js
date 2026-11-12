"use strict";
const $ = id => document.getElementById(id);
const colors = ['#087e70','#e18b43','#6574cd','#cd668a','#329cb5','#9b7f42','#a153bf','#71a340','#cf534e','#456f90','#ac8aaf','#53744d'];
let points = [], runs = [], position = 0, timer = null, revision = 0, pending = null, playback = 0;
const storageKey = 'ai-visual-lab.datasets.v1';
function notify(message) { $('status').textContent = message; }
function pause() { clearTimeout(timer); timer = null; playback++; }
function invalidate() { pause(); revision++; runs = []; position = 0; $('timeline').max = 0; $('timeline').value = 0; draw(); }
function number(id) { const input = $(id); if (!input.checkValidity() || input.value === '') throw Error('Sprawdź pole: ' + input.parentElement.textContent.trim()); return Number(input.value); }
async function api(path, body) {
  const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw Error(data.error || 'Błąd serwera.');
  return data;
}
async function guarded(action) { try { await action(); } catch(error) { notify(error.message); } }
function current(run) { return run?.history[Math.min(position, run.history.length - 1)]; }
function surface(id) {
  const canvas = $(id), rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * ratio)); canvas.height = Math.max(1, Math.round(rect.height * ratio));
  const ctx = canvas.getContext('2d'); ctx.scale(ratio,ratio);
  return {canvas,ctx,w:rect.width,h:rect.height};
}
function bounds() {
  return [0,1].map(axis => {
    const values = points.map(p=>p[axis]);
    const low = Math.min(0,...values), high = Math.max(100,...values);
    return [low,high];
  });
}
function plot(id,run) {
  const {ctx,w,h} = surface(id), pad=30, limits=bounds();
  const x=v=>pad+(v-limits[0][0])/(limits[0][1]-limits[0][0])*(w-2*pad);
  const y=v=>h-pad-(v-limits[1][0])/(limits[1][1]-limits[1][0])*(h-2*pad);
  ctx.font='10px system-ui'; ctx.fillStyle='#82939c';
  for(let i=0;i<=5;i++) {
    const vx=limits[0][0]+i/5*(limits[0][1]-limits[0][0]), vy=limits[1][0]+i/5*(limits[1][1]-limits[1][0]);
    ctx.strokeStyle='#eaf0f2';ctx.beginPath();ctx.moveTo(x(vx),pad);ctx.lineTo(x(vx),h-pad);ctx.moveTo(pad,y(vy));ctx.lineTo(w-pad,y(vy));ctx.stroke();
    ctx.fillText(vx.toFixed(0),x(vx)-5,h-10);ctx.fillText(vy.toFixed(0),2,y(vy)+3);
  }
  const state=current(run);
  points.forEach((point,i)=>{ctx.fillStyle=state?colors[state.labels[i]]:'#8da9a4';ctx.beginPath();ctx.arc(x(point[0]),y(point[1]),4,0,Math.PI*2);ctx.fill();});
  if(state) state.centroids.forEach((center,j)=>{
    ctx.strokeStyle=colors[j];ctx.lineWidth=1.5;ctx.beginPath();
    run.history.slice(0,Math.min(position+1,run.history.length)).forEach((step,i)=>{const c=step.centroids[j];if(i===0)ctx.moveTo(x(c[0]),y(c[1]));else ctx.lineTo(x(c[0]),y(c[1]));});ctx.stroke();
    ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x(center[0])-6,y(center[1])-6);ctx.lineTo(x(center[0])+6,y(center[1])+6);ctx.moveTo(x(center[0])-6,y(center[1])+6);ctx.lineTo(x(center[0])+6,y(center[1])-6);ctx.stroke();
  });
}
function loss() {
  const {ctx,w,h}=surface('loss'),pad=30;
  if(!runs.length){ctx.fillStyle='#82939c';ctx.font='12px system-ui';ctx.fillText('Uruchom algorytm, aby zobaczyć historię.',pad,h/2);return;}
  const max=Math.max(1,...runs.flatMap(r=>r.history.map(s=>s.inertia))), steps=Number($('timeline').max)||1;
  ctx.strokeStyle='#e4ebee';ctx.beginPath();ctx.moveTo(pad,10);ctx.lineTo(pad,h-pad);ctx.lineTo(w-10,h-pad);ctx.stroke();
  ctx.fillStyle='#82939c';ctx.font='10px system-ui';ctx.fillText(max.toFixed(0),0,15);ctx.fillText('0',12,h-pad);ctx.fillText('Iteracja →',w-85,h-8);
  runs.forEach((run,j)=>{ctx.strokeStyle=colors[j];ctx.lineWidth=2;ctx.beginPath();run.history.slice(0,position+1).forEach((s,i)=>{const x=pad+s.iteration/steps*(w-pad-12),y=h-pad-s.inertia/max*(h-pad-10);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();});
}
function metrics(run) {
  const state=current(run);if(!state)return 'Gotowe do eksperymentu · ' + points.length + ' punktów';
  const done=position>=run.history.length-1;
  return `k = ${run.k} · Iteracja ${state.iteration} · Inertia ${state.inertia.toFixed(2)} · ${done?(run.converged?'Zbieżność osiągnięta':'Osiągnięto limit'):'W trakcie'}`;
}
function draw() {
  if($('kmeans').hidden)return;
  $('count').textContent=`${points.length} punktów`;
  $('comparison').hidden=!$('compare').checked;$('plots').classList.toggle('two',$('compare').checked);
  plot('plot',runs[0]);$('metrics').textContent=metrics(runs[0]);
  if($('compare').checked){plot('plot2',runs[1]);$('metrics2').textContent=metrics(runs[1]);}
  loss();$('timeline').value=position;
}
async function prepare() {
  if(runs.length)return true;
  if(pending)return pending;
  const version=revision;
  const task=(async()=>{
    const body={points:points.map(p=>[...p]),k:number('k'),seed:number('seed'),max_iterations:number('limit')};
    const requests=[api('/api/kmeans',body)];
    if($('compare').checked)requests.push(api('/api/kmeans',{...body,k:number('k2')}));
    const result=await Promise.all(requests);
    if(version!==revision)return false;
    runs=result;position=0;$('timeline').max=Math.max(...runs.map(r=>r.history.length-1));draw();return true;
  })();
  pending=task;
  try{return await task;}finally{if(pending===task)pending=null;}
}
function tick() {
  if(position<Number($('timeline').max)){position++;draw();timer=setTimeout(tick,1600-number('speed'));}
  else {pause();notify('Eksperyment zakończony. Możesz przejrzeć iteracje suwakiem.');}
}
$('play').onclick=()=>guarded(async()=>{pause();const token=playback;if(!await prepare()||token!==playback)return;if(position>=Number($('timeline').max))position=0;draw();timer=setTimeout(tick,1600-number('speed'));notify('Odtwarzanie przebiegu K-Means.');});
$('pause').onclick=()=>{pause();notify('Animacja zatrzymana. Uruchom, aby wznowić.');};
$('step').onclick=()=>guarded(async()=>{pause();const token=playback;if(!await prepare()||token!==playback)return;position=Math.min(position+1,Number($('timeline').max));draw();});
$('reset').onclick=()=>{invalidate();notify('Przebieg zresetowany. Dane pozostają dostępne.');};
$('timeline').oninput=()=>{pause();position=Number($('timeline').value);draw();};
['k','k2','seed','limit'].forEach(id=>$(id).onchange=invalidate);
$('compare').onchange=()=>{$('second-label').hidden=!$('compare').checked;invalidate();};
let generation=0;
$('generate').onclick=()=>guarded(async()=>{
  const body={n:number('n'),clusters:number('clusters'),noise:number('noise'),seed:number('seed')};
  const ticket=++generation,version=revision;
  const result=await api('/api/datasets/generate',body);
  if(ticket!==generation||version!==revision)return;
  points=result.points;invalidate();notify('Nowy zbiór jest gotowy. Kliknij wykres, aby go edytować.');
});
$('plot').onclick=event=>guarded(()=>{
  const rect=$('plot').getBoundingClientRect(),pad=30,px=event.clientX-rect.left,py=event.clientY-rect.top;
  if(px<pad||px>rect.width-pad||py<pad||py>rect.height-pad)return;
  const limits=bounds();
  if($('edit').value==='remove') {
    let best=-1,distance=14;
    points.forEach((p,i)=>{const x=pad+(p[0]-limits[0][0])/(limits[0][1]-limits[0][0])*(rect.width-2*pad),y=rect.height-pad-(p[1]-limits[1][0])/(limits[1][1]-limits[1][0])*(rect.height-2*pad);const d=Math.hypot(px-x,py-y);if(d<distance){distance=d;best=i;}});
    if(best>=0)points.splice(best,1);else return;
  }else{
    if(points.length>=2000)throw Error('Limit wynosi 2000 punktów.');
    points.push([limits[0][0]+(px-pad)/(rect.width-2*pad)*(limits[0][1]-limits[0][0]),limits[1][0]+(rect.height-pad-py)/(rect.height-2*pad)*(limits[1][1]-limits[1][0])]);
  }invalidate();
});
function saved() {const data=JSON.parse(localStorage.getItem(storageKey)||'{}');if(!data||typeof data!=='object'||Array.isArray(data))throw Error('Niepoprawny format zapisów w przeglądarce.');return data;}
function refresh() {const select=$('saved');select.replaceChildren(new Option('Wybierz zbiór',''));Object.keys(saved()).forEach(name=>select.add(new Option(name,name)));}
function validatePoints(data) {
  if(!Array.isArray(data)||data.length<1||data.length>2000||data.some(p=>!Array.isArray(p)||p.length!==2||p.some(v=>typeof v!=='number'||!Number.isFinite(v)||Math.abs(v)>1000)))throw Error('Niepoprawny zbiór: wymagane 1–2000 par liczb w zakresie [-1000, 1000].');
  return data;
}
$('save').onclick=()=>guarded(()=>{const name=$('dataset-name').value.trim();if(!name)throw Error('Wpisz nazwę zbioru.');validatePoints(points);const all=saved();if(Object.hasOwn(all,name))throw Error('Ta nazwa już istnieje. Wybierz inną lub usuń poprzedni zapis.');Object.defineProperty(all,name,{value:points,enumerable:true,configurable:true});localStorage.setItem(storageKey,JSON.stringify(all));refresh();$('saved').value=name;notify('Zapisano zbiór: '+name);});
$('load').onclick=()=>guarded(()=>{const data=saved()[$('saved').value];points=validatePoints(data).map(p=>[...p]);invalidate();notify('Wczytano zapisany zbiór.');});
$('delete').onclick=()=>guarded(()=>{const name=$('saved').value;if(!name)throw Error('Wybierz zapisany zbiór.');const all=saved();delete all[name];localStorage.setItem(storageKey,JSON.stringify(all));refresh();notify('Usunięto zapis: '+name);});
$('export').onclick=()=>guarded(()=>{validatePoints(points);const url=URL.createObjectURL(new Blob([JSON.stringify({version:1,points},null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='ai-visual-lab-dataset.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
$('import').onclick=()=>$('file').click();
$('file').onchange=()=>guarded(async()=>{try{const file=$('file').files[0];if(!file)return;if(file.size>512*1024)throw Error('Plik przekracza limit 512 KB.');const data=JSON.parse(await file.text());points=validatePoints(data.points).map(p=>[...p]);invalidate();notify('Zaimportowano zbiór.');}finally{$('file').value='';}});
function route() {
  pause();const page=['home','kmeans','knn','gradient'].includes(location.hash.slice(1))?location.hash.slice(1):'home';
  document.querySelectorAll('.page').forEach(p=>p.hidden=p.id!==page);
  document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('selected',a.hash==='#'+page));draw();
}
window.addEventListener('hashchange',route);window.addEventListener('resize',draw);
route();guarded(refresh);
guarded(async()=>{const result=await api('/api/datasets/generate',{n:200,clusters:3,noise:5,seed:42});if(!revision){points=result.points;invalidate();}});
