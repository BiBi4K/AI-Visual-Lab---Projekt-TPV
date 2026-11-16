"use strict";
(() => {
  const el = id => document.getElementById('gd-' + id);
  const palette = ['#087e70', '#d57526', '#6574cd', '#cd668a'];
  let data = null, position = 0, timer = null, revision = 0, pending = null, playback = 0;
  const stop = () => { clearTimeout(timer); timer = null; playback++; };
  const limit = () => data ? Math.max(...data.runs.map(r => r.history.length - 1)) : 0;
  const state = run => run.history[Math.min(position, run.history.length - 1)];
  function number(id) {
    if (!el(id).checkValidity() || el(id).value === '') throw Error('Sprawdź pole: ' + el(id).parentElement.textContent.trim());
    return Number(el(id).value);
  }
  function reset() {
    stop(); revision++; data = null; position = 0;
    el('timeline').max = 0; el('status').textContent = 'Gotowe. Uruchom eksperyment lub wybierz Krok.';
    draw();
  }
  async function prepare() {
    if (data) return true;
    if (pending && pending.version === revision) return pending.task;
    const version = revision;
    const body = {n:number('n'), noise:number('noise'), seed:number('seed'), mode:el('mode').value,
      learning_rate:number('lr'), max_iterations:number('limit'), beta:number('beta'), compare:el('compare').checked};
    el('status').textContent = 'Obliczanie przebiegu…';
    const task = (async () => {
      const response = await fetch('/api/gradient', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
      const result = await response.json();
      if (version !== revision) return false;
      if (!response.ok) throw Error(result.error || 'Błąd serwera.');
      data = result; position = 0; el('timeline').max = limit();
      el('status').textContent = 'Historia gotowa. Możesz przeglądać kolejne kroki.';
      draw(); return true;
    })();
    const request = {version, task}; pending = request;
    try { return await task; } finally { if (pending === request) pending = null; }
  }
  async function guarded(action) {
    try { await action(); } catch (error) { el('status').textContent = error.message; }
  }
  function surface(id) {
    const canvas = el(id), rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * ratio)); canvas.height = Math.max(1, Math.round(rect.height * ratio));
    const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
    return {ctx, w:rect.width, h:rect.height};
  }
  function axes(id, xmin, xmax, ymin, ymax, xlabel) {
    const {ctx,w,h} = surface(id), pad = 52;
    const x = v => pad + (v-xmin)/(xmax-xmin)*(w-pad-18);
    const y = v => h-35-(v-ymin)/(ymax-ymin)*(h-55);
    ctx.font = '10px system-ui';
    const format = v => Math.abs(v) >= 10000 ? v.toExponential(1) : Number(v.toPrecision(3)).toString();
    for (let i=0;i<=4;i++) {
      const vx=xmin+(xmax-xmin)*i/4, vy=ymin+(ymax-ymin)*i/4;
      ctx.strokeStyle='#e4ebee'; ctx.beginPath();ctx.moveTo(x(vx),20);ctx.lineTo(x(vx),h-35);ctx.moveTo(pad,y(vy));ctx.lineTo(w-18,y(vy));ctx.stroke();
      ctx.fillStyle='#657682';ctx.fillText(format(vx),x(vx)-6,h-20);ctx.fillText(format(vy),0,y(vy)+3);
    }
    ctx.fillText(xlabel,w-85,h-4);
    return {ctx,x,y,w,h};
  }
  function draw() {
    if (document.getElementById('gradient').hidden) return;
    el('timeline').value = position; el('position').textContent = position;
    if (!data) {
      ['plot','loss'].forEach(id => {const {ctx}=surface(id);ctx.fillStyle='#657682';ctx.font='13px system-ui';ctx.fillText('Uruchom eksperyment, aby zobaczyć wyniki.',20,55);});
      el('metrics').textContent='Brak przebiegu.';el('legend').replaceChildren();return;
    }
    const ys = data.points.map(p=>p[1]), lo=Math.min(0,...ys), hi=Math.max(10,...ys), margin=(hi-lo)*0.1;
    const plot=axes('plot',0,2,lo-margin,hi+margin,'x');
    const {ctx,x,y,w,h}=plot;
    ctx.save();ctx.beginPath();ctx.rect(52,20,Math.max(0,w-70),Math.max(0,h-55));ctx.clip();
    data.points.forEach(p=>{ctx.fillStyle='#94aaa7';ctx.beginPath();ctx.arc(x(p[0]),y(p[1]),3,0,Math.PI*2);ctx.fill();});
    data.runs.forEach((run,i)=>{
      const [b,a]=state(run).theta;ctx.strokeStyle=palette[i];ctx.lineWidth=2;
      // Clip line in data coordinates to avoid huge canvas coordinates during divergence.
      let left=0,right=2;
      if (Math.abs(a)>1e-15) {const cuts=[(lo-margin-b)/a,(hi+margin-b)/a].sort((u,v)=>u-v);left=Math.max(left,cuts[0]);right=Math.min(right,cuts[1]);}
      if(left<=right){ctx.beginPath();ctx.moveTo(x(left),y(b+a*left));ctx.lineTo(x(right),y(b+a*right));ctx.stroke();}
    });ctx.restore();
    el('legend').replaceChildren();el('metrics').replaceChildren();
    data.runs.forEach((run,i)=>{
      const label = (i===0?'Twój wybór: ':'Standard: ')+run.mode+' · LR '+run.learning_rate+(run.mode==='Momentum'?' · beta '+run.beta:'');
      const item=document.createElement('div');item.style.color=palette[i];item.textContent='━ '+label;el('legend').append(item);
      const s=state(run), done=position>=run.history.length-1;
      const row=document.createElement('div');row.textContent=`${i===0?'Twój wybór':'Standard'} ${run.mode}: krok ${s.iteration} · b = ${s.theta[0].toFixed(4)} · a = ${s.theta[1].toFixed(4)} · błąd = ${s.loss.toPrecision(5)} · ${done?(run.diverged?'Rozbieżność — zmniejsz LR':'Osiągnięto limit'):'W trakcie'}`;el('metrics').append(row);
    });
    const log=el('log').checked, transform=v=>log?Math.log10(Math.max(v,1e-15)):v;
    const values=data.runs.flatMap(r=>r.history.map(s=>transform(s.loss)));
    let low=log?Math.min(...values):0,high=Math.max(...values);if(high-low<1e-10)high=low+1;
    const loss=axes('loss',0,Math.max(1,limit()),low,high,'Krok');
    if(log){loss.ctx.fillStyle='#657682';loss.ctx.fillText('log10(błąd)',55,12);}
    data.runs.forEach((run,i)=>{loss.ctx.strokeStyle=palette[i];loss.ctx.lineWidth=2;loss.ctx.beginPath();run.history.slice(0,position+1).forEach((s,j)=>{const px=loss.x(s.iteration),py=loss.y(transform(s.loss));if(j===0)loss.ctx.moveTo(px,py);else loss.ctx.lineTo(px,py);});loss.ctx.stroke();});
  }
  function tick() {
    if(position<limit()){position++;draw();timer=setTimeout(tick,number('speed'));}
    else {stop();el('status').textContent='Odtwarzanie zakończone. Wyniki możesz przejrzeć suwakiem.';}
  }
  el('play').onclick=()=>guarded(async()=>{stop();const token=playback;if(!await prepare()||token!==playback)return;if(position>=limit())position=0;draw();timer=setTimeout(tick,number('speed'));});
  el('step').onclick=()=>guarded(async()=>{stop();const token=playback;if(!await prepare()||token!==playback)return;position=Math.min(position+1,limit());draw();});
  el('pause').onclick=()=>{stop();el('status').textContent='Animacja zatrzymana.';};
  el('reset').onclick=reset;
  el('timeline').oninput=()=>{stop();position=Number(el('timeline').value);draw();};
  ['n','noise','seed','mode','lr','limit','beta','compare'].forEach(id=>el(id).addEventListener('change',()=>{el('beta-label').hidden=el('mode').value!=='Momentum';reset();}));
  el('log').onchange=draw;
  window.addEventListener('resize',draw);
  window.addEventListener('hashchange',()=>{stop();draw();});
  reset();
})();
