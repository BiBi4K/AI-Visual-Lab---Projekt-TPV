"use strict";
(() => {
  const el = id => document.getElementById('knn-' + id);
  const canvas = el('plot');
  const palette = ['#2878bf', '#e58027', '#209781', '#9361b5', '#d64e65', '#9d8035'];
  const symbols = ['●', '■', '▲', '◆', '▼', '✚'];
  let data, pattern, queries = [], edits = [], views = [], viewIndex = 0;
  let result, background, drag, revision = 0, generation = 0, fetching = false, timer;
  const status = message => { el('status').textContent = message; };
  const checked = id => el(id).checked;
  const currentView = () => views[viewIndex];
  const numeric = id => number('knn-' + id);

  function fitView() {
    const axes = [0, 1].map(i => data.points.map(p => p[i]));
    const extents = axes.map(a => [Math.min(...a), Math.max(...a)]);
    const radius = Math.max(1, ...extents.map(([a,b]) => (b-a)*0.64));
    return extents.map(([a,b]) => [(a+b)/2-radius, (a+b)/2+radius]);
  }

  async function generate() {
    const ticket = ++generation;
    try {
      const settings = {dataset:el('dataset').value};
      for (const id of ['n_samples','n_classes','noise','separation','imbalance','seed','stretch','rotation']) {
        settings[id] = numeric(id);
      }
      status('Generowanie danych…');
      const next = await api('/api/knn/generate', settings);
      if (ticket !== generation) return;
      data = next;
      pattern = settings.dataset;
      queries = []; edits = []; views = [fitView()]; viewIndex = 0;
      const previous = Number(el('label').value);
      el('label').replaceChildren(...Array.from({length:settings.n_classes}, (_, i) => new Option(i, i)));
      el('label').value = Math.min(previous, settings.n_classes - 1);
      refresh(true);
    } catch (error) { if (ticket === generation) status(error.message); }
  }

  // Query-only changes reuse the last decision map. At most one classification is in flight.
  function refresh(invalidateMap = false) {
    if (!data) return;
    if (invalidateMap) background = null;
    revision++;
    result = null;
    updateInspector();
    draw();
    status('Obliczanie…');
    clearTimeout(timer);
    timer = setTimeout(classify, 60);
  }

  async function classify() {
    if (fetching) return;
    fetching = true;
    const ticket = revision;
    try {
      const body = {
        points:data.points, labels:data.labels, queries, k:numeric('k'),
        metric:el('metric').value, weights:el('weights').value, standardize:checked('standardize'),
        p:el('p').disabled ? 2 : numeric('p')
      };
      if (checked('regions') && !background) {
        body.resolution = numeric('resolution');
        body.bounds = currentView();
      }
      const answer = await api('/api/knn', body);
      if (ticket !== revision) return;
      if (answer.regions) background = regionImage(answer.regions);
      result = answer;
      updateInspector();
      draw();
      status(answer.k < body.k ? 'Liczbę sąsiadów ograniczono do liczby dostępnych punktów.' : '');
    } catch (error) { if (ticket === revision) status(error.message); }
    finally {
      fetching = false;
      if (ticket !== revision) { clearTimeout(timer); timer = setTimeout(classify, 0); }
    }
  }

  function updateInspector() {
    el('rows').replaceChildren();
    const indices = result?.indices || [];
    el('neighbors-box').hidden = !indices.length;
    if (!indices.length) {
      el('prediction').textContent = queries.length ? 'Obliczanie prognozy…' : 'Kliknij wykres w trybie przewidywania, aby zobaczyć głosy sąsiadów.';
      return;
    }
    const q = queries.at(-1);
    const shares = result.shares.map((v,i) => 'K' + result.classes[i] + ' ' + (100*v).toFixed(1) + '%');
    const units = checked('standardize') ? 'po standaryzacji' : 'w jednostkach cech';
    el('prediction').textContent = '(' + q.map(v => v.toFixed(2)).join(', ') + ') → klasa ' +
      result.predictions.at(-1) + ' · ' + shares.join(' / ') + ' · Odległości ' + units;
    indices.forEach((idx,i) => {
      const row = el('rows').insertRow();
      for (const value of [i+1, data.labels[idx], result.distances[i].toFixed(4), (100*result.weights[i]).toFixed(1)+'%']) {
        row.insertCell().textContent = value;
      }
    });
  }

  function regionImage(rows) {
    const image = document.createElement('canvas');
    const n = rows.length;
    image.width = image.height = n;
    const ctx = image.getContext('2d');
    rows.forEach((row,y) => row.forEach((label,x) => {
      ctx.fillStyle = palette[label];
      ctx.fillRect(x, n-1-y, 1, 1);
    }));
    return image;
  }

  function layout() {
    const rect = canvas.getBoundingClientRect();
    const side = Math.min(rect.width - 58, rect.height - 98);
    const left = (rect.width - side + 26)/2, top = 16;
    const [[xmin,xmax],[ymin,ymax]] = currentView();
    const pixel = ([x,y]) => [left + side*(x-xmin)/(xmax-xmin), top + side*(ymax-y)/(ymax-ymin)];
    const point = ([x,y]) => [xmin + (x-left)/side*(xmax-xmin), ymax - (y-top)/side*(ymax-ymin)];
    return {rect, side, left, top, pixel, point};
  }

  function draw() {
    if (!data || $('knn').hidden) return;
    const {rect, side, left, top, pixel} = layout();
    if (side <= 0) return;
    const {ctx,w,h} = surface('knn-plot');
    ctx.fillStyle = 'white';
    ctx.fillRect(0,0,w,h);
    if (background && checked('regions')) {
      ctx.globalAlpha = 0.16;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(background, left, top, side, side);
      ctx.globalAlpha = 1;
    }
    const line = (a,b) => {
      ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
    };
    const bounds = currentView();
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#53627a';
    ctx.strokeStyle = '#cdd9dd';
    ctx.strokeRect(left,top,side,side);
    for (let i=0; i<=4; i++) {
      const x = left + side*i/4, y = top + side*(1-i/4);
      if (checked('grid')) {
        line([x,top],[x,top+side]); line([left,y],[left+side,y]);
      }
      const tick = axis => Number((bounds[axis][0] + i/4*(bounds[axis][1]-bounds[axis][0])).toPrecision(3));
      ctx.textAlign = 'center'; ctx.fillText(tick(0),x,top+side+16);
      ctx.textAlign = 'right'; ctx.fillText(tick(1),left-4,y+4);
    }
    ctx.textAlign = 'center'; ctx.fillText('Cecha 1',left+side/2,top+side+32);
    ctx.save(); ctx.translate(10,top+side/2); ctx.rotate(-Math.PI/2); ctx.fillText('Cecha 2',0,0); ctx.restore();
    ctx.save();
    ctx.beginPath(); ctx.rect(left,top,side,side); ctx.clip();
    const q = queries.at(-1);
    const indices = result?.indices || [];
    ctx.strokeStyle = '#53627a'; ctx.setLineDash([3,3]);
    if (checked('neighbors')) indices.forEach(i => line(pixel(q),pixel(data.points[i])));
    if (checked('radius') && result?.outline) {
      ctx.setLineDash([1,3]); ctx.beginPath();
      result.outline.forEach((p,i) => i ? ctx.lineTo(...pixel(p)) : ctx.moveTo(...pixel(p)));
      ctx.stroke();
    }
    ctx.setLineDash([]);
    const size = Math.sqrt(el('point_size').checkValidity() ? numeric('point_size') : 48)*1.8;
    const ring = (x,y,r) => {
      ctx.beginPath(); ctx.arc(x,y,r,0,2*Math.PI); ctx.strokeStyle = '#253247'; ctx.stroke();
    };
    ctx.textBaseline = 'middle'; ctx.font = size+'px system-ui';
    data.points.forEach((p,i) => {
      const [x,y] = pixel(p), label = data.labels[i];
      ctx.fillStyle = palette[label];
      ctx.strokeStyle = 'white'; ctx.lineWidth = 1.5;
      ctx.strokeText(symbols[label],x,y); ctx.fillText(symbols[label],x,y);
      ctx.lineWidth = 1;
      if (data.added_by_click[i]) ring(x,y,size*0.57);
    });
    if (checked('neighbors')) indices.forEach((idx,i) => {
      const [x,y] = pixel(data.points[idx]); ring(x,y,size*0.72);
      if (checked('ranks')) {
        ctx.fillStyle = '#182b39'; ctx.font = 'bold 11px system-ui'; ctx.fillText(i+1,x+9,y-9);
      }
    });
    result?.predictions.forEach((label,i) => {
      const [x,y] = pixel(queries[i]), latest = i === queries.length-1;
      ctx.globalAlpha = latest ? 1 : 0.65; ctx.font = (latest ? 32 : 21)+'px system-ui';
      ctx.strokeStyle = '#253247'; ctx.fillStyle = palette[label];
      ctx.strokeText('★',x,y); ctx.fillText('★',x,y);
    });
    ctx.restore();
    if (drag && el('mode').value === 'zoom') {
      ctx.strokeStyle = '#087e70';
      ctx.strokeRect(...drag.start, drag.end[0]-drag.start[0], drag.end[1]-drag.start[1]);
    }
    const counts = new Map();
    data.labels.forEach(l => counts.set(l,(counts.get(l)||0)+1));
    const entries = [...counts.keys()].sort().map(l => [palette[l],symbols[l]+' K'+l+' ('+counts.get(l)+')']);
    entries.push(['#253247','★ Zapytanie']);
    let x = 6, y = top+side+52;
    ctx.font = '11px system-ui'; ctx.textAlign = 'left';
    entries.forEach(([color,label]) => {
      const width = ctx.measureText(label).width + 12;
      if (x+width > rect.width) { x=6; y+=16; }
      ctx.fillStyle=color; ctx.fillText(label,x,y); x+=width;
    });
    el('summary').textContent = pattern+' · '+data.points.length+' pkt'+(result ? ' · k = '+result.k : '');
    el('undo').disabled = !edits.length;
    el('clear').disabled = !queries.length;
    el('back').disabled = viewIndex === 0;
    el('forward').disabled = viewIndex === views.length-1;
    el('save').disabled = !result;
  }

  function setView(bounds) {
    if (bounds.some(([a,b]) => !Number.isFinite(a+b) || b-a < 1e-8 || b-a > 1e8)) return;
    views.splice(viewIndex+1);
    views.push(bounds); viewIndex++; refresh(true);
  }

  function editAt(position) {
    const {point,pixel} = layout(), q = point(position);
    const mode = el('mode').value;
    if (mode === 'predict') {
      queries.push(q); if (queries.length > 100) queries.shift();
      refresh(); return;
    }
    if (mode === 'add') {
      if (data.points.length >= 1500) return status('Limit wynosi 1500 punktów treningowych.');
      edits.push({mode, index:data.points.length});
      data.points.push(q); data.labels.push(Number(el('label').value)); data.added_by_click.push(true);
    } else if (mode === 'remove') {
      if (data.points.length === 1) return status('Zachowaj co najmniej jeden punkt.');
      let index = -1, best = Infinity;
      data.points.forEach((p,i) => {
        const [x,y] = pixel(p), d = Math.hypot(x-position[0],y-position[1]);
        if (d < best) { best=d; index=i; }
      });
      if (best > 14) return status('Kliknij bliżej punktu treningowego.');
      edits.push({mode, index, point:data.points.splice(index,1)[0], label:data.labels.splice(index,1)[0],
        added:data.added_by_click.splice(index,1)[0]});
    } else return;
    if (edits.length > 100) edits.shift();
    refresh(true);
  }

  function position(event) {
    const rect = canvas.getBoundingClientRect();
    return [event.clientX-rect.left, event.clientY-rect.top];
  }
  canvas.onpointerdown = event => {
    if (!data || event.button !== 0 || !event.isPrimary) return;
    const [x,y] = position(event), {left,top,side} = layout();
    if (x<left || x>left+side || y<top || y>top+side) return;
    drag = {start:[x,y], end:[x,y]};
    canvas.setPointerCapture(event.pointerId);
  };
  canvas.onpointermove = event => {
    if (!drag || !event.isPrimary) return;
    drag.end = position(event);
    if (el('mode').value === 'zoom') draw();
  };
  canvas.onpointercancel = () => { drag=null; draw(); };
  canvas.onpointerup = event => {
    if (!drag || !event.isPrimary) return;
    const start = drag.start, end = position(event), {point} = layout();
    drag = null;
    const a = point(start), b = point(end), mode = el('mode').value;
    if (mode === 'pan') {
      setView(currentView().map((axis,i) => axis.map(v => v+a[i]-b[i])));
    } else if (mode === 'zoom') {
      const radius = Math.max(Math.abs(b[0]-a[0]),Math.abs(b[1]-a[1]))/2;
      if (Math.hypot(end[0]-start[0],end[1]-start[1]) > 4) {
        setView(a.map((v,i) => [(v+b[i])/2-radius,(v+b[i])/2+radius]));
      } else draw();
    } else if (Math.hypot(end[0]-start[0],end[1]-start[1]) < 6) editAt(start);
  };

  el('undo').onclick = () => {
    const edit = edits.pop(); if (!edit) return;
    if (edit.mode === 'add') {
      data.points.pop(); data.labels.pop(); data.added_by_click.pop();
    } else {
      data.points.splice(edit.index,0,edit.point); data.labels.splice(edit.index,0,edit.label);
      data.added_by_click.splice(edit.index,0,edit.added);
    }
    refresh(true);
  };
  el('clear').onclick = () => { queries=[]; refresh(); };
  el('home').onclick = () => { if (data) setView(fitView()); };
  el('back').onclick = () => { if (viewIndex>0) { viewIndex--; refresh(true); } };
  el('forward').onclick = () => { if (viewIndex+1<views.length) { viewIndex++; refresh(true); } };
  function download(blob, filename) {
    if (!blob) return;
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href=url; a.download=filename; a.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
  }
  el('export').onclick = () => {
    if (data) download(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),'knn_training_data.json');
  };
  el('save').onclick = () => { if (result) canvas.toBlob(blob => download(blob,'knn.png')); };
  el('generate').onclick = generate;
  el('random').onclick = () => { el('seed').value=Math.floor(Math.random()*1000000); generate(); };
  el('dataset').onchange = () => {
    const binary = ['XOR','Circles','Moons'].includes(el('dataset').value);
    el('n_classes').disabled=binary;
    if (binary) el('n_classes').value=2;
  };
  el('mode').onchange = () => { el('label-row').hidden=el('mode').value!=='add'; };
  el('model').oninput = () => { el('p').disabled=el('metric').value!=='minkowski'; refresh(true); };
  el('p').disabled=true;
  el('display').oninput = event => {
    if (!event.target.checkValidity()) return status('Sprawdź ustawienia wykresu.');
    if (event.target === el('resolution')) refresh(true);
    else if (event.target === el('regions') && checked('regions') && !background) refresh();
    else draw();
  };
  function enter() {
    if ($('knn').hidden) return;
    if (!data) generate(); else draw();
  }
  window.addEventListener('hashchange',enter);
  window.addEventListener('resize',draw);
  enter();
})();
