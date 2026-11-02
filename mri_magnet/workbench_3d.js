/* MRI Workbench v5 · native offline geometry viewer, no field calculation. */
(function (root) {
  'use strict';
  const colors = Object.freeze({out: '#AB2D35', in: '#1955B0', down: '#A56E27', up: '#22877A'});
  const polarityLabels = Object.freeze({out: 'N 极径向朝外', in: 'N 极径向朝内', down: 'N 极轴向 −Z', up: 'N 极轴向 +Z'});
  const TAU = Math.PI * 2;
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const finite = (x, fallback) => Number.isFinite(Number(x)) ? Number(x) : fallback;
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const normalize = a => scale(a, 1 / (Math.hypot(...a) || 1));
  const rgb = hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255);
  const north = (polarity, theta) => polarity === 'out' ? [Math.cos(theta), Math.sin(theta), 0] : polarity === 'in' ? [-Math.cos(theta), -Math.sin(theta), 0] : [0, 0, polarity === 'up' ? 1 : -1];

  function buildMagnets(design) {
    if (!design || !Array.isArray(design.rings)) return [];
    const cube = design.cube_mm === undefined ? 12 : Number(design.cube_mm);
    if (!Number.isFinite(cube) || !(cube > 0)) throw new Error('磁块边长必须为有限正数。');
    let total = 0;
    for (const ring of design.rings) {
      if (!ring || !Number.isInteger(ring.count) || ring.count < 1 || !Number.isFinite(ring.radius_mm) || ring.radius_mm <= 0 || !Number.isFinite(ring.z_mm) || !Number.isFinite(ring.phase_deg) || !colors[ring.polarity]) throw new Error('磁环几何或极性无效：' + (ring?.id || '未命名磁环'));
      total += ring.count;
      if (total > 50000) throw new Error('交互预览最多支持 50,000 块磁铁。');
    }
    const result = [];
    design.rings.forEach((ring, ringIndex) => {
      const ringId = String(ring.id || 'R' + String(ringIndex + 1).padStart(2, '0'));
      for (let index = 0; index < ring.count; index++) {
        const thetaDeg = ring.phase_deg + index * 360 / ring.count;
        const theta = thetaDeg * Math.PI / 180;
        result.push({ring_id: ringId, magnet_id: ringId + '-A' + String(index + 1).padStart(2, '0'), index,
          ring_index: ringIndex, layer_index: ring.layer_index ?? ringIndex, slot_index: ring.slot_index,
          z_mm: ring.z_mm, radius_mm: ring.radius_mm, bottom_radius_mm: ring.radius_mm - cube / 2,
          position_mm: [ring.radius_mm * Math.cos(theta), ring.radius_mm * Math.sin(theta), ring.z_mm],
          theta_deg: ((thetaDeg % 360) + 360) % 360, theta, polarity: ring.polarity,
          count: ring.count, cube_mm: cube, north_vector: north(ring.polarity, theta), color: colors[ring.polarity]});
      }
    });
    return result;
  }

  function displayCenter(magnet, explode) {
    const e = clamp(finite(explode, 0), 0, 1);
    return [magnet.position_mm[0] * (1 + .32 * e), magnet.position_mm[1] * (1 + .32 * e), magnet.position_mm[2] * (1 + .75 * e)];
  }
  const localCorners = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]];
  const faces = [{v:[0,3,2,1],n:[0,0,-1]}, {v:[4,5,6,7],n:[0,0,1]}, {v:[0,1,5,4],n:[0,-1,0]}, {v:[1,2,6,5],n:[1,0,0]}, {v:[2,3,7,6],n:[0,1,0]}, {v:[3,0,4,7],n:[-1,0,0]}];
  const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  function cubeVertices(magnet, explode = 0) {
    const c = Math.cos(magnet.theta), s = Math.sin(magnet.theta), h = magnet.cube_mm / 2, p = displayCenter(magnet, explode);
    return localCorners.map(v => [p[0] + h * (v[0] * c - v[1] * s), p[1] + h * (v[0] * s + v[1] * c), p[2] + h * v[2]]);
  }

  function rayCubeDistance(origin, direction, magnet, explode = 0) {
    const p = sub(origin, displayCenter(magnet, explode)), c = Math.cos(magnet.theta), s = Math.sin(magnet.theta);
    const o = [p[0] * c + p[1] * s, -p[0] * s + p[1] * c, p[2]];
    const d = [direction[0] * c + direction[1] * s, -direction[0] * s + direction[1] * c, direction[2]];
    const h = magnet.cube_mm / 2;
    let near = -Infinity, far = Infinity;
    for (let axis = 0; axis < 3; axis++) {
      if (Math.abs(d[axis]) < 1e-12) { if (o[axis] < -h || o[axis] > h) return null; }
      else {
        const a = (-h - o[axis]) / d[axis], b = (h - o[axis]) / d[axis];
        near = Math.max(near, Math.min(a, b)); far = Math.min(far, Math.max(a, b));
        if (near > far) return null;
      }
    }
    return far < 0 ? null : Math.max(0, near);
  }

  function pickMagnet(magnets, origin, direction, options = {}) {
    let hit = null, distance = Infinity;
    for (const magnet of magnets) {
      if (options.cutaway && magnet.position_mm[1] < -1e-8) continue;
      const t = rayCubeDistance(origin, direction, magnet, options.explode || 0);
      if (t !== null && t < distance) { hit = magnet; distance = t; }
    }
    return hit;
  }

  function cameraBasis(azimuth, elevation) {
    const a = Math.cos(azimuth), b = Math.sin(azimuth), c = Math.cos(elevation), d = Math.sin(elevation);
    return {right: [-b, a, 0], up: [-a * d, -b * d, c], towardEye: [a * c, b * c, d]};
  }

  class Viewer {
    constructor(canvas, callbacks = {}) {
      if (!canvas || typeof canvas.getContext !== 'function') throw new Error('3D 预览需要 Canvas 元素。');
      this.canvas = canvas; this.onInspect = callbacks.onInspect || (() => {}); this.onStatus = callbacks.onStatus || (() => {});
      this.options = {cubes:true, vectors:true, target:true, bore:true, gizmo:true, cutaway:false, explode:0};
      this.camera = {azimuth:-Math.PI / 3, elevation:.52, zoom:1, pan:[0,0]};
      this.config = {}; this.design = null; this.label = ''; this.magnets = []; this.displayMagnets = [];
      this.active = true; this.disposed = false; this.dirty = true; this.geometryDirty = true; this.frameCount = 0;
      this.selectedId = null; this.hoveredId = null; this.inspectedId = null; this.bound = 300; this.boundR=220; this.boundZ=250;
      this._handlers = []; this._frame = null; this._transition = null; this._geometrySignature = ''; this._pointer = null;
      this.width=1;this.height=1;this.dpr=1;this._pendingSize=null;this._appliedSize=null;
      this.reducedMotion = !!root.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
      this._initRenderer(); this._initOverlay(); this._bind();
      this._resizeObserver = typeof root.ResizeObserver === 'function' ? new root.ResizeObserver(() => this.resize()) : null;
      this._resizeObserver?.observe(canvas); this.resize(); this._status();
    }

    _status(error = '') {
      this.onStatus({renderer:this.mode, magnetCount:this.magnets.length, visibleCount:this.displayMagnets.filter(m => !this.options.cutaway || m.position_mm[1] >= -1e-8).length, label:this.label,
        referenceOnly:!this.magnets.length, message:error || this.geometryError || (this.mode === 'webgl' ? '离线 WebGL · 深度渲染' : '离线 Canvas · 兼容渲染')});
    }

    _initRenderer() {
      let gl = null;
      try { gl = this.canvas.getContext('webgl', {alpha:false, antialias:true, depth:true, preserveDrawingBuffer:false}); } catch (_) {}
      if (gl) {
        try {
          const vertex = 'attribute vec3 aPosition;attribute vec4 aColor;uniform vec3 uRight,uUp,uEye,uCenter,uScale;varying vec4 vColor;void main(){vec3 p=aPosition-uCenter;gl_Position=vec4(dot(p,uRight)*uScale.x,dot(p,uUp)*uScale.y,-dot(p,uEye)*uScale.z,1.0);vColor=aColor;}';
          const fragment = 'precision mediump float;varying vec4 vColor;void main(){gl_FragColor=vColor;}';
          const shader = (type, source) => { const s=gl.createShader(type);gl.shaderSource(s, source);gl.compileShader(s);if(!gl.getShaderParameter(s, gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s; };
          const vs=shader(gl.VERTEX_SHADER, vertex), fs=shader(gl.FRAGMENT_SHADER, fragment), program=gl.createProgram();
          gl.attachShader(program,vs);gl.attachShader(program,fs);gl.linkProgram(program);gl.deleteShader(vs);gl.deleteShader(fs);
          if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
          this.gl=gl;this.program=program;gl.useProgram(program);
          this.locations={position:gl.getAttribLocation(program,'aPosition'),color:gl.getAttribLocation(program,'aColor')};
          for(const name of ['Right','Up','Eye','Center','Scale'])this.locations[name]=gl.getUniformLocation(program,'u'+name);
          this.buffers={};for(const name of ['solids','edges','vectors','references','highlight'])this.buffers[name]={buffer:gl.createBuffer(),count:0};
          gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
          this.mode='webgl';
          return;
        } catch (_) {
          // A canvas with a WebGL context cannot be reused as a 2D context.
          this.gl=null;this._makeFallbackCanvas();
        }
      }
      this.ctx=this.drawCanvas?.getContext('2d') || this.canvas.getContext('2d');this.mode='canvas2d';
      if(!this.ctx){this._makeFallbackCanvas();this.ctx=this.drawCanvas.getContext('2d');}
      if(!this.ctx)throw new Error('浏览器不支持 WebGL 或 Canvas 2D。');
    }

    _makeFallbackCanvas() {
      if(this.drawCanvas || !this.canvas.ownerDocument || !this.canvas.parentElement)return;
      const fallback=this.canvas.ownerDocument.createElement('canvas');fallback.setAttribute('aria-hidden','true');
      fallback.style.cssText='position:absolute;pointer-events:none;display:block;border-radius:inherit;z-index:1;';
      this.canvas.parentElement.appendChild(fallback);this.drawCanvas=fallback;
    }

    _initOverlay() {
      if(!this.canvas.ownerDocument || !this.canvas.parentElement)return;
      this.overlay=this.canvas.ownerDocument.createElement('canvas');this.overlay.setAttribute('aria-hidden','true');
      this.overlay.style.cssText='position:absolute;pointer-events:none;display:block;z-index:2;';this.canvas.parentElement.appendChild(this.overlay);
      this.overlayContext=this.overlay.getContext('2d');
    }

    _listen(target, type, callback, options) { target.addEventListener(type,callback,options);this._handlers.push([target,type,callback,options]); }

    _bind() {
      this._listen(this.canvas,'contextmenu',e=>e.preventDefault());
      this._listen(this.canvas,'pointerdown',e=>{
        if(e.button!==0&&e.button!==2)return;e.preventDefault();this.canvas.focus?.({preventScroll:true});
        this._pointer={id:e.pointerId,x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,button:e.button};
        this.canvas.setPointerCapture?.(e.pointerId);this.canvas.style.cursor=e.button===2?'move':'grabbing';
      });
      this._listen(this.canvas,'pointermove',e=>{
        if(this._pointer&&this._pointer.id===e.pointerId){
          const dx=e.clientX-this._pointer.x,dy=e.clientY-this._pointer.y;this._pointer.x=e.clientX;this._pointer.y=e.clientY;
          if(this._pointer.button===0){this.camera.azimuth-=dx*.008;this.camera.elevation=clamp(this.camera.elevation+dy*.008,-Math.PI/2+.001,Math.PI/2-.001);}
          else{const k=2*this._view().halfHeight/this.height;this.camera.pan[0]-=dx*k;this.camera.pan[1]+=dy*k;}
          this.hoveredId=null;this._notify();this.invalidate();
        }else{
          this._hoverPoint=[e.clientX,e.clientY];
          if(!this._pickFrame)this._pickFrame=root.requestAnimationFrame(()=>{this._pickFrame=null;if(!this._hoverPoint||this.disposed)return;const m=this._pickClient(...this._hoverPoint);this.hoveredId=m?.magnet_id||null;this.canvas.style.cursor=m?'pointer':'grab';this._notify();this.invalidate();});
        }
      });
      const release=e=>{
        if(!this._pointer||this._pointer.id!==e.pointerId)return;
        const pointer=this._pointer;this._pointer=null;this.canvas.style.cursor='grab';
        if(pointer.button===0&&Math.hypot(e.clientX-pointer.startX,e.clientY-pointer.startY)<5){
          const m=this._pickClient(e.clientX,e.clientY);this.selectedId=m?.magnet_id||null;this.hoveredId=null;this._notify(true);this.invalidate();
        }
        try{this.canvas.releasePointerCapture?.(e.pointerId);}catch(_){}
      };
      this._listen(this.canvas,'pointerup',release);
      this._listen(this.canvas,'pointercancel',e=>{this._pointer=null;this.canvas.style.cursor='grab';});
      this._listen(this.canvas,'pointerleave',()=>{this._hoverPoint=null;this.hoveredId=null;this._notify();this.invalidate();});
      this._listen(this.canvas,'wheel',e=>{e.preventDefault();this.camera.zoom=clamp(this.camera.zoom*Math.exp(-e.deltaY*.0012),.18,8);this.invalidate();},{passive:false});
      this._listen(this.canvas,'keydown',e=>{
        const keys={ArrowLeft:[-.08,0],ArrowRight:[.08,0],ArrowUp:[0,.08],ArrowDown:[0,-.08]};
        if(keys[e.key]){e.preventDefault();this.camera.azimuth+=keys[e.key][0];this.camera.elevation=clamp(this.camera.elevation+keys[e.key][1],-1.569,1.569);this.invalidate();}
        else if(e.key==='+'||e.key==='='){e.preventDefault();this.camera.zoom=clamp(this.camera.zoom*1.15,.18,8);this.invalidate();}
        else if(e.key==='-'){e.preventDefault();this.camera.zoom=clamp(this.camera.zoom/1.15,.18,8);this.invalidate();}
        else if(e.key==='Escape'){this.clearSelection();}
        else if(e.key.toLowerCase()==='r'){this.setView('reset');}
      });
      this._listen(this.canvas,'webglcontextlost',e=>{e.preventDefault();this.gl=null;this._makeFallbackCanvas();this.ctx=this.drawCanvas?.getContext('2d');this.mode='canvas2d';this.geometryDirty=true;this.resize();this._status('WebGL 上下文中断，已切换离线 Canvas 兼容渲染。');});
      if(root.addEventListener)this._listen(root,'resize',()=>this.resize(),{passive:true});
    }

    update({design=null,config={},label=''}={}) {
      if(this.disposed)return;
      const signature=JSON.stringify({rings:design?.rings,cube:design?.cube_mm,config:{envelope:config.envelope,target:config.target,manufacturing:config.manufacturing}});
      this.label=String(label||design?.name||'包络与成像区域');this.config=config||{};
      if(signature===this._geometrySignature){this._status();return;}
      this._geometrySignature=signature;
      let next;this.geometryError='';
      try{next=buildMagnets(design);}catch(error){next=[];this.geometryError=error.message;}
      const old=new Map(this.displayMagnets.map(m=>[m.magnet_id,m]));
      this.magnets=next;this.design=design;
      const envelope=this.config.envelope||{},r=finite(envelope.max_outer_diameter_mm,440)/2,z=finite(envelope.max_length_mm,500)/2;
      const actualR=next.reduce((v,m)=>Math.max(v,m.radius_mm+m.cube_mm),0),actualZ=next.reduce((v,m)=>Math.max(v,Math.abs(m.z_mm)+m.cube_mm),0);
      this.boundR=Math.max(20,r,actualR);this.boundZ=Math.max(20,z,actualZ);
      this.bound=Math.max(40,Math.hypot(this.boundR,this.boundZ));
      const ids=new Set(next.map(m=>m.magnet_id));if(!ids.has(this.selectedId))this.selectedId=null;if(!ids.has(this.hoveredId))this.hoveredId=null;
      this._transition=old.size&&next.length&&!this.reducedMotion?{start:root.performance.now(),old,duration:320}:null;
      this.displayMagnets=next;this.geometryDirty=true;this._notify(true);this._status();this.invalidate();
    }

    setOptions(values={}) {
      let geometry=false;
      for(const key of Object.keys(this.options))if(Object.prototype.hasOwnProperty.call(values,key)){
        const next=key==='explode'?clamp(finite(values[key],0),0,1):Boolean(values[key]);
        if(this.options[key]!==next){this.options[key]=next;if(!['gizmo'].includes(key))geometry=true;}
      }
      if(this.options.cutaway){const lookup=id=>this.magnets.find(m=>m.magnet_id===id);if(lookup(this.selectedId)?.position_mm[1]<-1e-8)this.selectedId=null;if(lookup(this.hoveredId)?.position_mm[1]<-1e-8)this.hoveredId=null;}
      this.geometryDirty ||= geometry;this._notify();this._status();this.invalidate();
    }

    setView(view) {
      if(view==='top'){this.camera.azimuth=-Math.PI/2;this.camera.elevation=Math.PI/2-.00001;}
      else if(view==='side'){this.camera.azimuth=-Math.PI/2;this.camera.elevation=0;}
      else{this.camera.azimuth=-Math.PI/3;this.camera.elevation=.52;}
      this.camera.pan=[0,0];this.camera.zoom=1;this.invalidate();
    }

    clearSelection(){this.selectedId=null;this.hoveredId=null;this.inspectedId=null;this.onInspect(null,{pinned:false});this.invalidate();}
    setActive(value){const was=this.active;this.active=Boolean(value);if(this.active){this.resize();if(!was||!this.frameCount||this._transition)this.invalidate();}else if(this._frame){root.cancelAnimationFrame(this._frame);this._frame=null;}}

    resize(){
      if(this.disposed)return;
      const rect=this.canvas.getBoundingClientRect();
      if(!(rect.width>0&&rect.height>0))return;
      const next={width:rect.width,height:rect.height,dpr:clamp(root.devicePixelRatio||1,1,2)},last=this._pendingSize||this._appliedSize;
      if(last&&last.width===next.width&&last.height===next.height&&last.dpr===next.dpr)return;
      this._pendingSize=next;this.invalidate();
    }

    _applyPendingSize(){
      const next=this._pendingSize;if(!next)return;this._pendingSize=null;
      this.width=next.width;this.height=next.height;this.dpr=next.dpr;
      for(const c of [this.canvas,this.drawCanvas,this.overlay].filter(Boolean)){
        const w=Math.round(this.width*this.dpr),h=Math.round(this.height*this.dpr);if(c.width!==w)c.width=w;if(c.height!==h)c.height=h;
        if(c!==this.canvas){c.style.width=this.width+'px';c.style.height=this.height+'px';c.style.left=this.canvas.offsetLeft+'px';c.style.top=this.canvas.offsetTop+'px';}
      }
      this._appliedSize=next;
    }

    invalidate(){this.dirty=true;if(this.active&&!this.disposed&&!this._frame)this._frame=root.requestAnimationFrame(t=>this._render(t));}

    _view(){
      const basis=cameraBasis(this.camera.azimuth,this.camera.elevation),aspect=this.width/this.height;
      const effectiveBound=Math.hypot(this.boundR*(1+.32*this.options.explode),this.boundZ*(1+.75*this.options.explode));
      const halfHeight=effectiveBound*1.075*Math.max(1,1/aspect)/this.camera.zoom;
      return {...basis,halfHeight,halfWidth:halfHeight*aspect,depth:effectiveBound*7,
        center:add(scale(basis.right,this.camera.pan[0]),scale(basis.up,this.camera.pan[1]))};
    }

    _project(point, view=this._view()) {
      const p=sub(point,view.center);
      return {x:this.width*(.5+dot(p,view.right)/(2*view.halfWidth)),y:this.height*(.5-dot(p,view.up)/(2*view.halfHeight)),depth:dot(p,view.towardEye)};
    }

    _pickClient(x,y){const rect=this.canvas.getBoundingClientRect();return this.pick(x-rect.left,y-rect.top);}
    pick(x,y){
      if(x<0||y<0||x>this.width||y>this.height)return null;
      const v=this._view(),p=add(v.center,add(scale(v.right,(x/this.width-.5)*2*v.halfWidth),scale(v.up,(.5-y/this.height)*2*v.halfHeight)));
      return pickMagnet(this.displayMagnets,add(p,scale(v.towardEye,v.depth)),scale(v.towardEye,-1),this.options);
    }

    _notify(force=false){
      const id=this.hoveredId||this.selectedId,magnet=this.magnets.find(m=>m.magnet_id===id)||null;
      if(force||id!==this.inspectedId){this.inspectedId=id;this.onInspect(magnet?{...magnet,position_mm:magnet.position_mm.slice(),north_vector:magnet.north_vector.slice()}:null,{pinned:!!this.selectedId&&id===this.selectedId});}
    }

    _referenceLines(){
      const lines=[],put=(a,b,color)=>lines.push({a,b,color});
      const ring=(radius,z,color,segments=72)=>{for(let i=0;i<segments;i++){const a=i*TAU/segments,b=(i+1)*TAU/segments;put([radius*Math.cos(a),radius*Math.sin(a),z],[radius*Math.cos(b),radius*Math.sin(b),z],color);}};
      const envelope=this.config.envelope||{},target=this.config.target||{},r=finite(envelope.protected_bore_diameter_mm,200)/2,outer=finite(envelope.max_outer_diameter_mm,440)/2,half=finite(envelope.max_length_mm,500)/2;
      if(this.options.bore){
        const pale=[.49,.61,.69,.24],outerColor=[.60,.67,.72,.22],seam=[.40,.55,.62,.30];
        for(const z of [-half,0,half]){ring(r,z,pale);ring(outer,z,outerColor);}
        for(let i=0;i<16;i++){const a=i*TAU/16;put([r*Math.cos(a),r*Math.sin(a),-half],[r*Math.cos(a),r*Math.sin(a),half],pale);}
        const sectors=clamp(Math.round(finite(this.config.manufacturing?.sectors_per_layer,4)),1,64);
        for(let i=0;i<sectors;i++){const a=i*TAU/sectors,c=Math.cos(a),s=Math.sin(a);for(const z of [-half,half])put([r*c,r*s,z],[outer*c,outer*s,z],seam);put([outer*c,outer*s,-half],[outer*c,outer*s,half],seam);}
      }
      if(this.options.target){
        const radius=Math.max(.001,finite(target.diameter_mm,20)/2),color=[.15,.61,.47,.47],z=finite(target.z_mm,0);
        if(target.geometry==='disk'){
          ring(radius,z,color,96);put([-radius,0,z],[radius,0,z],color);put([0,-radius,z],[0,radius,z],color);
          for(const fraction of [.33,.66])ring(radius*fraction,z,[.15,.61,.47,.18]);
        }else{
          for(let j=-3;j<=3;j++){const latitude=j*Math.PI/8;ring(radius*Math.cos(latitude),radius*Math.sin(latitude),color);}
          for(let j=0;j<8;j++){const longitude=j*Math.PI/8;for(let k=0;k<72;k++){const a=k*TAU/72,b=(k+1)*TAU/72;put([radius*Math.cos(a)*Math.cos(longitude),radius*Math.cos(a)*Math.sin(longitude),radius*Math.sin(a)],[radius*Math.cos(b)*Math.cos(longitude),radius*Math.cos(b)*Math.sin(longitude),radius*Math.sin(b)],color);}}
        }
      }
      return lines;
    }

    _arrows(magnet){
      const p=displayCenter(magnet,this.options.explode),v=north(magnet.polarity,magnet.theta),length=magnet.cube_mm*.88,tip=add(p,scale(v,length));
      const crossAxis=Math.abs(v[2])>.9?[1,0,0]:[0,0,1],side=normalize(cross(v,crossAxis)),base=add(tip,scale(v,-length*.28)),wing=length*.19;
      return [[p,tip],[tip,add(base,scale(side,wing))],[tip,add(base,scale(side,-wing))]];
    }

    _buildBuffers(){
      const solids=[],outlines=[],vectors=[],references=[],push=(arr,p,c)=>arr.push(p[0],p[1],p[2],c[0],c[1],c[2],c[3]??1),line=(arr,a,b,c)=>{push(arr,a,c);push(arr,b,c);};
      const light=normalize([-.35,-.6,1]);
      for(const m of this.displayMagnets){
        if(this.options.cutaway&&m.position_mm[1]<-1e-8)continue;
        const v=cubeVertices(m,this.options.explode),base=rgb(m.color),c=Math.cos(m.theta),s=Math.sin(m.theta);
        if(this.options.cubes){
          for(const face of faces){const n=[face.n[0]*c-face.n[1]*s,face.n[0]*s+face.n[1]*c,face.n[2]],shade=.62+.38*Math.max(0,dot(n,light)),color=base.map(x=>x*shade).concat(1);
            for(const index of [face.v[0],face.v[1],face.v[2],face.v[0],face.v[2],face.v[3]])push(solids,v[index],color);}
          for(const edge of edges)line(outlines,v[edge[0]],v[edge[1]],[base[0]*.45,base[1]*.45,base[2]*.45,.75]);
        }
        if(this.options.vectors)for(const segment of this._arrows(m))line(vectors,segment[0],segment[1],this.options.cubes?[.97,.98,1,.96]:base.concat(.95));
      }
      for(const item of this._referenceLines())line(references,item.a,item.b,item.color);
      const gl=this.gl;
      for(const [name,array] of Object.entries({solids,edges:outlines,vectors,references})){
        const record=this.buffers[name];gl.bindBuffer(gl.ARRAY_BUFFER,record.buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(array),gl.DYNAMIC_DRAW);record.count=array.length/7;
      }
    }

    _drawGL(){
      const gl=this.gl,v=this._view();gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.clearColor(.981,.986,.990,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(this.program);
      for(const [name,value] of Object.entries({Right:v.right,Up:v.up,Eye:v.towardEye,Center:v.center,Scale:[1/v.halfWidth,1/v.halfHeight,1/v.depth]}))gl.uniform3fv(this.locations[name],value);
      const draw=(name,type)=>{const item=this.buffers[name];if(!item.count)return;gl.bindBuffer(gl.ARRAY_BUFFER,item.buffer);gl.enableVertexAttribArray(this.locations.position);gl.vertexAttribPointer(this.locations.position,3,gl.FLOAT,false,28,0);gl.enableVertexAttribArray(this.locations.color);gl.vertexAttribPointer(this.locations.color,4,gl.FLOAT,false,28,12);gl.drawArrays(type,0,item.count);};
      draw('solids',gl.TRIANGLES);draw('edges',gl.LINES);gl.depthMask(false);draw('references',gl.LINES);draw('vectors',gl.LINES);gl.depthMask(true);
      const id=this.hoveredId||this.selectedId,active=this.displayMagnets.find(m=>m.magnet_id===id);
      if(active){
        const array=[],put=(p,color)=>array.push(...p,...color);
        for(const magnet of this.displayMagnets){if(magnet.ring_id!==active.ring_id||(this.options.cutaway&&magnet.position_mm[1]<-1e-8))continue;const vertices=cubeVertices(magnet,this.options.explode),color=magnet.magnet_id===id?[1,.69,.20,1]:[.43,.83,.78,.9];for(const edge of edges){put(vertices[edge[0]],color);put(vertices[edge[1]],color);}}
        const item=this.buffers.highlight;gl.bindBuffer(gl.ARRAY_BUFFER,item.buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(array),gl.DYNAMIC_DRAW);item.count=array.length/7;
        gl.disable(gl.DEPTH_TEST);draw('highlight',gl.LINES);gl.enable(gl.DEPTH_TEST);
      }
    }

    _draw2D(){
      const ctx=this.ctx,view=this._view();ctx.setTransform(this.dpr,0,0,this.dpr,0,0);ctx.fillStyle='#fafcfd';ctx.fillRect(0,0,this.width,this.height);
      const projected=[];
      for(const m of this.displayMagnets){
        if(this.options.cutaway&&m.position_mm[1]<-1e-8)continue;
        const vertices=cubeVertices(m,this.options.explode),points=vertices.map(p=>this._project(p,view)),c=Math.cos(m.theta),s=Math.sin(m.theta),base=rgb(m.color);
        if(this.options.cubes)for(const face of faces){const n=[face.n[0]*c-face.n[1]*s,face.n[0]*s+face.n[1]*c,face.n[2]];if(dot(n,view.towardEye)<=0)continue;const shade=.62+.38*Math.max(0,dot(n,normalize([-.35,-.6,1])));projected.push({points:face.v.map(i=>points[i]),depth:face.v.reduce((sum,i)=>sum+points[i].depth,0)/4,color:base.map(x=>Math.round(x*shade*255)),magnet:m});}
      }
      projected.sort((a,b)=>a.depth-b.depth);
      const active=this.displayMagnets.find(m=>m.magnet_id===(this.hoveredId||this.selectedId));
      for(const face of projected){ctx.beginPath();face.points.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();ctx.fillStyle='rgb('+face.color.join(',')+')';ctx.fill();ctx.strokeStyle=active?.magnet_id===face.magnet.magnet_id?'#ffc059':active?.ring_id===face.magnet.ring_id?'#70d5c5':'#20334966';ctx.lineWidth=active?.magnet_id===face.magnet.magnet_id?1.8:active?.ring_id===face.magnet.ring_id?1.1:.45;ctx.stroke();}
      const stroke=(a,b,color,width=.7)=>{const p=this._project(a,view),q=this._project(b,view);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();};
      for(const item of this._referenceLines())stroke(item.a,item.b,`rgba(${item.color.slice(0,3).map(x=>Math.round(x*255)).join(',')},${item.color[3]})`);
      if(this.options.vectors)for(const m of this.displayMagnets){if(this.options.cutaway&&m.position_mm[1]<-1e-8)continue;const centerDepth=this._project(displayCenter(m,this.options.explode),view).depth;
        // The compatible painter only draws foreground arrows, avoiding an unreadable X-ray overlay.
        if(this.options.cubes&&centerDepth<0)continue;
        for(const [a,b] of this._arrows(m))stroke(a,b,this.options.cubes?'#ffffffdf':m.color,.85);
      }
    }

    _drawOverlay(){
      const ctx=this.overlayContext;if(!ctx)return;ctx.setTransform(this.dpr,0,0,this.dpr,0,0);ctx.clearRect(0,0,this.width,this.height);
      if(this.options.gizmo){const v=this._view(),x=this.width-53,y=this.height-55,size=30;ctx.fillStyle='#ffffffcc';ctx.beginPath();ctx.arc(x,y,44,0,TAU);ctx.fill();ctx.font='11px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
        for(const [axis,color,name] of [[[1,0,0],'#bf6170','X'],[[0,1,0],'#5c9d82','Y'],[[0,0,1],'#5b8db6','Z']]){const dx=dot(axis,v.right)*size,dy=-dot(axis,v.up)*size;ctx.strokeStyle=color;ctx.lineWidth=1.6;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+dx,y+dy);ctx.stroke();ctx.fillStyle=color;ctx.fillText(name,x+dx*1.22,y+dy*1.22);}}
      ctx.fillStyle='#8a9aa8';ctx.font='10px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif';ctx.textAlign='left';ctx.fillText(this.magnets.length?'真实尺寸几何 · mm':'包络参考预览 · 尚无磁体排布',14,this.height-15);
      if(this.options.explode>0){ctx.fillStyle='#a87c41';ctx.fillText('爆炸展开仅用于观察，检查器显示原始工程尺寸',14,20);}
    }

    _render(time){
      this._frame=null;if(!this.active||this.disposed)return;
      this._applyPendingSize();
      let animate=false;
      if(this._transition){const fraction=clamp((time-this._transition.start)/this._transition.duration,0,1),t=fraction*fraction*(3-2*fraction),old=this._transition.old;
        this.displayMagnets=this.magnets.map(m=>{const source=old.get(m.magnet_id);if(!source)return m;const theta=source.theta+Math.atan2(Math.sin(m.theta-source.theta),Math.cos(m.theta-source.theta))*t;return {...m,theta,position_mm:m.position_mm.map((p,i)=>source.position_mm[i]+(p-source.position_mm[i])*t)};});
        this.geometryDirty=true;animate=fraction<1;if(!animate){this._transition=null;this.displayMagnets=this.magnets;}
      }
      if(this.gl){if(this.geometryDirty)this._buildBuffers();this._drawGL();}else this._draw2D();
      this._drawOverlay();this.geometryDirty=false;this.dirty=false;this.frameCount++;
      if(animate)this.invalidate();
    }

    diagnostics(){
      const visible=this.displayMagnets.filter(m=>!this.options.cutaway||m.position_mm[1]>=-1e-8),view=this._view();
      return {renderer:this.mode,label:this.label,magnetCount:this.magnets.length,visibleCount:visible.length,ringCount:new Set(this.magnets.map(m=>m.ring_id)).size,frameCount:this.frameCount,
        camera:{azimuth:this.camera.azimuth,elevation:this.camera.elevation,zoom:this.camera.zoom,pan:this.camera.pan.slice()},options:{...this.options},selection:this.selectedId,hover:this.hoveredId,
        canvas:{width:this.width,height:this.height,dpr:this.dpr},coordinateSpace:'projectedMagnets x/y are CSS pixels local to the canvas',
        projectedMagnets:visible.map(m=>{const point=this._project(displayCenter(m,this.options.explode),view);return {x:point.x,y:point.y,depth:point.depth,magnet_id:m.magnet_id,ring_id:m.ring_id,visible:point.x>=0&&point.x<=this.width&&point.y>=0&&point.y<=this.height};}).sort((a,b)=>b.depth-a.depth),
        glError:this.gl?this.gl.getError():0,animating:!!this._transition};
    }

    dispose(){
      this.disposed=true;if(this._frame)root.cancelAnimationFrame(this._frame);if(this._pickFrame)root.cancelAnimationFrame(this._pickFrame);this._resizeObserver?.disconnect();
      for(const [target,type,callback,options] of this._handlers)target.removeEventListener(type,callback,options);
      if(this.gl){for(const record of Object.values(this.buffers||{}))this.gl.deleteBuffer(record.buffer);this.gl.deleteProgram(this.program);}
      this.overlay?.remove();this.drawCanvas?.remove();this._handlers=[];
    }
  }

  const api={Viewer,colors,polarityLabels,buildMagnets,cubeVertices,displayCenter,rayCubeDistance,pickMagnet,cameraBasis};
  root.MRIMagnet3D=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window!=='undefined'?window:globalThis);
