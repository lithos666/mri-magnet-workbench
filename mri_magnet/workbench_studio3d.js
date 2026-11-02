/* MRI Workbench v5: N/S half-cubes from the original MRI ring explorer. */
import * as THREE from './vendor/three.module.js';
import {OrbitControls} from './vendor/OrbitControls.js';
import {setupStudio,MRI_REFERENCE_STYLE} from './workbench_studio_lighting.js';

const TAU=Math.PI*2;
const clamp=(value,lo,hi)=>Math.max(lo,Math.min(hi,value));
const finite=(value,fallback)=>Number.isFinite(Number(value))?Number(value):fallback;
const COLORS=Object.freeze({out:'#AB2D35',in:'#1955B0',down:'#A56E27',up:'#22877A'});
const PRESENTATION=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),-Math.PI/2);
const EDGES=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];

// The presentation rotates the complete assembly, never its engineering data.
export function engineeringToWorld(point){return new THREE.Vector3(...point).applyQuaternion(PRESENTATION).toArray();}
export function displayCenter(magnet,explode=0){const e=clamp(finite(explode,0),0,1);return [magnet.position_mm[0]*(1+.32*e),magnet.position_mm[1]*(1+.32*e),magnet.position_mm[2]*(1+.75*e)];}

function disposeObject(object){object.traverse(child=>{child.geometry?.dispose();for(const material of Array.isArray(child.material)?child.material:[child.material])material?.dispose();});}

export function makePoleMaterial(pole){
  return new THREE.MeshStandardMaterial({color:pole==='north'?MRI_REFERENCE_STYLE.north:MRI_REFERENCE_STYLE.south,
    metalness:MRI_REFERENCE_STYLE.metalness,roughness:MRI_REFERENCE_STYLE.roughness});
}
export function poleHalfSpec(magnet,pole,explode=0){
  const center=displayCenter(magnet,explode),sign=pole==='north'?1:-1,a=magnet.cube_mm;
  const radial=magnet.polarity==='out'||magnet.polarity==='in',sense=magnet.polarity==='in'?-1:1;
  const north=radial?[sense*Math.cos(magnet.theta),sense*Math.sin(magnet.theta),0]:[0,0,magnet.polarity==='up'?1:-1];
  const scale=[a,a,a];scale[magnet.polarity==='out'||magnet.polarity==='in'?0:2]=a/2;
  return {center:center.map((v,i)=>v+sign*north[i]*a/4),scale,theta:magnet.theta};
}
function replacementCanvas(canvas){const replacement=canvas.cloneNode(false);replacement.removeAttribute('width');replacement.removeAttribute('height');canvas.replaceWith(replacement);return replacement;}

export class StudioViewer {
  constructor(canvas,callbacks={}){
    let context=null;
    try{context=canvas.getContext('webgl2',{alpha:true,antialias:true,depth:true});}catch{}
    if(!context)return new window.MRIMagnet3D.Viewer(canvas,callbacks);
    this.canvas=canvas;this.onInspect=callbacks.onInspect||(()=>{});this.onStatus=callbacks.onStatus||(()=>{});
    this.options={cubes:true,vectors:true,target:true,bore:true,gizmo:true,cutaway:false,explode:0};
    this.magnets=[];this.displayMagnets=[];this.design=null;this.config={};this.label='';
    this.active=true;this.disposed=false;this.frameCount=0;this.selectedId=null;this.hoveredId=null;this.inspectedId=null;
    this._listeners=[];this._frame=null;this._transition=null;this._signature='';this._hasFit=false;this._homeDistance=1000;
    this.width=1;this.height=1;this.dpr=1;this._pendingSize=null;this._appliedSize=null;this.bufferResizeCount=0;
    this.reducedMotion=!!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    this.materials={north:makePoleMaterial('north'),south:makePoleMaterial('south')};
    this.geometry=new THREE.BoxGeometry(1,1,1);this._meshes=[];
    try{
      this.renderer=new THREE.WebGLRenderer({canvas,context,alpha:true,antialias:true});
      this.scene=new THREE.Scene();this.studio=setupStudio(this.renderer,this.scene);
      this.camera=new THREE.PerspectiveCamera(MRI_REFERENCE_STYLE.fov,1,.1,50000);this.camera.up.set(0,1,0);this.camera.position.set(640,620,940);
      this.assembly=new THREE.Group();this.assembly.quaternion.copy(PRESENTATION);this.scene.add(this.assembly);
      this.referenceGroup=new THREE.Group();this.vectorGroup=new THREE.Group();this.highlightGroup=new THREE.Group();
      this.assembly.add(this.referenceGroup,this.vectorGroup,this.highlightGroup);
      this.controls=new OrbitControls(this.camera,canvas);this.controls.enableDamping=true;this.controls.dampingFactor=MRI_REFERENCE_STYLE.damping;
      this.controls.screenSpacePanning=true;this.controls.maxDistance=50000;this.controls.minDistance=15;
      this.raycaster=new THREE.Raycaster();this.pointer=new THREE.Vector2();this._matrix=new THREE.Matrix4();this._rotation=new THREE.Quaternion();
      this._scale=new THREE.Vector3();this._position=new THREE.Vector3();this._color=new THREE.Color();
      this._installOverlay();this._bind();this._resizeObserver=new ResizeObserver(()=>this.resize());this._resizeObserver.observe(canvas);
      this.resize();this._status();
    }catch(error){
      this.controls?.dispose();this.studio?.dispose();this.renderer?.dispose();this.overlay?.remove();
      return new window.MRIMagnet3D.Viewer(replacementCanvas(canvas),callbacks);
    }
  }

  _listen(target,type,callback,options){target.addEventListener(type,callback,options);this._listeners.push([target,type,callback,options]);}
  _status(message=''){
    this.onStatus({renderer:'webgl',renderStyle:'mri-ring-explorer',magnetCount:this.magnets.length,visibleCount:this._visibleMagnets().length,
      label:this.label,referenceOnly:!this.magnets.length,message:message||'MRI 磁环网页 · N 红 / S 蓝'});
  }
  _installOverlay(){
    this.overlay=this.canvas.ownerDocument.createElement('canvas');this.overlay.setAttribute('aria-hidden','true');
    this.overlay.style.cssText='position:absolute;pointer-events:none;display:block;z-index:1;';
    this.canvas.parentElement.appendChild(this.overlay);this.overlayContext=this.overlay.getContext('2d');
  }
  _bind(){
    this._listen(this.controls,'change',()=>this.invalidate());
    this._listen(this.canvas,'contextmenu',event=>event.preventDefault());
    this._listen(this.canvas,'pointerdown',event=>{if(event.button===0||event.button===2)this._pointerDown={x:event.clientX,y:event.clientY,button:event.button};});
    this._listen(this.canvas,'pointermove',event=>{
      if(event.buttons){this.hoveredId=null;this._notify();return;}
      this._hover=[event.clientX,event.clientY];if(this._pickFrame)return;
      this._pickFrame=requestAnimationFrame(()=>{this._pickFrame=null;if(!this._hover||this.disposed)return;const magnet=this._pickClient(...this._hover);
        this.hoveredId=magnet?.magnet_id||null;this.canvas.style.cursor=magnet?'pointer':'grab';this._notify();this._updateHighlight();this.invalidate();});
    });
    this._listen(this.canvas,'pointerleave',()=>{this._hover=null;this.hoveredId=null;this._notify();this._updateHighlight();this.invalidate();});
    this._listen(this.canvas,'pointerup',event=>{
      const start=this._pointerDown;this._pointerDown=null;if(!start||start.button!==0||Math.hypot(start.x-event.clientX,start.y-event.clientY)>5)return;
      this.selectedId=this._pickClient(event.clientX,event.clientY)?.magnet_id||null;this.hoveredId=null;this._notify(true);this._updateHighlight();this.invalidate();
    });
    this._listen(this.canvas,'keydown',event=>{
      if(event.key==='Escape')this.clearSelection();else if(event.key.toLowerCase()==='r')this.setView('reset');
      else if(event.key==='+'||event.key==='='||event.key==='-'){event.preventDefault();const factor=event.key==='-'?1.15:1/1.15;this.camera.position.sub(this.controls.target).multiplyScalar(factor).add(this.controls.target);this.controls.update();this.invalidate();}
    });
    this._listen(this.canvas,'webglcontextlost',event=>{event.preventDefault();this._activateFallback();});
  }

  _activateFallback(){
    if(this._fallback||this.disposed)return;
    if(this._frame)cancelAnimationFrame(this._frame);if(this._pickFrame)cancelAnimationFrame(this._pickFrame);
    this._resizeObserver?.disconnect();this.controls.dispose();for(const [target,type,fn,options]of this._listeners)target.removeEventListener(type,fn,options);this._listeners=[];
    this.overlay?.remove();const canvas=replacementCanvas(this.canvas),getContext=canvas.getContext.bind(canvas);
    // A fresh 2D canvas is necessary: an existing WebGL2 canvas cannot change type.
    canvas.getContext=(type,...args)=>/^webgl/.test(type)?null:getContext(type,...args);
    this.canvas=canvas;this._fallback=new window.MRIMagnet3D.Viewer(canvas,{onInspect:this.onInspect,onStatus:this.onStatus});
    this._fallback.update({design:this.design,config:this.config,label:this.label});this._fallback.setOptions(this.options);this._fallback.setActive(this.active);
    this.onStatus({renderer:'canvas2d',message:'图形设备中断，已切换本地兼容视图'});
  }

  update({design=null,config={},label=''}={}){
    if(this._fallback){this._fallback.update({design,config,label});this.magnets=this._fallback.magnets;this.displayMagnets=this._fallback.displayMagnets;return;}
    if(this.disposed)return;
    const previousLabel=this.label,previousHasModel=this.magnets.length>0,previousTopology=this.magnets.map(m=>m.magnet_id).join('|');
    this.design=design;this.config=config||{};this.label=String(label||design?.name||'包络参考');
    const signature=JSON.stringify({rings:design?.rings,cube:design?.cube_mm,envelope:config.envelope,target:config.target,manufacturing:config.manufacturing});
    if(signature===this._signature){this._status();return;}this._signature=signature;
    const previous=new Map(this.displayMagnets.map(m=>[m.magnet_id,m]));
    try{this.magnets=window.MRIMagnet3D.buildMagnets(design);}catch(error){this.magnets=[];this._status(error.message);}
    this.displayMagnets=this.magnets;
    const ids=new Set(this.magnets.map(m=>m.magnet_id));if(!ids.has(this.selectedId))this.selectedId=null;if(!ids.has(this.hoveredId))this.hoveredId=null;
    this._transition=previous.size&&this.magnets.length&&!this.reducedMotion?{previous,start:performance.now(),duration:320}:null;
    this._rebuildInstances();this._rebuildReferences();this._updateMatrices();this._notify(true);this._status();
    if(!this._hasFit){this._fit('iso');this._hasFit=true;}
    else if(previousHasModel!==(this.magnets.length>0)||previousLabel!==this.label||previousTopology!==this.magnets.map(m=>m.magnet_id).join('|'))this._fit(null,true);
    this.invalidate();
  }

  _visibleMagnets(){return this.displayMagnets.filter(m=>!this.options.cutaway||m.position_mm[1]>=-1e-8);}
  _rebuildInstances(){
    for(const mesh of this._meshes){this.assembly.remove(mesh);mesh.dispose();}this._meshes=[];
    for(const polarity of Object.keys(COLORS)){
      const indices=[];this.magnets.forEach((m,index)=>{if(m.polarity===polarity&&(!this.options.cutaway||m.position_mm[1]>=-1e-8))indices.push(index);});
      if(!indices.length)continue;
      for(const pole of ['north','south']){
        const mesh=new THREE.InstancedMesh(this.geometry,this.materials[pole],indices.length);
        mesh.name='磁块 · '+polarity+' · '+pole;mesh.userData={indices,pole};
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);mesh.visible=this.options.cubes;
        this.assembly.add(mesh);this._meshes.push(mesh);
      }
    }
    this._updateVectors();
  }
  _updateMatrices(){
    const active=this.hoveredId||this.selectedId,selected=this.magnets.find(m=>m.magnet_id===active);
    for(const mesh of this._meshes){
      mesh.userData.indices.forEach((index,instance)=>{
        const magnet=this.displayMagnets[index],part=poleHalfSpec(magnet,mesh.userData.pole,this.options.explode);
        this._position.set(...part.center);this._rotation.setFromAxisAngle(new THREE.Vector3(0,0,1),part.theta);this._scale.set(...part.scale);
        this._matrix.compose(this._position,this._rotation,this._scale);mesh.setMatrixAt(instance,this._matrix);
        const same=selected?.ring_id===magnet.ring_id;
        this._color.setRGB(same?1.12:1,same?1.1:1,same?1.02:1);if(active===magnet.magnet_id)this._color.setRGB(1.3,1.16,.9);mesh.setColorAt(instance,this._color);
      });
      mesh.instanceMatrix.needsUpdate=true;if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;mesh.computeBoundingSphere();
    }
    this._updateVectors();this._updateHighlight(false);this.assembly.updateMatrixWorld(true);this.studio.markShadowsDirty();
    const box=this._worldBounds();this.studio.addFloor(box);this.studio.setModelVisible(this.options.cubes&&this.magnets.length>0);
  }
  _worldBounds(){
    const box=new THREE.Box3(),corner=new THREE.Vector3();
    for(const m of this.displayMagnets)for(const vertex of window.MRIMagnet3D.cubeVertices(m,this.options.explode))box.expandByPoint(corner.set(...vertex).applyQuaternion(PRESENTATION));
    if(box.isEmpty()){
      const envelope=this.config.envelope||{},radius=finite(envelope.max_outer_diameter_mm,440)/2,half=finite(envelope.max_length_mm,500)/2;
      for(const x of [-radius,radius])for(const y of [-radius,radius])for(const z of [-half,half])box.expandByPoint(corner.set(x,y,z).applyQuaternion(PRESENTATION));
    }
    return box;
  }

  _clearGroup(group){for(const child of [...group.children]){group.remove(child);disposeObject(child);}}
  _updateVectors(){
    this._clearGroup(this.vectorGroup);if(!this.options.vectors)return;
    const positions=[],colors=[];
    for(const magnet of this._visibleMagnets()){
      const p=new THREE.Vector3(...displayCenter(magnet,this.options.explode)),v=new THREE.Vector3(...magnet.north_vector),length=magnet.cube_mm*.8;
      const tip=p.clone().addScaledVector(v,length),base=tip.clone().addScaledVector(v,-length*.27),axis=Math.abs(v.z)>.9?new THREE.Vector3(1,0,0):new THREE.Vector3(0,0,1),side=new THREE.Vector3().crossVectors(v,axis).normalize().multiplyScalar(length*.16);
      for(const [a,b] of [[p,tip],[tip,base.clone().add(side)],[tip,base.clone().sub(side)]]){positions.push(...a.toArray(),...b.toArray());const color=this.options.cubes?new THREE.Color('#fffaf0'):new THREE.Color(COLORS[magnet.polarity]);colors.push(color.r,color.g,color.b,color.r,color.g,color.b);}
    }
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
    this.vectorGroup.add(new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.72,depthWrite:false,toneMapped:false})));
  }

  _rebuildReferences(){
    this._clearGroup(this.referenceGroup);const positions=[],colors=[];
    const line=(a,b,color)=>{positions.push(...a,...b);const c=new THREE.Color(color);colors.push(c.r,c.g,c.b,c.r,c.g,c.b);};
    const ring=(r,z,color,plane='xy')=>{for(let i=0;i<96;i++){const a=i*TAU/96,b=(i+1)*TAU/96;line(plane==='xy'?[r*Math.cos(a),r*Math.sin(a),z]:[r*Math.cos(a),z,r*Math.sin(a)],plane==='xy'?[r*Math.cos(b),r*Math.sin(b),z]:[r*Math.cos(b),z,r*Math.sin(b)],color);}};
    const envelope=this.config.envelope||{},target=this.config.target||{},r=finite(envelope.protected_bore_diameter_mm,200)/2,outer=finite(envelope.max_outer_diameter_mm,440)/2,half=finite(envelope.max_length_mm,500)/2;
    if(this.options.bore){
      const pale='#bdc9cf';for(const z of [-half,half]){ring(r,z,pale);ring(outer,z,'#d7dfe4');}
      for(let i=0;i<8;i++){const a=i*TAU/8;line([r*Math.cos(a),r*Math.sin(a),-half],[r*Math.cos(a),r*Math.sin(a),half],'#d8e2e7');}
      const sectors=clamp(Math.round(finite(this.config.manufacturing?.sectors_per_layer,4)),1,64);
      for(let i=0;i<sectors;i++){const a=i*TAU/sectors,c=Math.cos(a),s=Math.sin(a);for(const z of [-half,half])line([r*c,r*s,z],[outer*c,outer*s,z],'#c3cfd5');}
    }
    if(this.options.target){
      const radius=Math.max(.001,finite(target.diameter_mm,20)/2),z=finite(target.z_mm,0),green='#68a68a';
      if(target.geometry==='disk'){for(const f of [1,.5])ring(radius*f,z,green);line([-radius,0,z],[radius,0,z],green);line([0,-radius,z],[0,radius,z],green);}
      else{
        for(let j=-2;j<=2;j++){const phi=j*Math.PI/6;ring(radius*Math.cos(phi),radius*Math.sin(phi),green);}
        for(let j=0;j<6;j++){const longitude=j*Math.PI/6;for(let i=0;i<72;i++){const a=i*TAU/72,b=(i+1)*TAU/72;line([radius*Math.cos(a)*Math.cos(longitude),radius*Math.cos(a)*Math.sin(longitude),radius*Math.sin(a)],[radius*Math.cos(b)*Math.cos(longitude),radius*Math.cos(b)*Math.sin(longitude),radius*Math.sin(b)],green);}}
        const sphere=new THREE.Mesh(new THREE.SphereGeometry(radius,32,16),new THREE.MeshBasicMaterial({color:'#83b99e',transparent:true,opacity:.045,depthWrite:false,toneMapped:false}));this.referenceGroup.add(sphere);
      }
    }
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
    this.referenceGroup.add(new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.38,depthWrite:false,toneMapped:false})));
  }

  _updateHighlight(recolor=true){
    this._clearGroup(this.highlightGroup);const id=this.hoveredId||this.selectedId,magnet=this.displayMagnets.find(m=>m.magnet_id===id);
    if(recolor){
      const selected=this.magnets.find(m=>m.magnet_id===id);
      for(const mesh of this._meshes){mesh.userData.indices.forEach((index,instance)=>{const m=this.magnets[index],same=selected?.ring_id===m.ring_id;this._color.setRGB(same?1.12:1,same?1.1:1,same?1.02:1);if(m.magnet_id===id)this._color.setRGB(1.3,1.16,.9);mesh.setColorAt(instance,this._color);});if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;}
    }
    if(!magnet)return;const vertices=window.MRIMagnet3D.cubeVertices(magnet,this.options.explode),coordinates=[];
    for(const [a,b]of EDGES)coordinates.push(...vertices[a],...vertices[b]);
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(coordinates,3));
    const outline=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:'#b48542',transparent:true,opacity:.85,depthTest:false,toneMapped:false}));outline.renderOrder=20;this.highlightGroup.add(outline);
  }

  _fit(view=null,preserveZoom=false){
    if(!this.camera||!this.controls)return;
    const oldDistance=this.camera.position.distanceTo(this.controls.target)||1,oldZoom=this._homeDistance/oldDistance;
    const box=this._worldBounds(),center=box.getCenter(new THREE.Vector3());
    let direction=this.camera.position.clone().sub(this.controls.target).normalize();
    if(view==='top'){direction.set(0,1,.0001);this.camera.up.set(0,0,-1);}
    else if(view==='side'){direction.set(0,.0001,1);this.camera.up.set(1,0,0);}
    else if(view){direction.set(.64,.62,.94).normalize();this.camera.up.set(0,1,0);}
    const right=new THREE.Vector3().crossVectors(this.camera.up,direction).normalize(),up=new THREE.Vector3().crossVectors(direction,right),tangent=Math.tan(THREE.MathUtils.degToRad(this.camera.fov)/2);
    let distance=1;
    for(const x of [box.min.x,box.max.x])for(const y of [box.min.y,box.max.y])for(const z of [box.min.z,box.max.z]){const offset=new THREE.Vector3(x,y,z).sub(center);distance=Math.max(distance,offset.dot(direction)+Math.max(Math.abs(offset.dot(up))/tangent,Math.abs(offset.dot(right))/(tangent*this.camera.aspect)));}
    this._homeDistance=distance*1.08;const target=preserveZoom?this.controls.target.clone():center;
    this.camera.position.copy(direction.multiplyScalar(this._homeDistance/(preserveZoom?oldZoom:1)).add(target));this.controls.target.copy(target);
    const span=box.getSize(new THREE.Vector3()).length();this.camera.near=Math.max(.1,span/1000);this.camera.far=span*40;this.camera.updateProjectionMatrix();this.controls.update();
  }

  setView(view){if(this._fallback)return this._fallback.setView(view);this._fit(view==='reset'?'iso':view);this.hoveredId=null;this._notify();this.invalidate();}
  setOptions(values={}){
    if(this._fallback){this._fallback.setOptions(values);this.options=this._fallback.options;return;}
    const previous={...this.options};for(const key of Object.keys(this.options))if(Object.hasOwn(values,key))this.options[key]=key==='explode'?clamp(finite(values[key],0),0,1):Boolean(values[key]);
    if(previous.cutaway!==this.options.cutaway){const valid=id=>this.magnets.find(m=>m.magnet_id===id)?.position_mm[1]>=-1e-8;if(this.options.cutaway&&!valid(this.selectedId))this.selectedId=null;if(this.options.cutaway&&!valid(this.hoveredId))this.hoveredId=null;this._rebuildInstances();}
    this._meshes.forEach(mesh=>mesh.visible=this.options.cubes);this._rebuildReferences();this._updateMatrices();
    if(previous.explode!==this.options.explode)this._fit(null,true);this._notify();this._status();this.invalidate();
  }
  clearSelection(){if(this._fallback)return this._fallback.clearSelection();this.selectedId=null;this.hoveredId=null;this.inspectedId=null;this.onInspect(null,{pinned:false});this._updateHighlight();this.invalidate();}
  _notify(force=false){const id=this.hoveredId||this.selectedId;if(!force&&id===this.inspectedId)return;this.inspectedId=id;const m=this.magnets.find(m=>m.magnet_id===id);this.onInspect(m?{...m,position_mm:m.position_mm.slice(),north_vector:m.north_vector.slice()}:null,{pinned:!!this.selectedId&&id===this.selectedId});}
  _pickClient(x,y){const rect=this.canvas.getBoundingClientRect();return this.pick(x-rect.left,y-rect.top);}
  pick(x,y){
    if(this._fallback)return this._fallback.pick(x,y);if(x<0||y<0||x>this.width||y>this.height)return null;
    this.pointer.set(x/this.width*2-1,-y/this.height*2+1);this.camera.updateMatrixWorld();this.assembly.updateMatrixWorld(true);this.raycaster.setFromCamera(this.pointer,this.camera);
    const hit=this.raycaster.intersectObjects(this._meshes,false)[0];return hit?this.magnets[hit.object.userData.indices[hit.instanceId]]:null;
  }
  resize(){
    if(this._fallback)return this._fallback.resize();if(this.disposed||!this.renderer)return;
    const rect=this.canvas.getBoundingClientRect();if(!(rect.width>0&&rect.height>0))return;
    const next={width:rect.width,height:rect.height,dpr:Math.min(window.devicePixelRatio||1,2)},last=this._pendingSize||this._appliedSize;
    if(last&&last.width===next.width&&last.height===next.height&&last.dpr===next.dpr)return;
    // Keep the previous image while the DOM changes. Buffer allocation happens
    // only at the start of a render frame, immediately followed by its redraw.
    this._pendingSize=next;this.invalidate();
  }
  _applyPendingSize(){
    const next=this._pendingSize;if(!next)return;this._pendingSize=null;
    const last=this._appliedSize,pixelWidth=Math.max(1,Math.floor(next.width*next.dpr)),pixelHeight=Math.max(1,Math.floor(next.height*next.dpr));
    const changed=!last||last.width!==next.width||last.height!==next.height||last.dpr!==next.dpr;
    this.width=next.width;this.height=next.height;this.dpr=next.dpr;
    if(!last||this.canvas.width!==pixelWidth||this.canvas.height!==pixelHeight||last.dpr!==next.dpr){
      this.renderer.setDrawingBufferSize(this.width,this.height,this.dpr);this.bufferResizeCount++;
    }
    this.camera.aspect=this.width/this.height;this.camera.updateProjectionMatrix();
    if(this.overlay.width!==pixelWidth)this.overlay.width=pixelWidth;
    if(this.overlay.height!==pixelHeight)this.overlay.height=pixelHeight;
    this.overlay.style.width=this.width+'px';this.overlay.style.height=this.height+'px';this.overlay.style.left=this.canvas.offsetLeft+'px';this.overlay.style.top=this.canvas.offsetTop+'px';
    this._appliedSize=next;if(changed&&this._hasFit)this._fit(null,true);
  }
  setActive(active){const was=this.active;this.active=Boolean(active);if(this._fallback)return this._fallback.setActive(active);if(this.active){this.resize();if(!was||!this.frameCount||this._transition)this.invalidate();}else if(this._frame){cancelAnimationFrame(this._frame);this._frame=null;}}
  invalidate(){if(this.active&&!this.disposed&&!this._fallback&&!this._frame)this._frame=requestAnimationFrame(time=>this._render(time));}
  _render(time){
    this._frame=null;if(!this.active||this.disposed||this._fallback)return;
    this._applyPendingSize();
    if(this._transition){const transition=this._transition,part=clamp((time-transition.start)/transition.duration,0,1),t=part*part*(3-2*part);this.displayMagnets=this.magnets.map(m=>{const old=transition.previous.get(m.magnet_id);if(!old)return m;return {...m,theta:old.theta+Math.atan2(Math.sin(m.theta-old.theta),Math.cos(m.theta-old.theta))*t,position_mm:m.position_mm.map((v,i)=>old.position_mm[i]+(v-old.position_mm[i])*t)};});this._updateMatrices();if(part>=1){this._transition=null;this.displayMagnets=this.magnets;}}
    const moved=this.controls.update();this.renderer.render(this.scene,this.camera);this._drawOverlay();this.frameCount++;if(moved||this._transition)this.invalidate();
  }
  _drawOverlay(){
    const ctx=this.overlayContext;if(!ctx)return;ctx.setTransform(this.dpr,0,0,this.dpr,0,0);ctx.clearRect(0,0,this.width,this.height);
    if(this.options.gizmo){const origin={x:this.width-46,y:this.height-48},quaternion=this.camera.quaternion.clone().invert();ctx.fillStyle='#ffffffd9';ctx.beginPath();ctx.arc(origin.x,origin.y,39,0,TAU);ctx.fill();ctx.font='10px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
      for(const [vector,color,name]of [[[1,0,0],'#b26d75','X'],[[0,1,0],'#70987e','Y'],[[0,0,1],'#688aa7','Z']]){const v=new THREE.Vector3(...vector).applyQuaternion(PRESENTATION).applyQuaternion(quaternion);ctx.strokeStyle=color;ctx.lineWidth=1.4;ctx.beginPath();ctx.moveTo(origin.x,origin.y);ctx.lineTo(origin.x+v.x*25,origin.y-v.y*25);ctx.stroke();ctx.fillStyle=color;ctx.fillText(name,origin.x+v.x*33,origin.y-v.y*33);}}
    ctx.textAlign='left';ctx.fillStyle='#98a0a5';ctx.font='10px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif';ctx.fillText(this.design?.preview_only?'初步示意 · 未合成 / 未求场':this.options.explode?'爆炸展示 · 检查器保留真实尺寸':'工程尺寸 · mm',13,this.height-13);
  }
  diagnostics(){
    if(this._fallback)return {...this._fallback.diagnostics(),renderStyle:'compatibility'};
    this.camera.updateMatrixWorld();this.assembly.updateMatrixWorld(true);const distance=this.camera.position.distanceTo(this.controls.target),offset=this.camera.position.clone().sub(this.controls.target),points=this._visibleMagnets().map(m=>{const p=new THREE.Vector3(...displayCenter(m,this.options.explode)).applyQuaternion(PRESENTATION),depth=-p.distanceTo(this.camera.position);p.project(this.camera);return {x:(p.x*.5+.5)*this.width,y:(-.5*p.y+.5)*this.height,depth,magnet_id:m.magnet_id,ring_id:m.ring_id,visible:p.z>=-1&&p.z<=1&&Math.abs(p.x)<=1&&Math.abs(p.y)<=1};}).sort((a,b)=>b.depth-a.depth);
    return {renderer:'webgl',renderStyle:'mri-ring-explorer',label:this.label,magnetCount:this.magnets.length,visibleCount:this._visibleMagnets().length,ringCount:new Set(this.magnets.map(m=>m.ring_id)).size,frameCount:this.frameCount,
      camera:{azimuth:Math.atan2(offset.y,offset.x),elevation:Math.asin(offset.z/(distance||1)),zoom:this._homeDistance/(distance||1),pan:this.controls.target.toArray()},options:{...this.options},selection:this.selectedId,hover:this.hoveredId,
      canvas:{width:this.width,height:this.height,dpr:this.dpr},bufferResizeCount:this.bufferResizeCount,previewOnly:!!this.design?.preview_only,coordinateSpace:'projectedMagnets x/y are CSS pixels local to the canvas',projectedMagnets:points,glError:this.renderer.getContext().getError(),animating:!!this._transition,
      studio:{projection:'perspective',fov:this.camera.fov,background:'#ffffff',environment:!!this.scene.environment,lighting:'mri-ring-explorer',poleStyle:'N/S half-cubes',north:MRI_REFERENCE_STYLE.north,south:MRI_REFERENCE_STYLE.south}};
  }
  get gl(){return this._fallback?this._fallback.gl:this.renderer?.getContext();}
  dispose(){
    this.disposed=true;if(this._fallback){this._fallback.dispose();return;}if(this._frame)cancelAnimationFrame(this._frame);if(this._pickFrame)cancelAnimationFrame(this._pickFrame);this._resizeObserver?.disconnect();
    for(const [target,type,fn,options]of this._listeners)target.removeEventListener(type,fn,options);this.controls.dispose();this.studio.dispose();
    for(const mesh of this._meshes)mesh.dispose();this.geometry.dispose();Object.values(this.materials).forEach(material=>material.dispose());this._clearGroup(this.referenceGroup);this._clearGroup(this.vectorGroup);this._clearGroup(this.highlightGroup);this.renderer.dispose();this.overlay.remove();
  }
}
