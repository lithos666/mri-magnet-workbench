/* v5: illustrative geometry only. No source layout, field solve or seed state. */
(function(root){
  'use strict';
  function value(object,key,fallback){
    const raw=object?.[key];return raw===undefined?fallback:Number(raw);
  }
  function integers(values,fallback){
    return (Array.isArray(values)?values:fallback).filter(n=>Number.isInteger(n)&&n>0);
  }
  function makeDesign(config={}){
    const e=config.envelope||{},m=config.magnet||{},p=config.manufacturing||{},s=config.search||{};
    const a=value(m,'cube_mm',12),br=value(m,'remanence_T',1.43),mur=value(m,'reference_mur',1.05);
    const bore=value(e,'protected_bore_diameter_mm',200)/2,outer=value(e,'max_outer_diameter_mm',440)/2;
    const halfLength=value(e,'max_length_mm',500)/2,printing=p.enabled!==false;
    const clearance=printing?value(p,'pocket_clearance_per_side_mm',.2):0;
    const web=printing?value(p,'pocket_web_mm',1.6):0;
    const innerWall=printing?value(p,'inner_wall_mm',2.4):0,outerWall=printing?value(p,'outer_wall_mm',2.4):0;
    const base=printing?value(p,'base_thickness_mm',5):0,gap=printing?value(p,'min_axial_module_gap_mm',2):0;
    const innerClear=value(p,'radial_inner_clearance_mm',0),centralGap=value(p,'min_central_magnet_gap_mm',0);
    const maxCount=value(p,'max_magnet_count',4608),sectors=value(p,'sectors_per_layer',4);
    const finite=[a,br,mur,bore,outer,halfLength,clearance,web,innerWall,outerWall,base,gap,innerClear,centralGap,maxCount,sectors];
    if(!finite.every(Number.isFinite)||a<=0||br<=0||mur<1||bore<0||outer<=bore||halfLength<=0||
       [clearance,web,innerWall,outerWall,base,gap,innerClear,centralGap].some(x=>x<0)||
       !Number.isInteger(maxCount)||maxCount<8||!Number.isInteger(sectors)||sectors<2)return null;
    const pocket=a+2*clearance,ph=pocket/2;
    const rmin=Math.max(bore+innerWall+ph,bore+innerClear+a/2);
    const remainingOuter=outer-outerWall;
    if(remainingOuter<=ph)return null;
    const rmax=Math.sqrt(remainingOuter*remainingOuter-ph*ph)-ph;
    const pitch=printing&&p.layout_mode==='independent_pockets'?pocket+web:a;
    const radialCapacity=Math.floor((rmax-rmin+1e-8)/pitch)+1;
    const radialLimit=s.max_radial_rings_per_layer==null?radialCapacity:Number(s.max_radial_rings_per_layer);
    if(radialCapacity<1||!Number.isInteger(radialLimit)||radialLimit<1)return null;
    const zmin=Math.max((pocket+2*base+gap)/2,(a+centralGap)/2),zmax=halfLength-ph;
    const axialPitch=pocket+base+gap;
    const axialCapacity=Math.floor((zmax-zmin+1e-8)/axialPitch)+1;
    if(axialCapacity<1)return null;
    const desiredPairs=integers(config.de_novo?.pair_counts,[7])[0];
    if(!desiredPairs)return null;
    const columnValues=config.mode==='seeded'&&Array.isArray(s.azimuth_count_candidates)
      ?s.azimuth_count_candidates:config.de_novo?.column_counts;
    const counts=[...new Set(integers(columnValues,[28,24,32,36,20,16,12,8,4]))]
      .filter(n=>n>=4&&n<=512&&n%2===0&&(!printing||p.allow_split_pocket_seams===true||n%sectors===0))
      .sort((x,y)=>Math.abs(x-28)-Math.abs(y-28)||x-y);
    // Circumscribed pocket circles give a conservative clearance at every
    // azimuth, including between adjacent radial rings in distinct columns.
    const xyEdge=printing?pocket+web:a;
    const count=counts.find(n=>2*rmin*Math.sin(Math.PI/n)>=Math.SQRT2*xyEdge-1e-8&&2*n<=maxCount);
    if(!count)return null;
    const visualBudget=Math.min(maxCount,4096);
    const pairs=Math.min(desiredPairs,axialCapacity,Math.floor(visualBudget/(2*count)),32);
    if(pairs<1)return null;
    const radialCount=Math.min(4,radialCapacity,radialLimit);
    const perPair=Array.from({length:pairs},(_,i)=>Math.max(1,radialCount-(i>=pairs-2&&radialCount>2?1:0)));
    let available=Math.floor(visualBudget/(2*count));
    while(perPair.reduce((sum,n)=>sum+n,0)>available){
      let changed=false;
      for(let i=pairs-1;i>=0&&perPair.reduce((sum,n)=>sum+n,0)>available;i--){
        if(perPair[i]>1){perPair[i]--;changed=true;}
      }
      if(!changed)return null;
    }
    const phase=-90+180/count,rings=[];
    for(let i=0;i<pairs;i++){
      const z=pairs===1?zmin:zmax-(zmax-zmin)*i/(pairs-1),n=perPair[i];
      for(let j=0;j<n;j++){
        const radius=n===1?(rmin+rmax)/2:rmin+(rmax-rmin)*j/(n-1);
        const positivePolarity=i<pairs-2?'out':j%2===0?'down':'out';
        const negativePolarity=positivePolarity==='out'?'in':positivePolarity==='in'?'out':positivePolarity;
        const positiveLayer=i,negativeLayer=2*pairs-1-i;
        for(const [layer,zValue,polarity]of [[positiveLayer,z,positivePolarity],[negativeLayer,-z,negativePolarity]]){
          rings.push({id:'R'+String(layer+1).padStart(2,'0')+'-L'+(j+1),layer_index:layer,slot_index:j,
            radius_mm:radius,z_mm:zValue,count,polarity,phase_deg:phase});
        }
      }
    }
    rings.sort((x,y)=>x.layer_index-y.layer_index||x.slot_index-y.slot_index);
    return {rings,cube_mm:a,remanence_T:br,reference_mur:mur,name:'Initial geometry preview',preview_only:true};
  }
  const api=Object.freeze({makeDesign});root.MRIInitialPreview=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window!=='undefined'?window:globalThis);
