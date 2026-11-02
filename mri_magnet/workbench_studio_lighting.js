/* Visual reference: Mri-ring-explorer/components/mri-scene.tsx. */
import * as THREE from './vendor/three.module.js';
import {RoomEnvironment} from './vendor/RoomEnvironment.js';
export const MRI_REFERENCE_STYLE=Object.freeze({
  source:'Mri-ring-explorer/components/mri-scene.tsx',north:'#e03d34',south:'#0071e3',
  roughness:.4,metalness:.12,exposure:1.1,environmentIntensity:.35,fov:37,damping:.08,
  hemisphere:1.65,key:2.2,fill:.85,
});
export function setupStudio(renderer,scene){
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=MRI_REFERENCE_STYLE.exposure;
  renderer.setClearColor(0xffffff,1);scene.background=new THREE.Color(0xffffff);renderer.shadowMap.enabled=false;
  const room=new RoomEnvironment(),pmrem=new THREE.PMREMGenerator(renderer);let environment=null;
  try{environment=pmrem.fromScene(room,.05,.1,100,{size:128});scene.environment=environment.texture;scene.environmentIntensity=MRI_REFERENCE_STYLE.environmentIntensity;}
  catch{/* Direct light remains usable without an environment-map target. */}
  finally{room.dispose();pmrem.dispose();}
  const lights=new THREE.Group();lights.name='MRI ring explorer lighting';
  const hemisphere=new THREE.HemisphereLight(0xffffff,0xc6c9cd,MRI_REFERENCE_STYLE.hemisphere);
  const key=new THREE.DirectionalLight(0xffffff,MRI_REFERENCE_STYLE.key),fill=new THREE.DirectionalLight(0xffffff,MRI_REFERENCE_STYLE.fill);
  key.position.set(400,750,600);fill.position.set(-500,75,-325);
  lights.add(hemisphere,key,fill);scene.add(lights);
  return {addFloor(){},markShadowsDirty(){},setModelVisible(){},environment,key,fill,
    dispose(){scene.remove(lights);scene.environment=null;environment?.dispose();}};
}
