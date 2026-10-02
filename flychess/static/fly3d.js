/* The fly's brain in 3D: every FlyWire neuron as a point, lighting up as spikes stream in
   from the simulation. A small 3D fly sits in the corner as the avatar (its head holds a
   miniature copy of the same glowing brain). */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const KIND = { input: new THREE.Color("#0284c7"), brain: new THREE.Color("#f59e0b"), dn: new THREE.Color("#e11d48") };
const REST = { 2: new THREE.Color("#7c93a3"), 4: new THREE.Color("#a98391"), other: new THREE.Color("#8a8792") };
const VP_CLASS = 2, DN_CLASS = 4;
const INSET = { w: 210, h: 170, margin: 14 };

export class FlyScene {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0xffffff, 1);
    this.renderer.autoClear = false;
    container.appendChild(this.renderer.domElement);

    // main scene: the brain
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.05, 100);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 1.2; this.controls.maxDistance = 12;
    this.controls.autoRotate = true; this.controls.autoRotateSpeed = 0.5;
    this.controls.addEventListener("start", () => { this.controls.autoRotate = false; });
    this.renderer.domElement.addEventListener("dblclick", () => this.resetView());
    this.resetView();
    this.brainGroup = new THREE.Group(); this.scene.add(this.brainGroup);

    // inset scene: the fly
    this.flyScene = new THREE.Scene();
    this.flyCamera = new THREE.PerspectiveCamera(32, INSET.w / INSET.h, 0.1, 50);
    this.flyCamera.position.set(4.2, 2.6, 6.4); this.flyCamera.lookAt(0, 0, -1.2);
    this.flyScene.add(new THREE.HemisphereLight(0xffffff, 0xcfc4b4, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.5); key.position.set(4, 6, 5); this.flyScene.add(key);
    const rim = new THREE.DirectionalLight(0xbfe3ff, 0.8); rim.position.set(-5, 2, -4); this.flyScene.add(rim);
    this._buildFly();

    this.eyeGlow = 0; this.thinking = false; this.clock = new THREE.Clock(); this.active = new Set();
    new ResizeObserver(() => this._resize()).observe(container);
    this._resize(); this._loop();
  }

  resetView() {
    this.camera.position.set(0.4, 0.9, 4.8);
    this.controls.target.set(0, -0.25, 0);
    this.controls.update();
  }

  // ------------------------------------------------------------- fly avatar
  _buildFly() {
    const fly = new THREE.Group(); this.fly = fly; this.flyScene.add(fly);
    const chitin = new THREE.MeshStandardMaterial({ color: 0x5a4030, roughness: 0.5, metalness: 0.25 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x3a2a1e, roughness: 0.6, metalness: 0.2 });
    this.head = new THREE.Mesh(new THREE.SphereGeometry(1.0, 40, 40),
      new THREE.MeshPhysicalMaterial({ color: 0x8a6d52, roughness: 0.3, transparent: true, opacity: 0.28, clearcoat: 0.7, depthWrite: false }));
    fly.add(this.head);
    this.headGroup = new THREE.Group(); fly.add(this.headGroup);         // mini brain goes here
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0xb42318, roughness: 0.35, metalness: 0.2, emissive: 0x3a0a0a, emissiveIntensity: 0.4 });
    this.eyes = [];
    for (const sx of [-1, 1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.42, 32, 32, 0, Math.PI * 2, 0, Math.PI * 0.6), eyeMat.clone());
      eye.position.set(sx * 0.78, 0.05, 0.36); eye.lookAt(sx * 4, 0.2, 1.4); eye.rotateX(-Math.PI / 2);
      fly.add(eye); this.eyes.push(eye);
    }
    for (const sx of [-1, 1]) {
      const a = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.035, 0.4, 8), dark);
      a.position.set(sx * 0.18, 0.5, 0.9); a.rotation.x = Math.PI / 3; a.rotation.z = -sx * 0.5; fly.add(a);
    }
    const prob = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.13, 0.55, 12), dark);
    prob.position.set(0, -0.9, 0.4); prob.rotation.x = 0.35; fly.add(prob);
    const thorax = new THREE.Mesh(new THREE.SphereGeometry(1, 40, 40), chitin);
    thorax.scale.set(1.15, 1.05, 1.3); thorax.position.set(0, -0.05, -1.75); fly.add(thorax);
    const abd = new THREE.Mesh(new THREE.SphereGeometry(1, 40, 40), new THREE.MeshStandardMaterial({ color: 0x4a3524, roughness: 0.5, metalness: 0.3 }));
    abd.scale.set(1.0, 0.92, 2.0); abd.position.set(0, -0.35, -4.2); fly.add(abd);
    for (let i = 0; i < 4; i++) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.8 - i * 0.12, 0.035, 8, 48), new THREE.MeshStandardMaterial({ color: 0xa8864f, roughness: 0.6 }));
      ring.position.set(0, -0.35, -3.4 - i * 0.6); ring.scale.set(1.15, 1.05, 1); fly.add(ring);
    }
    this.wings = [];
    const wingMat = new THREE.MeshPhysicalMaterial({ color: 0x9cc7f0, transparent: true, opacity: 0.35, roughness: 0.15, side: THREE.DoubleSide, iridescence: 0.9, iridescenceIOR: 1.3, depthWrite: false });
    const shape = new THREE.Shape();
    shape.moveTo(0, 0); shape.bezierCurveTo(1.2, 0.6, 3.2, 0.9, 4.2, 0.2); shape.bezierCurveTo(3.6, -0.5, 1.8, -0.7, 0, -0.25);
    for (const sx of [-1, 1]) {
      const pivot = new THREE.Group(); pivot.position.set(sx * 0.6, 0.85, -1.6);
      const w = new THREE.Mesh(new THREE.ShapeGeometry(shape, 24), wingMat);
      w.rotation.x = -Math.PI / 2; w.scale.x = sx; w.rotation.z = sx * 0.25; pivot.add(w);
      fly.add(pivot); this.wings.push(pivot);
    }
    const legMat = new THREE.MeshStandardMaterial({ color: 0x2f231a, roughness: 0.7 });
    for (const sx of [-1, 1]) {
      [-0.9, -1.7, -2.6].forEach((z, i) => {
        const hip = new THREE.Group(); hip.position.set(sx * 0.9, -0.6, z); fly.add(hip);
        const femur = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.045, 1.4, 8), legMat);
        femur.position.set(sx * 0.6, 0.15, 0); femur.rotation.z = sx * 1.15; femur.rotation.y = (i - 1) * 0.5 * sx; hip.add(femur);
        const knee = new THREE.Group(); knee.position.set(sx * 1.25, 0.45, (i - 1) * 0.55); hip.add(knee);
        const tibia = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.025, 1.9, 8), legMat);
        tibia.position.set(sx * 0.25, -0.9, 0); tibia.rotation.z = sx * 0.25; knee.add(tibia);
      });
    }
    fly.position.set(0, 0.3, 0); fly.rotation.y = -0.35;
  }

  // ------------------------------------------------------------- brain
  loadBrain(buffer, n) {
    const xyz = new Float32Array(buffer, 0, n * 3), cls = new Uint8Array(buffer, n * 12, n);
    let mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
    for (let i = 0; i < n; i++) { if (xyz[3 * i] !== xyz[3 * i]) continue;
      for (let k = 0; k < 3; k++) { const v = xyz[3 * i + k]; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v; } }
    const c = mn.map((v, k) => (v + mx[k]) / 2), ext = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]);
    const s = 3.6 / ext;                                            // brain spans ~3.6 units in the main view
    const pos = new Float32Array(n * 3), base = new Float32Array(n * 3), lit = new Float32Array(n * 3), act = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
      if (x !== x) { pos[3 * i] = pos[3 * i + 1] = pos[3 * i + 2] = 1e3; continue; }
      pos[3 * i] = (x - c[0]) * s; pos[3 * i + 1] = -(y - c[1]) * s; pos[3 * i + 2] = -(z - c[2]) * s * 1.15;
      const k = cls[i], b = REST[k] || REST.other, l = k === DN_CLASS ? KIND.dn : k === VP_CLASS ? KIND.input : KIND.brain;
      base[3 * i] = b.r; base[3 * i + 1] = b.g; base[3 * i + 2] = b.b; lit[3 * i] = l.r; lit[3 * i + 1] = l.g; lit[3 * i + 2] = l.b;
    }
    this.cls = cls; this.act = act; this.n = n;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("aBase", new THREE.BufferAttribute(base, 3));
    geo.setAttribute("aLit", new THREE.BufferAttribute(lit, 3));
    this.actAttr = new THREE.BufferAttribute(act, 1); this.actAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("aAct", this.actAttr);

    const uniforms = { uPx: { value: this.renderer.getPixelRatio() }, uScale: { value: 1.0 } };
    const vert = (sizeExpr) => `attribute vec3 aBase; attribute vec3 aLit; attribute float aAct; uniform float uPx; uniform float uScale;
      varying vec3 vColor; varying float vAct;
      void main(){ vAct = aAct; vColor = mix(aBase, aLit, clamp(aAct * 1.5, 0.0, 1.0));
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = (${sizeExpr}) * uPx * uScale * (4.5 / -mv.z); gl_Position = projectionMatrix * mv; }`;
    const rest = new THREE.ShaderMaterial({ uniforms, vertexShader: vert("1.7"),
      fragmentShader: `varying vec3 vColor; void main(){ vec2 d = gl_PointCoord - 0.5; float r = length(d); if (r > 0.5) discard;
        gl_FragColor = vec4(vColor, 0.34 * smoothstep(0.5, 0.2, r)); }`, transparent: true, depthWrite: false });
    const halo = new THREE.ShaderMaterial({ uniforms, vertexShader: vert("aAct * 38.0"),
      fragmentShader: `varying vec3 vColor; varying float vAct; void main(){ if (vAct < 0.03) discard;
        vec2 d = gl_PointCoord - 0.5; float r = length(d); if (r > 0.5) discard;
        gl_FragColor = vec4(vColor, (1.0 - smoothstep(0.0, 0.5, r)) * 0.55 * vAct); }`, transparent: true, depthWrite: false, depthTest: false });
    const litMat = new THREE.ShaderMaterial({ uniforms, vertexShader: vert("3.5 + aAct * 14.0"),
      fragmentShader: `varying vec3 vColor; varying float vAct; void main(){ if (vAct < 0.03) discard;
        vec2 d = gl_PointCoord - 0.5; float r = length(d); if (r > 0.5) discard;
        float a = smoothstep(0.5, 0.3, r) * (0.6 + 0.4 * vAct);
        gl_FragColor = vec4(mix(vColor, vec3(1.0), 0.4 * vAct * smoothstep(0.3, 0.0, r)), a); }`, transparent: true, depthWrite: false, depthTest: false });
    this.materials = [rest, halo, litMat];

    const makeLayers = (parent, order0) => {
      [rest, halo, litMat].forEach((m, i) => { const p = new THREE.Points(geo, m); p.frustumCulled = false; p.renderOrder = order0 + i; parent.add(p); });
    };
    this.brainGroup.clear(); makeLayers(this.brainGroup, 1);
    // miniature copy inside the avatar's head
    this.headGroup.clear(); this.headGroup.scale.setScalar(1.7 / 3.6); this.headGroup.position.set(0, 0, 0.05);
    makeLayers(this.headGroup, 1);
  }
  spike(indices) {
    if (!this.act) return;
    let input = 0;
    for (const i of indices) { this.act[i] = 1; this.active.add(i); if (this.cls[i] === VP_CLASS) input++; }
    this.eyeGlow = Math.min(1, this.eyeGlow + input * 0.04);
    this.actAttr.needsUpdate = true;
  }
  reset() { if (!this.act) return; for (const i of this.active) this.act[i] = 0; this.active.clear(); this.actAttr.needsUpdate = true; }
  setThinking(v) { this.thinking = v; if (v) this.controls.autoRotate = false; }

  // ------------------------------------------------------------- loop
  _resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    this.size = [w, h];
  }
  _loop() {
    requestAnimationFrame(() => this._loop());
    const dt = Math.min(this.clock.getDelta(), 0.05), t = this.clock.elapsedTime;
    if (this.active.size) {
      const k = Math.exp(-dt / 0.55);
      for (const i of this.active) { this.act[i] *= k; if (this.act[i] < 0.02) { this.act[i] = 0; this.active.delete(i); } }
      this.actAttr.needsUpdate = true;
    }
    // avatar animation
    const flap = this.thinking ? Math.sin(t * 40) * 0.55 : Math.sin(t * 1.4) * 0.03;
    this.wings[0].rotation.z = 0.35 + flap; this.wings[1].rotation.z = -0.35 - flap;
    this.fly.position.y = 0.3 + Math.sin(t * 1.2) * 0.05 + (this.thinking ? Math.sin(t * 9) * 0.03 : 0);
    this.fly.rotation.y = -0.35 + Math.sin(t * 0.4) * 0.15;
    this.eyeGlow *= Math.exp(-dt / 0.35);
    for (const e of this.eyes) { e.material.emissiveIntensity = 0.4 + this.eyeGlow * 2.5; e.material.emissive.setHex(this.eyeGlow > 0.2 ? 0x0284c7 : 0x3a0a0a); }
    this.controls.update();

    // render: brain full-frame, then the fly in the corner
    const [w, h] = this.size || [1, 1];
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, w, h); this.renderer.clear();
    if (this.materials) this.materials.forEach(m => m.uniforms.uScale.value = 1.0);
    this.renderer.render(this.scene, this.camera);
    const k = Math.min(1, (w * 0.3) / INSET.w);                 // shrink the inset on narrow panels
    const iw = Math.round(INSET.w * k), ih = Math.round(INSET.h * k), ix = w - iw - INSET.margin, iy = INSET.margin;
    this.flyCamera.aspect = iw / ih; this.flyCamera.updateProjectionMatrix();
    const frame = this.container.querySelector(".inset-frame");
    if (frame) { frame.style.width = iw + "px"; frame.style.height = ih + "px"; }
    this.renderer.setScissorTest(true);
    this.renderer.setScissor(ix, iy, iw, ih); this.renderer.setViewport(ix, iy, iw, ih);
    this.renderer.clearDepth();
    this.renderer.setClearColor(0xf7f8fa, 1); this.renderer.clear(true, true, false); this.renderer.setClearColor(0xffffff, 1);
    if (this.materials) this.materials.forEach(m => m.uniforms.uScale.value = 0.45);
    this.renderer.render(this.flyScene, this.flyCamera);
    this.renderer.setScissorTest(false);
  }
}

export async function createFlyScene(container) {
  const scene = new FlyScene(container);
  const r = await fetch("/api/positions");
  const n = parseInt(r.headers.get("X-Neurons"), 10);
  scene.loadBrain(await r.arrayBuffer(), n);
  return scene;
}
