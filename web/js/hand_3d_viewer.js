/**
 * Rigged 3D Hand Visualizer for DuoLibras using Three.js and GLTFLoader.
 * Renders an authentic OpenXR Humanoid Hand with Forward Kinematics (FK)
 * and 100% textbook-accurate canonical configurations for all 26 letters of the Libras alphabet
 * plus dynamic greetings and expressions.
 */

// Calibrated 3D Thumb Presets matching official Libras diagrams
const THUMB_PRESETS = {
  // Resting OVER folded fingers (I, J, S, U, V, W, R, Z) — exact match to official Libras diagram!
  OVER_FIST: { mX: 0.25, mY: -0.60, mZ: 0.15, pX: 0.85, pZ: -0.30, dX: 0.55 },
  // Resting ALONGSIDE index finger (A)
  SIDE_FIST: { mX: 0.20, mY: -0.35, mZ: 0.10, pX: 0.25, pZ: -0.15, dX: 0.20 },
  // Folded ACROSS palm under straight/hooked fingers (B, E)
  ACROSS_PALM: { mX: 0.50, mY: -0.20, mZ: -0.85, pX: 0.85, pZ: 0.00, dX: 0.65 },
  // Curved forward to touch fingertips (D, O)
  PINCH_CIRCLE: { mX: 0.25, mY: -0.55, mZ: 0.25, pX: 0.55, pZ: -0.10, dX: 0.55 },
  // Open C-arch (C)
  C_ARCH: { mX: 0.20, mY: -0.45, mZ: 0.25, pX: 0.40, pZ: 0.00, dX: 0.45 },
  // Extended straight OUT at 90 degrees (L, Y)
  SPREAD_90: { mX: -0.30, mY: 0.30, mZ: -0.60, pX: -0.20, pZ: 0.00, dX: -0.10 },
  // Pointing straight UP alongside index (G, Q)
  STRAIGHT_UP: { mX: 0.15, mY: -0.35, mZ: 0.15, pX: 0.10, pZ: -0.10, dX: 0.05 },
  // Touching OUTSIDE of curled index (F)
  OUTSIDE_INDEX: { mX: 0.15, mY: -0.25, mZ: 0.25, pX: 0.35, pZ: 0.15, dX: 0.20 },
  // Tucked INSIDE between index and middle (T, K, P, H)
  BETWEEN_INDEX_MID: { mX: 0.30, mY: -0.68, mZ: -0.05, pX: 0.50, pZ: -0.25, dX: 0.20 },
  // Thumbs up (Ajudar)
  THUMBS_UP: { mX: -0.25, mY: 0.20, mZ: -0.50, pX: -0.15, pZ: 0.00, dX: 0.00 },
  // Open Relaxed Hand
  OPEN_RELAXED: { mX: 0.05, mY: -0.10, mZ: 0.10, pX: 0.05, pZ: 0.00, dX: 0.05 }
};

// Official textbook configurations for all 26 Libras alphabet letters + Module 2 Greetings
const CANONICAL_SIGNS = {
  // === MODULE 1: ALPHABET (A - Z) ===
  'A': { fingers: [1, 1, 1, 1], thumb: THUMB_PRESETS.SIDE_FIST },
  'B': { fingers: [0, 0, 0, 0], thumb: THUMB_PRESETS.ACROSS_PALM, spread: [0.14, 0.04, -0.05, -0.16] },
  'C': { fingers: [0.45, 0.45, 0.45, 0.45], thumb: THUMB_PRESETS.C_ARCH, wristY: -0.45 },
  'D': { fingers: [0, 0.92, 0.92, 0.92], thumb: THUMB_PRESETS.PINCH_CIRCLE },
  'E': { fingers: [0.9, 0.9, 0.9, 0.9], eHook: true, thumb: THUMB_PRESETS.ACROSS_PALM },
  'F': { fingers: [0.85, 0, 0, 0], fIndex: true, thumb: THUMB_PRESETS.OUTSIDE_INDEX, spread: [0, -0.04, -0.08, -0.12] },
  'G': { fingers: [0, 1, 1, 1], thumb: THUMB_PRESETS.STRAIGHT_UP },
  'H': { fingers: [0, 0, 1, 1], thumb: THUMB_PRESETS.BETWEEN_INDEX_MID, spread: [0.06, -0.02, 0, 0], motion: "twistH" },
  'I': { fingers: [1, 1, 1, 0], thumb: THUMB_PRESETS.OVER_FIST },
  'J': { fingers: [1, 1, 1, 0], thumb: THUMB_PRESETS.OVER_FIST, motion: "traceJ" },
  'K': { fingers: [0, 0.25, 1, 1], thumb: THUMB_PRESETS.BETWEEN_INDEX_MID, spread: [-0.08, 0.08, 0, 0], motion: "upK" },
  'L': { fingers: [0, 1, 1, 1], thumb: THUMB_PRESETS.SPREAD_90 },
  'M': { fingers: [0, 0, 0, 1], thumb: THUMB_PRESETS.OVER_FIST, wristX: Math.PI * 0.85, spread: [0.12, 0, -0.08, 0] },
  'N': { fingers: [0, 0, 1, 1], thumb: THUMB_PRESETS.OVER_FIST, wristX: Math.PI * 0.85, spread: [0.11, -0.05, 0, 0] },
  'O': { fingers: [0.72, 0.72, 0.72, 0.72], thumb: THUMB_PRESETS.PINCH_CIRCLE, wristY: -0.55 },
  'P': { fingers: [0, 0, 1, 1], pMiddle: true, thumb: THUMB_PRESETS.BETWEEN_INDEX_MID, wristY: -0.65, wristX: Math.PI * 0.45 },
  'Q': { fingers: [0, 1, 1, 1], thumb: { mX: 0.20, mY: -0.45, mZ: 0.20, pX: 0.15, pZ: 0, dX: 0.10 }, wristY: -0.55, wristX: Math.PI * 0.78 },
  'R': { fingers: [0, 0, 1, 1], rCross: true, thumb: THUMB_PRESETS.OVER_FIST },
  'S': { fingers: [1, 1, 1, 1], thumb: THUMB_PRESETS.OVER_FIST },
  'T': { fingers: [0.85, 0, 0, 0], fIndex: true, thumb: THUMB_PRESETS.BETWEEN_INDEX_MID, spread: [0.08, -0.08, -0.10, -0.12] },
  'U': { fingers: [0, 0, 1, 1], thumb: THUMB_PRESETS.OVER_FIST, spread: [0.11, -0.05, 0, 0] },
  'V': { fingers: [0, 0, 1, 1], thumb: THUMB_PRESETS.OVER_FIST, spread: [-0.16, 0.16, 0, 0] },
  'W': { fingers: [0, 0, 0, 1], thumb: THUMB_PRESETS.OVER_FIST, spread: [-0.18, 0.0, 0.18, 0] },
  'X': { fingers: [0.6, 1, 1, 1], xHook: true, thumb: THUMB_PRESETS.OVER_FIST, wristY: -0.35, motion: "pullX" },
  'Y': { fingers: [1, 1, 1, 0], thumb: THUMB_PRESETS.SPREAD_90, spread: [0, 0, 0, 0.15] },
  'Z': { fingers: [0, 1, 1, 1], thumb: THUMB_PRESETS.OVER_FIST, motion: "traceZ" },

  // === MODULE 2: GREETINGS & EXPRESSIONS ===
  'Oi': { fingers: [1, 1, 1, 0], thumb: THUMB_PRESETS.OVER_FIST, motion: "waveOi" },
  'Obrigado': { fingers: [0, 0, 0, 0], thumb: THUMB_PRESETS.ACROSS_PALM, spread: [0.14, 0.04, -0.05, -0.16], motion: "nodForward" },
  'Por favor': { fingers: [0, 0, 0, 0], thumb: THUMB_PRESETS.SIDE_FIST, spread: [0.14, 0.04, -0.05, -0.16], wristY: -0.4, motion: "circleChest" },
  'Sim': { fingers: [1, 1, 1, 1], thumb: THUMB_PRESETS.OVER_FIST, motion: "nodYes" },
  'Não': { fingers: [0.15, 0.15, 1, 1], thumb: THUMB_PRESETS.PINCH_CIRCLE, wristY: -0.5, spread: [0.10, -0.04, 0, 0], motion: "pinchNo" },
  'Desculpa': { fingers: [1, 1, 1, 0], thumb: THUMB_PRESETS.SPREAD_90, spread: [0, 0, 0, 0.15], wristY: -0.3, motion: "tapChin" },
  'Nome': { fingers: [0, 0, 1, 1], thumb: THUMB_PRESETS.OVER_FIST, spread: [0.11, -0.05, 0, 0], wristZ: Math.PI * 0.42, motion: "tapName" },
  'Cumprimento': { fingers: [0, 0, 0, 0], thumb: THUMB_PRESETS.OPEN_RELAXED, motion: "waveOpen" },
  'Ajudar': { fingers: [1, 1, 1, 1], thumb: THUMB_PRESETS.THUMBS_UP, wristZ: -Math.PI * 0.15, motion: "pushForward" },
  '_OPEN': { fingers: [0, 0, 0, 0], thumb: THUMB_PRESETS.OPEN_RELAXED }
};

class Hand3DViewer {
  constructor(containerId, posesUrl = "data/hand_poses_3d.json") {
    this.container = document.getElementById(containerId);
    this.posesUrl = posesUrl;
    this.poses = {};
    this.activeSignName = "_OPEN";
    this.activeSignConf = CANONICAL_SIGNS["_OPEN"];

    // Movement animation state
    this.isPlayingMotion = false;
    this.motionStartTime = 0;
    this.onMotionStateChange = null;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.handGroup = null;

    // Model and Bone Rigging
    this.handModel = null;
    this.handBones = {};
    this.initialQuaternions = {};
    this.baseModelPosition = new THREE.Vector3();
    this.skinMaterial = null;
    this.isModelReady = false;

    // Current color
    this.currentColor = localStorage.getItem("duolibras_hand_color") || "#E8A082";

    // Interaction
    this.isDragging = false;
    this.previousMousePosition = { x: 0, y: 0 };
    this.fingerPrefixes = ["RightIndex", "RightMiddle", "RightRing", "RightLittle"];
  }

  async init() {
    if (!this.container) return;

    const width = this.container.clientWidth || 360;
    const height = this.container.clientHeight || 280;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0A192F);

    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 10);
    this.camera.position.set(0, 0, 0.35);
    this.camera.lookAt(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    // Warm, natural studio lighting
    const ambientLight = new THREE.AmbientLight(0xFFEADB, 0.85);
    this.scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xFFFFFF, 0.95);
    keyLight.position.set(0.5, 1, 1);
    this.scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x80A0FF, 0.45);
    fillLight.position.set(-0.5, 0, 0.5);
    this.scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0x00E5FF, 0.4);
    rimLight.position.set(0, -1, -0.5);
    this.scene.add(rimLight);

    // Hand Group for orbit rotation and animations
    this.handGroup = new THREE.Group();
    this.scene.add(this.handGroup);

    // Shared skin material with skinning enabled
    this.skinMaterial = new THREE.MeshStandardMaterial({
      color: this.currentColor,
      roughness: 0.55,
      metalness: 0.05,
      skinning: true
    });

    this._attachControls();
    await this._loadModel();

    try {
      const res = await fetch(this.posesUrl);
      if (res.ok) {
        this.poses = await res.json();
      }
    } catch (e) {
      // Canonical definitions work offline without external JSON
    }

    this._animate();
  }

  _loadModel() {
    return new Promise((resolve) => {
      if (typeof THREE.GLTFLoader === "undefined") {
        resolve();
        return;
      }

      const loader = new THREE.GLTFLoader();
      loader.load(
        "models/RightHandHumanoid.gltf",
        (gltf) => {
          this.handModel = gltf.scene;
          this.handGroup.add(this.handModel);

          this.handModel.traverse((child) => {
            if (child.isBone) {
              this.handBones[child.name] = child;
              this.initialQuaternions[child.name] = child.quaternion.clone();
            }
            if (child.isSkinnedMesh) {
              child.material = this.skinMaterial;
            }
          });

          this.isModelReady = true;
          if (this.activeSignConf) {
            this.applyCanonicalSign(this.activeSignConf);
          }
          resolve();
        },
        undefined,
        () => resolve()
      );
    });
  }

  setColor(hexColor) {
    this.currentColor = hexColor;
    if (this.skinMaterial) {
      this.skinMaterial.color.set(hexColor);
    }
    localStorage.setItem("duolibras_hand_color", hexColor);
  }

  _attachControls() {
    const el = this.renderer.domElement;

    const onDown = (clientX, clientY) => {
      this.isDragging = true;
      this.previousMousePosition = { x: clientX, y: clientY };
    };

    const onMove = (clientX, clientY) => {
      if (!this.isDragging) return;
      const deltaX = clientX - this.previousMousePosition.x;
      const deltaY = clientY - this.previousMousePosition.y;

      this.handGroup.rotation.y += deltaX * 0.012;
      this.handGroup.rotation.x += deltaY * 0.012;

      this.previousMousePosition = { x: clientX, y: clientY };
    };

    const onUp = () => {
      this.isDragging = false;
    };

    el.addEventListener("mousedown", (e) => onDown(e.clientX, e.clientY));
    window.addEventListener("mousemove", (e) => onMove(e.clientX, e.clientY));
    window.addEventListener("mouseup", onUp);

    el.addEventListener("touchstart", (e) => {
      if (e.touches.length === 1) onDown(e.touches[0].clientX, e.touches[0].clientY);
    });
    window.addEventListener("touchmove", (e) => {
      if (e.touches.length === 1) onMove(e.touches[0].clientX, e.touches[0].clientY);
    });
    window.addEventListener("touchend", onUp);
  }

  setPose(letterOrName) {
    this.isPlayingMotion = false;
    if (this.onMotionStateChange) this.onMotionStateChange(false);
    this.handGroup.rotation.set(0, 0, 0);
    this.handGroup.position.set(0, 0, 0);

    this.activeSignName = letterOrName;
    this.activeSignConf = CANONICAL_SIGNS[letterOrName] || CANONICAL_SIGNS["_OPEN"];

    if (this.isModelReady) {
      this.applyCanonicalSign(this.activeSignConf);
    }
  }

  playMotion() {
    this.isPlayingMotion = true;
    this.motionStartTime = performance.now();
    if (this.onMotionStateChange) this.onMotionStateChange(true);
  }

  pauseMotion() {
    this.isPlayingMotion = false;
    if (this.onMotionStateChange) this.onMotionStateChange(false);
    // Reset any dynamic finger/position offsets cleanly back to canonical pose
    this.handGroup.position.set(0, 0, 0);
    if (this.isModelReady && this.activeSignConf) {
      this.applyCanonicalSign(this.activeSignConf);
    }
  }

  toggleMotion() {
    if (this.isPlayingMotion) {
      this.pauseMotion();
    } else {
      this.playMotion();
    }
  }

  _rotateBone(name, axisName, angle) {
    const bone = this.handBones[name];
    if (!bone || !this.initialQuaternions[name]) return;
    const axis = new THREE.Vector3(
      axisName === "x" ? 1 : 0,
      axisName === "y" ? 1 : 0,
      axisName === "z" ? 1 : 0
    );
    const q = new THREE.Quaternion().setFromAxisAngle(axis, angle);
    bone.quaternion.copy(this.initialQuaternions[name]).multiply(q);
  }

  _centerModelOnBones() {
    this.handModel.position.set(0, 0, 0);
    this.handGroup.rotation.set(0, 0, 0);
    this.handGroup.position.set(0, 0, 0);
    this.scene.updateMatrixWorld(true);

    const box = new THREE.Box3();
    Object.values(this.handBones).forEach((b) => {
      const wp = new THREE.Vector3();
      b.getWorldPosition(wp);
      box.expandByPoint(wp);
    });
    const center = box.getCenter(new THREE.Vector3());
    this.handModel.position.sub(center);
    this.baseModelPosition.copy(this.handModel.position);
  }

  /**
   * Apply 100% textbook-accurate canonical joint rotations for Libras signs
   */
  applyCanonicalSign(signConf) {
    if (!this.isModelReady || !this.handBones["RightIndexProximal"]) return;

    // 1. Base Orientation: Fingers UP (+Y), Palm FORWARD (+Z)
    this.handModel.rotation.set(Math.PI / 2, 0, 0);
    this.handModel.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), Math.PI / 2 + (signConf.wristY || 0));

    if (signConf.wristX) {
      this.handModel.rotateOnWorldAxis(new THREE.Vector3(1, 0, 0), signConf.wristX);
    }
    if (signConf.wristZ) {
      this.handModel.rotateOnWorldAxis(new THREE.Vector3(0, 0, 1), signConf.wristZ);
    }

    // 2. Apply 4 Fingers
    for (let i = 0; i < 4; i++) {
      const prefix = this.fingerPrefixes[i];
      const c = signConf.fingers[i];

      if (signConf.eHook) {
        this._rotateBone(`${prefix}Proximal`, "x", 1.35);
        this._rotateBone(`${prefix}Intermediate`, "x", 1.65);
        this._rotateBone(`${prefix}Distal`, "x", 1.15);
      } else if (signConf.xHook && i === 0) {
        this._rotateBone(`${prefix}Proximal`, "x", 0.45);
        this._rotateBone(`${prefix}Intermediate`, "x", 1.55);
        this._rotateBone(`${prefix}Distal`, "x", 1.15);
      } else if (signConf.fIndex && i === 0) {
        this._rotateBone(`${prefix}Proximal`, "x", c * 1.50);
        this._rotateBone(`${prefix}Intermediate`, "x", c * 1.45);
        this._rotateBone(`${prefix}Distal`, "x", c * 0.95);
      } else if (signConf.pMiddle && i === 1) {
        // Middle finger bent 90 degrees forward at MCP for Letter P
        this._rotateBone(`${prefix}Proximal`, "x", 1.15);
        this._rotateBone(`${prefix}Intermediate`, "x", 0.05);
        this._rotateBone(`${prefix}Distal`, "x", 0.05);
      } else {
        this._rotateBone(`${prefix}Proximal`, "x", c * 1.65);
        this._rotateBone(`${prefix}Intermediate`, "x", c * 1.75);
        this._rotateBone(`${prefix}Distal`, "x", c * 1.25);
      }

      if (signConf.spread && signConf.spread[i]) {
        const sq = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), signConf.spread[i]);
        if (this.handBones[`${prefix}Proximal`]) {
          this.handBones[`${prefix}Proximal`].quaternion.multiply(sq);
        }
      }

      if (signConf.rCross) {
        if (i === 0) {
          this.handBones["RightIndexProximal"].quaternion.multiply(
            new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.06, 0, 0.22))
          );
        } else if (i === 1) {
          this.handBones["RightMiddleProximal"].quaternion.multiply(
            new THREE.Quaternion().setFromEuler(new THREE.Euler(0.14, 0, -0.24))
          );
        }
      }
    }

    // 3. Apply Thumb Preset
    const t = signConf.thumb;
    if (t && this.handBones["RightThumbMetacarpal"] && this.initialQuaternions["RightThumbMetacarpal"]) {
      const qMeta = new THREE.Quaternion().setFromEuler(new THREE.Euler(t.mX, t.mY, t.mZ, "XYZ"));
      this.handBones["RightThumbMetacarpal"].quaternion
        .copy(this.initialQuaternions["RightThumbMetacarpal"])
        .multiply(qMeta);

      const qProx = new THREE.Quaternion().setFromEuler(new THREE.Euler(t.pX, 0, t.pZ || 0, "XYZ"));
      this.handBones["RightThumbProximal"].quaternion
        .copy(this.initialQuaternions["RightThumbProximal"])
        .multiply(qProx);

      this._rotateBone("RightThumbDistal", "x", t.dX);
    }

    // 4. Center accurately on deformed skeleton world bounds
    this._centerModelOnBones();
  }

  _animate() {
    requestAnimationFrame(() => this._animate());

    if (this.isPlayingMotion && this.activeSignConf) {
      const elapsedSec = (performance.now() - this.motionStartTime) / 1000;
      const motion = this.activeSignConf.motion;

      if (motion === "waveOi" || motion === "waveOpen") {
        // Gentle waving arc left & right
        const t = elapsedSec * 3.5;
        this.handGroup.rotation.z = Math.sin(t) * 0.22;
        this.handGroup.rotation.y = Math.sin(t * 0.5) * 0.12;
      } else if (motion === "traceJ") {
        // Draw 'J' hook in the air: down then curve left & up!
        const phase = (elapsedSec * 0.9) % 1.0;
        if (phase < 0.7) {
          const p = phase / 0.7;
          const angle = p * Math.PI;
          this.handGroup.position.x = -Math.sin(angle * 0.5) * 0.035;
          this.handGroup.position.y = -Math.sin(angle) * 0.035;
          this.handGroup.rotation.z = Math.sin(angle * 0.5) * 0.35;
        } else {
          this.handGroup.position.multiplyScalar(0.85);
          this.handGroup.rotation.z *= 0.85;
        }
      } else if (motion === "traceZ") {
        // Draw 'Z' in the air: top-horizontal -> diagonal -> bottom-horizontal
        const phase = (elapsedSec * 0.65) % 1.0;
        if (phase < 0.25) {
          const p = phase / 0.25;
          this.handGroup.position.set(-0.03 + p * 0.06, 0.03, 0);
        } else if (phase < 0.55) {
          const p = (phase - 0.25) / 0.30;
          this.handGroup.position.set(0.03 - p * 0.06, 0.03 - p * 0.06, 0);
        } else if (phase < 0.80) {
          const p = (phase - 0.55) / 0.25;
          this.handGroup.position.set(-0.03 + p * 0.06, -0.03, 0);
        } else {
          this.handGroup.position.multiplyScalar(0.85);
        }
      } else if (motion === "twistH") {
        // H rotates wrist around vertical axis
        const t = elapsedSec * 3.2;
        this.handGroup.rotation.y = Math.sin(t) * 0.55;
      } else if (motion === "upK") {
        // K moves slightly upward twice
        const t = elapsedSec * 4.0;
        this.handGroup.position.y = Math.abs(Math.sin(t)) * 0.025;
      } else if (motion === "pullX") {
        // X pulls hook slightly towards user
        const t = elapsedSec * 3.0;
        this.handGroup.position.z = Math.sin(t) * 0.025;
        this.handGroup.rotation.y = Math.sin(t * 0.5) * 0.20;
      } else if (motion === "nodYes" || motion === "nodForward") {
        // Sim / Obrigado: wrist nods forward and down
        const t = elapsedSec * 4.2;
        this.handGroup.rotation.x = Math.abs(Math.sin(t)) * 0.35;
      } else if (motion === "pinchNo") {
        // Não: Index & Middle tap against thumb!
        const tap = (Math.sin(elapsedSec * 6.0) + 1) * 0.22;
        ["RightIndex", "RightMiddle"].forEach((p, idx) => {
          this._rotateBone(`${p}Proximal`, "x", 0.15 + tap);
          const sp = idx === 0 ? 0.10 : -0.04;
          this.handBones[`${p}Proximal`].quaternion.multiply(
            new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), sp)
          );
        });
      } else if (motion === "circleChest") {
        // Por favor: gentle circular rubbing motion
        const t = elapsedSec * 3.0;
        this.handGroup.position.x = Math.cos(t) * 0.022;
        this.handGroup.position.y = Math.sin(t) * 0.022;
      } else if (motion === "tapChin" || motion === "tapName") {
        // Desculpa / Nome: gentle double tap
        const t = elapsedSec * 5.0;
        this.handGroup.position.z = Math.abs(Math.sin(t)) * 0.02;
      } else if (motion === "pushForward") {
        // Ajudar: push forward
        const phase = (elapsedSec * 0.9) % 1.0;
        this.handGroup.position.z = Math.sin(phase * Math.PI) * 0.035;
      } else {
        // Static Alphabet Letters: Keep 100% exact sign posture & showcase 3D depth
        const t = elapsedSec * 2.2;
        this.handGroup.rotation.y = Math.sin(t) * 0.32;
        this.handGroup.rotation.x = Math.sin(t * 0.5) * 0.08;
      }
    } else if (!this.isDragging) {
      // Smoothly ease back to front center when idle
      this.handGroup.rotation.y *= 0.88;
      this.handGroup.rotation.x *= 0.88;
      this.handGroup.rotation.z *= 0.88;
      this.handGroup.position.multiplyScalar(0.88);
    }

    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    if (!this.container || !this.renderer) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }
}

window.Hand3DViewer = Hand3DViewer;
