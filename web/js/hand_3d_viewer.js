/**
 * Photorealistic Volumetric 3D Anatomical Hand Visualizer using Three.js.
 * Renders a sculpted, organic 3D hand with fleshy palm muscles, articulated finger phalanges,
 * rounded fingertips, and warm skin shading.
 * 
 * Features:
 * - 100% Driven by 3D landmark poses (instant, glitch-free deformation).
 * - Real anatomical structure: Palm volume + thenar/hypothenar pads + 15 tapered finger phalanges.
 * - Palm facing forward toward user.
 * - Interactive Play / Pause / Replay animation with smooth easing.
 * - Full 360-degree mouse/touch orbit rotation.
 */
class Hand3DViewer {
  constructor(containerId, posesUrl = "data/hand_poses_3d.json") {
    this.container = document.getElementById(containerId);
    this.posesUrl = posesUrl;
    this.poses = {};
    this.currentPose = null;
    this.activeSignData = null;

    // Movement animation state
    this.isPlayingMotion = false;
    this.motionStartTime = 0;
    this.onMotionStateChange = null;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.handGroup = null;

    // Anatomical hand components
    this.palmPad = null;
    this.thenarPad = null;
    this.hypothenarPad = null;
    this.knuckleSpheres = [];
    this.fingerSegments = [];
    this.metacarpalSegments = [];

    // Reusable math objects
    this._tmpV1 = null;
    this._tmpV2 = null;
    this._tmpDir = null;
    this._upVec = null;

    this.isDragging = false;
    this.previousMousePosition = { x: 0, y: 0 };

    // 15 Phalanx segments with realistic anatomical tapering [jointA, jointB, r1, r2]
    this.segmentDefs = [
      // Thumb
      { a: 1, b: 2, r1: 0.065, r2: 0.058 },
      { a: 2, b: 3, r1: 0.058, r2: 0.050 },
      { a: 3, b: 4, r1: 0.050, r2: 0.042 },
      // Index
      { a: 5, b: 6, r1: 0.058, r2: 0.050 },
      { a: 6, b: 7, r1: 0.050, r2: 0.044 },
      { a: 7, b: 8, r1: 0.044, r2: 0.038 },
      // Middle
      { a: 9, b: 10, r1: 0.060, r2: 0.052 },
      { a: 10, b: 11, r1: 0.052, r2: 0.045 },
      { a: 11, b: 12, r1: 0.045, r2: 0.038 },
      // Ring
      { a: 13, b: 14, r1: 0.056, r2: 0.048 },
      { a: 14, b: 15, r1: 0.048, r2: 0.042 },
      { a: 15, b: 16, r1: 0.042, r2: 0.035 },
      // Pinky
      { a: 17, b: 18, r1: 0.050, r2: 0.043 },
      { a: 18, b: 19, r1: 0.043, r2: 0.036 },
      { a: 19, b: 20, r1: 0.036, r2: 0.030 }
    ];

    // Metacarpals connecting wrist (0) to finger roots
    this.metacarpalDefs = [
      { a: 0, b: 1, r: 0.068 },
      { a: 0, b: 5, r: 0.062 },
      { a: 0, b: 9, r: 0.064 },
      { a: 0, b: 13, r: 0.060 },
      { a: 0, b: 17, r: 0.054 }
    ];

    this.fingertipLandmarks = [4, 8, 12, 16, 20];
  }

  async init() {
    if (!this.container) return;

    const width = this.container.clientWidth || 360;
    const height = this.container.clientHeight || 280;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0A192F);

    // Frame the hand comfortably with padding
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    this.camera.position.set(0, 0, 3.0);
    this.camera.lookAt(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    this._tmpV1 = new THREE.Vector3();
    this._tmpV2 = new THREE.Vector3();
    this._tmpDir = new THREE.Vector3();
    this._upVec = new THREE.Vector3(0, 1, 0);

    // Warm, soft three-point lighting
    const ambientLight = new THREE.AmbientLight(0xFFEADB, 0.75);
    this.scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xFFF0E0, 1.0);
    keyLight.position.set(2, 3, 3);
    this.scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x80A0FF, 0.45);
    fillLight.position.set(-3, 0, 2);
    this.scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0x00E5FF, 0.6);
    rimLight.position.set(0, -2, -3);
    this.scene.add(rimLight);

    // Hand Group
    this.handGroup = new THREE.Group();
    this.scene.add(this.handGroup);

    // Natural warm skin material with organic roughness
    const skinMaterial = new THREE.MeshStandardMaterial({
      color: 0xE8A082,
      roughness: 0.52,
      metalness: 0.03
    });

    // 1. Organic Fleshy Palm Body
    const palmGeo = new THREE.CylinderGeometry(0.20, 0.22, 0.09, 20);
    this.palmPad = new THREE.Mesh(palmGeo, skinMaterial);
    this.handGroup.add(this.palmPad);

    // Thenar thumb muscle pad
    const thenarGeo = new THREE.SphereGeometry(0.14, 16, 16);
    this.thenarPad = new THREE.Mesh(thenarGeo, skinMaterial);
    this.handGroup.add(this.thenarPad);

    // Hypothenar pinky muscle pad
    const hypoGeo = new THREE.SphereGeometry(0.12, 16, 16);
    this.hypothenarPad = new THREE.Mesh(hypoGeo, skinMaterial);
    this.handGroup.add(this.hypothenarPad);

    // 2. Metacarpals (connecting wrist 0 to finger roots)
    for (const def of this.metacarpalDefs) {
      const geo = new THREE.CylinderGeometry(def.r, def.r, 1, 14);
      const mesh = new THREE.Mesh(geo, skinMaterial);
      this.handGroup.add(mesh);
      this.metacarpalSegments.push({ mesh, def });
    }

    // 3. Knuckles & Rounded Fingertips
    for (let i = 0; i < 21; i++) {
      const radius = this.fingertipLandmarks.includes(i) ? 0.044 : 0.055;
      const geo = new THREE.SphereGeometry(radius, 16, 16);
      const mesh = new THREE.Mesh(geo, skinMaterial);
      this.handGroup.add(mesh);
      this.knuckleSpheres.push(mesh);
    }

    // 4. Volumetric Finger Segments
    for (const def of this.segmentDefs) {
      const geo = new THREE.CylinderGeometry(def.r2, def.r1, 1, 14);
      const mesh = new THREE.Mesh(geo, skinMaterial);
      this.handGroup.add(mesh);
      this.fingerSegments.push({ mesh, def });
    }

    // Attach 360-degree orbit rotation controls
    this._attachControls();

    // Load canonical 3D poses
    try {
      const res = await fetch(this.posesUrl);
      if (res.ok) {
        this.poses = await res.json();
        console.log(`Carregadas ${Object.keys(this.poses).length} poses 3D volumetricas!`);
        if (this.poses["_OPEN"]) {
          this.setPose("_OPEN");
        }
      }
    } catch (e) {
      console.warn("Nao foi possivel carregar poses 3D:", e);
    }

    this._animate();
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
    const poseData = this.poses[letterOrName];
    if (!poseData) return;

    this.activeSignData = poseData;
    this.isPlayingMotion = false; // Idle posture by default
    if (this.onMotionStateChange) this.onMotionStateChange(false);

    // Reset rotation: Palm facing directly toward user
    this.handGroup.rotation.set(0, 0, 0);

    // Initial idle pose is the apex of the sign
    const target = poseData.landmarks || (poseData.frames ? poseData.frames[Math.floor(poseData.frames.length / 2)] : null);
    if (target) {
      this.currentPose = JSON.parse(JSON.stringify(target));
    }
  }

  playMotion() {
    if (!this.activeSignData) return;
    this.isPlayingMotion = true;
    this.motionStartTime = performance.now();
    if (this.onMotionStateChange) this.onMotionStateChange(true);
  }

  pauseMotion() {
    this.isPlayingMotion = false;
    if (this.onMotionStateChange) this.onMotionStateChange(false);
  }

  toggleMotion() {
    if (this.isPlayingMotion) {
      this.pauseMotion();
    } else {
      this.playMotion();
    }
  }

  _updateSegment(mesh, p1, p2) {
    this._tmpV1.set(p1[0], p1[1], p1[2]);
    this._tmpV2.set(p2[0], p2[1], p2[2]);
    const dist = this._tmpV1.distanceTo(this._tmpV2);

    mesh.scale.set(1, Math.max(dist, 1e-6), 1);
    mesh.position.copy(this._tmpV1).add(this._tmpV2).multiplyScalar(0.5);

    if (dist > 1e-6) {
      this._tmpDir.subVectors(this._tmpV2, this._tmpV1).divideScalar(dist);
      mesh.quaternion.setFromUnitVectors(this._upVec, this._tmpDir);
    }
  }

  _lerpPoses(poseA, poseB, t) {
    const clampedT = Math.max(0, Math.min(1, t));
    const ease = clampedT * clampedT * (3 - 2 * clampedT);
    for (let i = 0; i < 21; i++) {
      for (let axis = 0; axis < 3; axis++) {
        this.currentPose[i][axis] = poseA[i][axis] + (poseB[i][axis] - poseA[i][axis]) * ease;
      }
    }
  }

  _animate() {
    requestAnimationFrame(() => this._animate());

    if (!this.currentPose || !this.activeSignData) return;

    // Execute movement animation when triggered
    if (this.isPlayingMotion) {
      const elapsedMs = performance.now() - this.motionStartTime;

      // 1. Dynamic Greeting Sequences (trajectory playback)
      if (this.activeSignData.frames && this.activeSignData.frames.length > 1) {
        const frames = this.activeSignData.frames;
        const totalFrames = frames.length;
        const fps = 20;
        const frameDuration = 1000 / fps;
        const totalDuration = totalFrames * frameDuration + 1000;
        const loopTime = elapsedMs % totalDuration;

        if (loopTime < totalFrames * frameDuration) {
          const exactFrame = loopTime / frameDuration;
          const idxA = Math.floor(exactFrame);
          const idxB = Math.min(totalFrames - 1, idxA + 1);
          const frac = exactFrame - Math.floor(exactFrame);
          this._lerpPoses(frames[idxA], frames[idxB], frac);
        } else {
          const apex = this.activeSignData.landmarks || frames[Math.floor(totalFrames / 2)];
          this._lerpPoses(apex, apex, 1.0);
        }
      }
      // 2. Alphabet Letters: Open Hand -> Fold into Sign -> Hold -> Open Hand!
      else if (this.activeSignData.landmarks) {
        const neutralPose = (this.poses["_OPEN"] && this.poses["_OPEN"].landmarks) || this.activeSignData.landmarks;
        const targetPose = this.activeSignData.landmarks;

        const cycleMs = 3400;
        const cycleTime = elapsedMs % cycleMs;

        if (cycleTime < 900) {
          this._lerpPoses(neutralPose, targetPose, cycleTime / 900);
        } else if (cycleTime < 2200) {
          this._lerpPoses(targetPose, targetPose, 1.0);
        } else if (cycleTime < 2900) {
          this._lerpPoses(targetPose, neutralPose, (cycleTime - 2200) / 700);
        } else {
          this._lerpPoses(neutralPose, neutralPose, 1.0);
        }
      }
    }

    const pose = this.currentPose;

    // 1. Update Organic Fleshy Palm Pads
    if (this.palmPad) {
      const p0 = pose[0];
      const p9 = pose[9];
      this.palmPad.position.set((p0[0] + p9[0]) * 0.5, (p0[1] + p9[1]) * 0.5, (p0[2] + p9[2]) * 0.5);
    }
    if (this.thenarPad) {
      const p0 = pose[0];
      const p1 = pose[1];
      this.thenarPad.position.set((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5, (p0[2] + p1[2]) * 0.5);
    }
    if (this.hypothenarPad) {
      const p0 = pose[0];
      const p17 = pose[17];
      this.hypothenarPad.position.set((p0[0] + p17[0]) * 0.5, (p0[1] + p17[1]) * 0.5, (p0[2] + p17[2]) * 0.5);
    }

    // 2. Update Metacarpals (Palm internal structure)
    for (const { mesh, def } of this.metacarpalSegments) {
      this._updateSegment(mesh, pose[def.a], pose[def.b]);
    }

    // 3. Update Knuckles and rounded fingertips
    for (let i = 0; i < 21; i++) {
      const pt = pose[i];
      this.knuckleSpheres[i].position.set(pt[0], pt[1], pt[2]);
    }

    // 4. Update Volumetric Finger Segments
    for (const { mesh, def } of this.fingerSegments) {
      this._updateSegment(mesh, pose[def.a], pose[def.b]);
    }

    // Subtle breathing float when not dragging
    if (!this.isDragging) {
      this.handGroup.rotation.y = Math.sin(performance.now() * 0.001) * 0.05;
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
