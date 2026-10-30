/**
 * Feature Extractor replicating the exact Python mathematical preprocessing:
 * 1. Dactilologia: 225 rotation, translation, and scale invariant geometric features.
 * 2. Temporal: 128 normalized coordinate features for sequence classification.
 */

// 210 pairwise pairs for 21 hand landmarks
const PAIRWISE_PAIRS = [];
for (let i = 0; i < 21; i++) {
  for (let j = i + 1; j < 21; j++) {
    PAIRWISE_PAIRS.push([i, j]);
  }
}

// 15 anatomical triplet angles across 5 fingers
const ANGLE_TRIPLETS = [
  [1, 2, 3], [2, 3, 4], [0, 1, 2],
  [0, 5, 6], [5, 6, 7], [6, 7, 8],
  [0, 9, 10], [9, 10, 11], [10, 11, 12],
  [0, 13, 14], [13, 14, 15], [14, 15, 16],
  [0, 17, 18], [17, 18, 19], [18, 19, 20]
];

class FeatureExtractor {
  /**
   * Extract 225 invariant features from 21 MediaPipe hand landmarks.
   * @param {Array<{x: number, y: number, z: number}>} landmarks 
   * @param {boolean} isLeftHand 
   * @returns {Float32Array} 225-element vector
   */
  static extractAlphabetFeatures(landmarks, isLeftHand = false) {
    if (!landmarks || landmarks.length !== 21) {
      return new Float32Array(225);
    }

    // 1. Center relative to wrist (landmark 0)
    const wrist = landmarks[0];
    const centered = [];
    for (let i = 0; i < 21; i++) {
      let x = landmarks[i].x - wrist.x;
      const y = landmarks[i].y - wrist.y;
      const z = (landmarks[i].z || 0) - (wrist.z || 0);

      // Invert X for left hand to mirror canonical right-hand sign space
      if (isLeftHand) {
        x = -x;
      }
      centered.push([x, y, z]);
    }

    // 2. Scale by distance between wrist (0) and middle finger MCP (9)
    const mcp = centered[9];
    let scale = Math.sqrt(mcp[0] * mcp[0] + mcp[1] * mcp[1] + mcp[2] * mcp[2]);
    if (scale < 1e-6) scale = 1.0;

    const normalized = centered.map(([x, y, z]) => [x / scale, y / scale, z / scale]);

    // 3. 210 Pairwise Euclidean distances
    const distances = new Float32Array(210);
    for (let idx = 0; idx < PAIRWISE_PAIRS.length; idx++) {
      const [i, j] = PAIRWISE_PAIRS[idx];
      const pi = normalized[i];
      const pj = normalized[j];
      const dx = pi[0] - pj[0];
      const dy = pi[1] - pj[1];
      const dz = pi[2] - pj[2];
      distances[idx] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    // 4. 15 Anatomical 3D joint angles
    const angles = new Float32Array(15);
    for (let idx = 0; idx < ANGLE_TRIPLETS.length; idx++) {
      const [a, b, c] = ANGLE_TRIPLETS[idx];
      const pa = normalized[a];
      const pb = normalized[b];
      const pc = normalized[c];

      const u = [pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]];
      const v = [pc[0] - pb[0], pc[1] - pb[1], pc[2] - pb[2]];

      const dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
      const normU = Math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
      const normV = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      const den = Math.max(normU * normV, 1e-6);

      const cosine = Math.max(-1.0, Math.min(1.0, dot / den));
      angles[idx] = Math.acos(cosine);
    }

    // Concat 210 distances + 15 angles = 225 features
    const output = new Float32Array(225);
    output.set(distances, 0);
    output.set(angles, 210);
    return output;
  }

  /**
   * Extract 128 normalized features for sequential GRU.
   * 64 floats for left hand, 64 floats for right hand.
   */
  static extractFrame128(leftLandmarks, rightLandmarks) {
    const out = new Float32Array(128);

    const presentHands = [];
    if (leftLandmarks && leftLandmarks.length === 21) presentHands.push(leftLandmarks);
    if (rightLandmarks && rightLandmarks.length === 21) presentHands.push(rightLandmarks);

    if (presentHands.length === 0) {
      return out;
    }

    // Calculate center as mean of wrists
    let centerX = 0, centerY = 0, centerZ = 0;
    for (const h of presentHands) {
      centerX += h[0].x;
      centerY += h[0].y;
      centerZ += (h[0].z || 0);
    }
    centerX /= presentHands.length;
    centerY /= presentHands.length;
    centerZ /= presentHands.length;

    // Calculate max radius scale
    let maxDist = 0;
    for (const h of presentHands) {
      for (const p of h) {
        const dx = p.x - centerX;
        const dy = p.y - centerY;
        const dz = (p.z || 0) - centerZ;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist > maxDist) maxDist = dist;
      }
    }
    const scale = Math.max(maxDist, 1e-6);

    function fillHand(landmarks, offset) {
      if (!landmarks || landmarks.length !== 21) {
        return;
      }
      for (let i = 0; i < 21; i++) {
        out[offset + i * 3] = (landmarks[i].x - centerX) / scale;
        out[offset + i * 3 + 1] = (landmarks[i].y - centerY) / scale;
        out[offset + i * 3 + 2] = ((landmarks[i].z || 0) - centerZ) / scale;
      }
      out[offset + 63] = 1.0; // Presence mask
    }

    fillHand(leftLandmarks, 0);
    fillHand(rightLandmarks, 64);
    return out;
  }
}

window.FeatureExtractor = FeatureExtractor;
