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

  static _fingerState(p, mcp, pip, dip, tip) {
    const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    const norm = (a) => Math.hypot(a[0], a[1], a[2]);
    const segSum = d(p[pip], p[mcp]) + d(p[dip], p[pip]) + d(p[tip], p[dip]);
    const ratio = d(p[tip], p[mcp]) / Math.max(segSum, 1e-6);
    const rMcp = norm(p[mcp]);
    const rTip = norm(p[tip]);
    return {
      ext: ratio > 0.75 && rTip > rMcp * 1.20,
      fold: ratio < 0.68 || rTip < rMcp * 1.15
    };
  }

  /**
   * Biomechanical minimal-pair disambiguation (U vs V vs R vs N, M vs W, G vs L vs Q).
   */
  static refineAlphabetPrediction(landmarks, isLeftHand, prediction) {
    if (!landmarks || landmarks.length !== 21 || !prediction || prediction.letter === "?") {
      return prediction;
    }
    const wrist = landmarks[0];
    const rel = landmarks.map((lm) => [
      (isLeftHand ? -(lm.x - wrist.x) : lm.x - wrist.x),
      lm.y - wrist.y,
      (lm.z || 0) - (wrist.z || 0)
    ]);
    const scale = Math.hypot(rel[9][0], rel[9][1], rel[9][2]);
    if (scale < 1e-6) return prediction;
    const p = rel.map(([x, y, z]) => [x / scale, y / scale, z / scale]);

    const idx = FeatureExtractor._fingerState(p, 5, 6, 7, 8);
    const mid = FeatureExtractor._fingerState(p, 9, 10, 11, 12);
    const rng = FeatureExtractor._fingerState(p, 13, 14, 15, 16);
    const pnk = FeatureExtractor._fingerState(p, 17, 18, 19, 20);

    let refined = prediction.letter;

    // 1. Two-finger family: Index & Middle extended, Ring & Pinky folded
    if (idx.ext && mid.ext && rng.fold && pnk.fold) {
      const dyIdx = p[8][1] - p[5][1];
      const dyMid = p[12][1] - p[9][1];
      if (dyIdx > 0.45 && dyMid > 0.45) {
        refined = "N";
      } else if (dyIdx < -0.55 && dyMid < -0.55 && p[8][1] < -1.35 && p[12][1] < -1.35) {
        const dThumb6 = Math.hypot(p[4][0] - p[6][0], p[4][1] - p[6][1], p[4][2] - p[6][2]);
        const dThumb10 = Math.hypot(p[4][0] - p[10][0], p[4][1] - p[10][1], p[4][2] - p[10][2]);
        if (!((refined === "K" || refined === "H") && Math.min(dThumb6, dThumb10) < 0.36)) {
          const mcpDir = (p[5][0] - p[9][0]) >= 0 ? 1.0 : -1.0;
          const signedSep = (p[8][0] - p[12][0]) * mcpDir;
          const dXY = Math.hypot(p[8][0] - p[12][0], p[8][1] - p[12][1]);
          const v1x = p[8][0] - p[5][0], v1y = p[8][1] - p[5][1];
          const v2x = p[12][0] - p[9][0], v2y = p[12][1] - p[9][1];
          const cosA = (v1x * v2x + v1y * v2y) / Math.max(Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y), 1e-6);
          const angDeg = Math.acos(Math.max(-1.0, Math.min(1.0, cosA))) * (180 / Math.PI);
          if (signedSep < -0.02) {
            refined = "R";
          } else if (dXY >= 0.24 || angDeg >= 7.2) {
            refined = "V";
          } else {
            refined = "U";
          }
        }
      }
    }
    // 2. Three-finger family: M (down) vs W (up)
    else if (idx.ext && mid.ext && rng.ext && pnk.fold) {
      if (p[8][1] - p[5][1] > 0.45 && p[12][1] - p[9][1] > 0.45) refined = "M";
      else if (p[8][1] - p[5][1] < -0.45 && p[12][1] - p[9][1] < -0.45) refined = "W";
    }
    // 3. Index-only family: Q (down) vs L (90-deg thumb)
    else if (idx.ext && mid.fold && rng.fold && pnk.fold) {
      if (p[8][1] - p[5][1] > 0.55) refined = "Q";
      else if (Math.hypot(p[4][0] - p[6][0], p[4][1] - p[6][1]) > 0.75) refined = "L";
    }

    if (refined !== prediction.letter) {
      return { ...prediction, letter: refined, confidence: Math.max(prediction.confidence, 0.85) };
    }
    return prediction;
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
