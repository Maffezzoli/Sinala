/**
 * ONNX Runtime Web runner for local client-side inference.
 * Operates entirely inside the browser using WASM/WebGL.
 */
class OnnxRunner {
  constructor() {
    this.alphabetSession = null;
    this.greetingsSession = null;
    this.alphabetClasses = [];
    this.greetingsClasses = [];
    this.isReady = false;
  }

  async loadModels(modelsDir = "models") {
    // 1. Load metadata sidecars
    try {
      const alphaMetaRes = await fetch(`${modelsDir}/alphabet_metadata.json`);
      if (alphaMetaRes.ok) {
        const meta = await alphaMetaRes.json();
        this.alphabetClasses = meta.classes || [];
      } else {
        this.alphabetClasses = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
      }

      const greetMetaRes = await fetch(`${modelsDir}/greetings_metadata.json`);
      if (greetMetaRes.ok) {
        const meta = await greetMetaRes.json();
        this.greetingsClasses = meta.classes || [];
      }
    } catch (e) {
      console.warn("Nao foi possivel carregar metadata JSON, usando defaults:", e);
      this.alphabetClasses = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
      this.greetingsClasses = ["Ajudar", "Cumprimento", "Desculpa", "Não", "Nome", "Obrigado", "Oi", "Por favor", "Sim"];
    }

    // 2. Load ONNX sessions via ort (ONNX Runtime Web)
    const sessionOptions = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    };

    this.alphabetSession = await ort.InferenceSession.create(`${modelsDir}/alphabet.onnx`, sessionOptions);
    try {
      this.greetingsSession = await ort.InferenceSession.create(`${modelsDir}/greetings.onnx`, sessionOptions);
    } catch (e) {
      console.warn("Modelo de saudacoes nao carregado:", e);
    }

    this.isReady = true;
    console.log("Modelos ONNX carregados com sucesso!");
  }

  softmax(logits) {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > max) max = logits[i];
    }
    const exps = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      exps[i] = Math.exp(logits[i] - max);
      sum += exps[i];
    }
    for (let i = 0; i < logits.length; i++) {
      exps[i] /= sum;
    }
    return exps;
  }

  async predictAlphabet(features225) {
    if (!this.alphabetSession) return { letter: "?", confidence: 0, margin: 0 };

    const tensorIn = new ort.Tensor("float32", features225, [1, 225]);
    const results = await this.alphabetSession.run({ features: tensorIn });
    const outputTensor = results.logits || Object.values(results)[0];
    const logits = outputTensor.data;
    const probs = this.softmax(logits);

    // Find top 2 predictions
    const indexed = Array.from(probs).map((p, i) => ({ prob: p, index: i }));
    indexed.sort((a, b) => b.prob - a.prob);

    const top1 = indexed[0];
    const top2 = indexed[1] || { prob: 0, index: 0 };
    const letter = this.alphabetClasses[top1.index] || "?";
    const confidence = top1.prob;
    const margin = top1.prob - top2.prob;

    return {
      letter,
      confidence,
      margin,
      topCandidates: indexed.slice(0, 3).map(item => ({
        label: this.alphabetClasses[item.index],
        prob: item.prob
      }))
    };
  }

  async predictGreeting(sequence32x128) {
    if (!this.greetingsSession) return { label: "?", confidence: 0, margin: 0 };

    const tensorIn = new ort.Tensor("float32", sequence32x128, [1, 32, 128]);
    const results = await this.greetingsSession.run({ sequence: tensorIn });
    const outputTensor = results.logits || Object.values(results)[0];
    const logits = outputTensor.data;
    const probs = this.softmax(logits);

    const indexed = Array.from(probs).map((p, i) => ({ prob: p, index: i }));
    indexed.sort((a, b) => b.prob - a.prob);

    const top1 = indexed[0];
    const top2 = indexed[1] || { prob: 0, index: 0 };
    const label = this.greetingsClasses[top1.index] || "?";
    const confidence = top1.prob;
    const margin = top1.prob - top2.prob;

    return {
      label,
      confidence,
      margin,
      topCandidates: indexed.slice(0, 3).map(item => ({
        label: this.greetingsClasses[item.index],
        prob: item.prob
      }))
    };
  }
}

window.OnnxRunner = OnnxRunner;
