/**
 * DuoLibras Application Controller.
 * Manages gamification state, MediaPipe Hands pipeline, and interactive practice flow.
 */

const MODULE_LESSONS = {
  alphabet: [
    {
      id: "alpha_1",
      title: "Vogais Fundamentais",
      subtitle: "Aprenda A, E, I, O, U",
      avatar: "assets/animals/sapo.png",
      animal: "Sapo-chifrudo",
      challenges: [
        { type: "alphabet", target: "A", hint: "Feche a mão em punho com o polegar encostado na lateral do indicador." },
        { type: "alphabet", target: "E", hint: "Dobre os quatro dedos para baixo, apoiando as pontas no polegar recolhido." },
        { type: "alphabet", target: "I", hint: "Feche a mão mantendo apenas o dedo mínimo estendido para cima." },
        { type: "alphabet", target: "O", hint: "Una todas as pontas dos dedos com o polegar formando um círculo perfeito." },
        { type: "alphabet", target: "U", hint: "Mantenha o indicador e o médio estendidos e bem juntos para cima." }
      ]
    },
    {
      id: "alpha_2",
      title: "Primeiras Consoantes",
      subtitle: "Aprenda B, C, D, L, M",
      avatar: "assets/animals/tracaja.png",
      animal: "Tracajá",
      challenges: [
        { type: "alphabet", target: "B", hint: "Quatro dedos estendidos para cima bem juntos e o polegar dobrado na palma." },
        { type: "alphabet", target: "C", hint: "Curve a mão simulando o formato da letra C." },
        { type: "alphabet", target: "D", hint: "Indicador estendido para cima; os outros dedos tocam a ponta do polegar formando um 'd'." },
        { type: "alphabet", target: "L", hint: "Polegar e indicador estendidos formando um ângulo de 90 graus (formato de L)." },
        { type: "alphabet", target: "M", hint: "Três dedos (indicador, médio e anelar) apontados para baixo." }
      ]
    },
    {
      id: "alpha_3",
      title: "Consoantes com Calma",
      subtitle: "Aprenda F, G, P, Q, V",
      avatar: "assets/animals/preguica.png",
      animal: "Bicho-preguiça",
      challenges: [
        { type: "alphabet", target: "F", hint: "Indicador dobrado para baixo com o polegar encostado por FORA dele." },
        { type: "alphabet", target: "G", hint: "Indicador apontado para cima com o polegar estendido ao lado." },
        { type: "alphabet", target: "P", hint: "Dedo médio dobrado para frente, indicador estendido e polegar apoiado entre eles." },
        { type: "alphabet", target: "Q", hint: "Indicador e polegar apontados para baixo como uma pinça." },
        { type: "alphabet", target: "V", hint: "Indicador e médio abertos em formato de V da vitória." }
      ]
    },
    {
      id: "alpha_4",
      title: "Consoantes Ágeis",
      subtitle: "Aprenda R, S, T, W, Y",
      avatar: "assets/animals/macaco.png",
      animal: "Macaco-de-cheiro",
      challenges: [
        { type: "alphabet", target: "R", hint: "Cruze o dedo médio por trás do indicador estendido para cima." },
        { type: "alphabet", target: "S", hint: "Punho fechado com o polegar dobrado cruzando a frente dos dedos." },
        { type: "alphabet", target: "T", hint: "Indicador dobrado com o polegar enfiado por DENTRO dele." },
        { type: "alphabet", target: "W", hint: "Três dedos estendidos para cima formando a letra W." },
        { type: "alphabet", target: "Y", hint: "Polegar e mindinho estendidos para os lados (sinal do hang loose)." }
      ]
    },
    {
      id: "alpha_5",
      title: "Desafio Mestre da Mata",
      subtitle: "Soletrando Palavras da Amazônia",
      avatar: "assets/animals/onca.png",
      animal: "Onça-pintada",
      challenges: [
        { type: "alphabet", target: "B", word: "BOLA", hint: "Primeira letra da palavra BOLA: faça a letra B." },
        { type: "alphabet", target: "O", word: "BOLA", hint: "Segunda letra da palavra BOLA: faça a letra O." },
        { type: "alphabet", target: "L", word: "BOLA", hint: "Terceira letra da palavra BOLA: faça a letra L." },
        { type: "alphabet", target: "A", word: "BOLA", hint: "Última letra da palavra BOLA: faça a letra A." }
      ]
    }
  ],
  greetings: [
    {
      id: "greet_1",
      title: "Saudações do Rio",
      subtitle: "Oi, Obrigado e Por Favor",
      avatar: "assets/animals/boto.png",
      animal: "Boto-cor-de-rosa",
      challenges: [
        { type: "greeting", target: "Oi", hint: "Levante a mão com o mínimo estendido (letra I) e balance suavemente em arco." },
        { type: "greeting", target: "Obrigado", hint: "Toque a testa com a mão aberta e leve-a para frente em agradecimento." },
        { type: "greeting", target: "Por favor", hint: "Palmas das mãos unidas em frente ao peito com movimento circular suave." }
      ]
    },
    {
      id: "greet_2",
      title: "Respostas em Família",
      subtitle: "Sim, Não e Desculpa",
      avatar: "assets/animals/ariranha.png",
      animal: "Ariranha",
      challenges: [
        { type: "greeting", target: "Sim", hint: "Mão em punho fechado (letra S) articulando para baixo duas vezes como um aceno." },
        { type: "greeting", target: "Não", hint: "Indicador e médio estendidos batendo contra o polegar abrindo e fechando." },
        { type: "greeting", target: "Desculpa", hint: "Mão em letra Y com o polegar tocando o queixo suavemente." }
      ]
    },
    {
      id: "greet_3",
      title: "Expressões da Mata",
      subtitle: "Nome e Cumprimento",
      avatar: "assets/animals/tucano.png",
      animal: "Tucano-toco",
      challenges: [
        { type: "greeting", target: "Nome", hint: "Indicador e médio em formato de U deitados, tocando a palma da outra mão." },
        { type: "greeting", target: "Cumprimento", hint: "Mãos abertas com dedos abertos tocando-se à frente do corpo." }
      ]
    },
    {
      id: "greet_4",
      title: "Amizade & Cuidado",
      subtitle: "Ajudar e Cortesia",
      avatar: "assets/animals/tamandua.png",
      animal: "Tamanduá-bandeira",
      challenges: [
        { type: "greeting", target: "Ajudar", hint: "Uma mão fechada em punho com polegar para cima empurrada para frente pela outra palma." },
        { type: "greeting", target: "Por favor", hint: "Palmas unidas suavemente deslizando à frente do peito." }
      ]
    },
    {
      id: "greet_5",
      title: "Celebração das Cores",
      subtitle: "Revisão Geral de Saudações",
      avatar: "assets/animals/arara.png",
      animal: "Arara-vermelha",
      challenges: [
        { type: "greeting", target: "Oi", hint: "Mindinho para cima fazendo o aceno clássico de Oi em Libras!" },
        { type: "greeting", target: "Obrigado", hint: "Mão na testa saindo para frente em agradecimento solene." },
        { type: "greeting", target: "Sim", hint: "Aceno afirmativo com punho em S!" }
      ]
    }
  ]
};

class DuoLibrasApp {
  constructor() {
    this.audio = new AudioSynth();
    this.onnx = new OnnxRunner();
    this.hand3d = new Hand3DViewer("hand-3d-container");

    this.currentModule = "alphabet";
    this.currentLesson = null;
    this.challengeIndex = 0;

    this.greetingSigner = 1;
    this.greetingPlaybackRate = 1.0;
    // Gamification persistence
    this.xp = parseInt(localStorage.getItem("duolibras_xp") || "0", 10);
    this.hearts = parseInt(localStorage.getItem("duolibras_hearts") || "5", 10);
    this.streak = parseInt(localStorage.getItem("duolibras_streak") || "1", 10);
    this.completedLessons = new Set(JSON.parse(localStorage.getItem("duolibras_completed") || "[]"));

    // MediaPipe & Webcam
    this.handLandmarker = null;
    this.webcamRunning = false;
    this.videoElement = document.getElementById("webcam-video");
    this.canvasElement = document.getElementById("webcam-canvas");
    this.canvasCtx = this.canvasElement ? this.canvasElement.getContext("2d") : null;

    // Temporal sequence buffer for greetings (32 frames)
    this.temporalBuffer = [];
    this.loopGeneration = 0;
    // Hold verification timer (must hold correct sign for 1.2s)
    this.holdFrames = 0;
    this.REQUIRED_HOLD_FRAMES = 12; // ~0.8s to 1.2s
    this.isChallengeResolved = false;

    this.HAND_CONNECTIONS = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [5, 9], [9, 10], [10, 11], [11, 12],
      [9, 13], [13, 14], [14, 15], [15, 16],
      [13, 17], [17, 18], [18, 19], [19, 20],
      [0, 17]
    ];
  }

  async init() {
    this.updateStatsUI();
    this.renderLearningPath();
    this.bindEvents();

    // Load 3D Viewer & ONNX Models in parallel
    await Promise.all([
      this.hand3d.init(),
      this.onnx.loadModels("models")
    ]);

    // Initialize MediaPipe Vision Tasks HandLandmarker
    await this.initMediaPipe();
  }

  async initMediaPipe() {
    try {
      const VisionModule = window.Vision || window;
      const FilesetResolver = VisionModule.FilesetResolver || window.FilesetResolver;
      const HandLandmarker = VisionModule.HandLandmarker || window.HandLandmarker;

      if (!FilesetResolver || !HandLandmarker) {
        throw new Error("Modulo MediaPipe Vision nao foi encontrado no escopo global.");
      }

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );

      // Try local model asset first, fallback to CDN URL
      let taskUrl = "models/hand_landmarker.task";
      try {
        const check = await fetch(taskUrl, { method: "HEAD" });
        if (!check.ok) taskUrl = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
      } catch {
        taskUrl = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
      }

      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: taskUrl,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      console.log("MediaPipe HandLandmarker inicializado com GPU!");
      document.getElementById("webcam-status-text").textContent = "IA Pronta";
    } catch (e) {
      console.warn("GPU delegate falhou no MediaPipe, tentando CPU:", e);
      try {
        const VisionModule = window.Vision || window;
        const FilesetResolver = VisionModule.FilesetResolver || window.FilesetResolver;
        const HandLandmarker = VisionModule.HandLandmarker || window.HandLandmarker;
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "models/hand_landmarker.task",
            delegate: "CPU"
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
        console.log("MediaPipe HandLandmarker pronto com fallback CPU!");
        document.getElementById("webcam-status-text").textContent = "IA Pronta (CPU)";
      } catch (err) {
        console.error("Falha fatal ao carregar MediaPipe:", err);
        document.getElementById("webcam-status-text").textContent = "Erro na IA";
      }
    }
  }

  updateStatsUI() {
    document.getElementById("stat-xp").textContent = `${this.xp} XP`;
    document.getElementById("stat-hearts").textContent = this.hearts;
    document.getElementById("stat-streak").textContent = `${this.streak}d`;

    localStorage.setItem("duolibras_xp", this.xp.toString());
    localStorage.setItem("duolibras_hearts", this.hearts.toString());
    localStorage.setItem("duolibras_streak", this.streak.toString());
    localStorage.setItem("duolibras_completed", JSON.stringify(Array.from(this.completedLessons)));
  }

  renderLearningPath() {
    const container = document.getElementById("learning-path-container");
    container.innerHTML = "";

    const lessons = MODULE_LESSONS[this.currentModule];
    const isGreetings = this.currentModule === "greetings";

    // Section Header Banner
    // Section Header Banner with Amazon Fauna Art
    const sectionCard = document.createElement("div");
    sectionCard.className = `section-card ${isGreetings ? "greetings" : ""}`;
    const bannerArt = isGreetings ? "assets/animals/boto.png" : "assets/animals/sapo.png";
    sectionCard.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 16px;">
        <div>
          <div class="section-title">${isGreetings ? "Módulo 2: Cumprimentos & Expressões" : "Módulo 1: Alfabeto Manual (Dactilologia)"}</div>
          <div class="section-subtitle">${isGreetings ? "Aprenda saudações com a Ariranha e o Boto" : "Domine as 26 letras com os animais da Amazônia"}</div>
        </div>
        <img src="${bannerArt}" alt="Fauna Amazônica" class="section-banner-art">
      </div>
    `;
    container.appendChild(sectionCard);

    // Render Nodes
    lessons.forEach((lesson, index) => {
      const isCompleted = this.completedLessons.has(lesson.id);
      const isLocked = index > 0 && !this.completedLessons.has(lessons[index - 1].id) && !isCompleted;

      const nodeWrapper = document.createElement("div");
      nodeWrapper.className = "path-node-wrapper";

      const nodeBtn = document.createElement("button");
      nodeBtn.className = `path-node ${isCompleted ? "completed" : ""} ${isLocked ? "locked" : ""}`;
      
      if (isLocked) {
        nodeBtn.innerHTML = `
          <img src="${lesson.avatar}" alt="${lesson.animal}" class="node-asset-img locked">
          <div class="node-lock-badge">🔒</div>
        `;
      } else if (isCompleted) {
        nodeBtn.innerHTML = `
          <img src="${lesson.avatar}" alt="${lesson.animal}" class="node-asset-img">
          <img src="assets/ui/star.png" alt="Concluído" class="node-star-badge">
        `;
      } else {
        nodeBtn.innerHTML = `<img src="${lesson.avatar}" alt="${lesson.animal}" class="node-asset-img">`;
      }

      if (!isLocked) {
        nodeBtn.onclick = () => {
          this.audio.playClick();
          this.startLesson(lesson);
        };
      }

      const label = document.createElement("div");
      label.className = "node-label";
      label.innerHTML = `<span>${lesson.title}</span><div class="node-sublabel">${lesson.animal}</div>`;

      nodeWrapper.appendChild(nodeBtn);
      nodeWrapper.appendChild(label);
      container.appendChild(nodeWrapper);
    });
  }

  bindEvents() {
    // Module Tabs
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.onclick = () => {
        this.audio.playClick();
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.currentModule = btn.dataset.module;
        this.renderLearningPath();
      };
    });

    // Close Practice View
    document.getElementById("btn-close-practice").onclick = () => {
      this.audio.playClick();
      this.stopPractice();
    };

    // Continue Next Challenge Button
    document.getElementById("btn-continue-feedback").onclick = () => {
      this.audio.playClick();
      document.getElementById("feedback-drawer").classList.remove("active", "success", "error");
      this.advanceChallenge();
    };

    // Play / Pause Motion Button (3D Hand for Alphabet, Training Video for Greetings)
    const playBtn = document.getElementById("btn-play-motion");
    if (playBtn) {
      playBtn.onclick = () => {
        this.audio.playClick();
        const activeChallenge = this.currentLesson && this.currentLesson.challenges[this.challengeIndex];
        if (activeChallenge && activeChallenge.type === "greeting") {
          const refVid = document.getElementById("greeting-ref-video");
          if (refVid) {
            if (refVid.paused) {
              refVid.play().catch(() => {});
              this._updateMotionButtonUI(true, "Reproduzir");
            } else {
              refVid.pause();
              this._updateMotionButtonUI(false, "Reproduzir");
            }
          }
        } else {
          this.hand3d.toggleMotion();
        }
      };
    }

    this.hand3d.onMotionStateChange = (isPlaying) => {
      const activeChallenge = this.currentLesson && this.currentLesson.challenges[this.challengeIndex];
      if (!activeChallenge || activeChallenge.type === "alphabet") {
        this._updateMotionButtonUI(isPlaying, "Ver Movimento");
      }
    };

    // Greeting Training Video Controls (Articulador 1 / 2 / 3 + Speed 1x / 0.5x)
    const signerPills = document.querySelectorAll(".video-pill[data-signer]");
    signerPills.forEach((pill) => {
      pill.onclick = () => {
        this.audio.playClick();
        signerPills.forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");
        this.greetingSigner = parseInt(pill.dataset.signer, 10) || 1;
        const activeChallenge = this.currentLesson && this.currentLesson.challenges[this.challengeIndex];
        if (activeChallenge && activeChallenge.type === "greeting") {
          this._loadGreetingTrainingVideo(activeChallenge.target);
        }
      };
    });

    const speedBtn = document.getElementById("btn-video-speed");
    if (speedBtn) {
      speedBtn.onclick = () => {
        this.audio.playClick();
        this.greetingPlaybackRate = this.greetingPlaybackRate === 1.0 ? 0.5 : 1.0;
        speedBtn.textContent = this.greetingPlaybackRate === 1.0 ? "1x" : "0.5x";
        speedBtn.classList.toggle("slow", this.greetingPlaybackRate < 1.0);
        const refVid = document.getElementById("greeting-ref-video");
        if (refVid) refVid.playbackRate = this.greetingPlaybackRate;
      };
    }

    // Hand Color / Skin Tone Customizer
    const savedColor = localStorage.getItem("duolibras_hand_color") || "#E8A082";
    const swatches = document.querySelectorAll(".swatch-btn");
    swatches.forEach((swatch) => {
      if (swatch.dataset.color.toLowerCase() === savedColor.toLowerCase()) {
        swatches.forEach((s) => s.classList.remove("active"));
        swatch.classList.add("active");
      }
      swatch.onclick = () => {
        this.audio.playClick();
        swatches.forEach((s) => s.classList.remove("active"));
        swatch.classList.add("active");
        const color = swatch.dataset.color;
        this.hand3d.setColor(color);
        const customInput = document.getElementById("hand-custom-color");
        if (customInput) customInput.value = color;
      };
    });

    const customColorInput = document.getElementById("hand-custom-color");
    if (customColorInput) {
      customColorInput.value = savedColor;
      customColorInput.oninput = (e) => {
        const color = e.target.value;
        swatches.forEach((s) => s.classList.remove("active"));
        this.hand3d.setColor(color);
      };
    }

    // Custom In-App Completion Modal Close Button
    const modalCloseBtn = document.getElementById("btn-modal-close");
    if (modalCloseBtn) {
      modalCloseBtn.onclick = () => {
        this.audio.playClick();
        document.getElementById("duo-modal").classList.remove("active");
        this.stopPractice();
      };
    }
  }

  async startLesson(lesson) {
    this.currentLesson = lesson;
    this.challengeIndex = 0;
    this.isChallengeResolved = false;

    document.getElementById("dashboard-view").style.display = "none";
    document.getElementById("practice-view").classList.add("active");

    await this.startWebcam();
    this.loadChallenge(this.challengeIndex);
  }

  stopPractice() {
    this.stopWebcam();
    const refVid = document.getElementById("greeting-ref-video");
    if (refVid) {
      refVid.pause();
      refVid.removeAttribute("src");
    }
    document.getElementById("practice-view").classList.remove("active");
    document.getElementById("feedback-drawer").classList.remove("active");
    document.getElementById("dashboard-view").style.display = "block";
    this.renderLearningPath();
  }

  async startWebcam() {
    if (this.webcamRunning) return;
    try {
      this.loopGeneration++;
      const currentGen = this.loopGeneration;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } }
      });
      this.videoElement.srcObject = stream;
      await this.videoElement.play();
      this.webcamRunning = true;
      document.getElementById("webcam-status-text").textContent = "Câmera Ativa";
      document.getElementById("webcam-status-dot").classList.add("active");
      this.predictLoop(currentGen);
    } catch (e) {
      console.error("Nao foi possivel acessar a webcam:", e);
      document.getElementById("webcam-status-text").textContent = "Câmera Bloqueada";
      const statusDot = document.getElementById("webcam-status-dot");
      if (statusDot) statusDot.className = "status-dot";
    }
  }

  stopWebcam() {
    this.loopGeneration++;
    if (this.videoElement && this.videoElement.srcObject) {
      this.videoElement.srcObject.getTracks().forEach((track) => track.stop());
      this.videoElement.srcObject = null;
    }
    this.webcamRunning = false;
  }

  loadChallenge(index) {
    const challenge = this.currentLesson.challenges[index];
    this.isChallengeResolved = false;
    this.holdFrames = 0;
    this.updateHoldProgress(0);

    // Update Progress bar
    const progress = (index / this.currentLesson.challenges.length) * 100;
    document.getElementById("lesson-progress-bar").style.width = `${progress}%`;

    // Title & Instructions
    const titleEl = document.getElementById("challenge-title");
    const hintEl = document.getElementById("challenge-hint");

    const hand3dBox = document.getElementById("hand-3d-container");
    const handPalette = document.getElementById("hand-color-palette");
    const videoBox = document.getElementById("greeting-video-container");
    const videoControls = document.getElementById("greeting-video-controls");
    const panelTitle = document.getElementById("ref-panel-title");
    const panelHint = document.getElementById("ref-panel-hint");

    if (challenge.type === "alphabet") {
      titleEl.innerHTML = `Faça o sinal da letra <span style="color: var(--primary);">${challenge.target}</span>`;
      hintEl.textContent = challenge.hint;

      const refVid = document.getElementById("greeting-ref-video");
      if (refVid) refVid.pause();
      if (videoBox) videoBox.style.display = "none";
      if (videoControls) videoControls.style.display = "none";
      if (hand3dBox) hand3dBox.style.display = "block";
      if (handPalette) handPalette.style.display = "flex";
      if (panelTitle) panelTitle.textContent = "Mão 3D";
      if (panelHint) panelHint.textContent = "Gire 360° com o mouse";

      this.hand3d.resize();
      this.hand3d.setPose(challenge.target);
    } else {
      titleEl.innerHTML = `Faça o sinal de <span style="color: var(--blue);">${challenge.target}</span>`;
      hintEl.textContent = challenge.hint;

      this.hand3d.pauseMotion();
      if (hand3dBox) hand3dBox.style.display = "none";
      if (handPalette) handPalette.style.display = "none";
      if (videoBox) videoBox.style.display = "block";
      if (videoControls) videoControls.style.display = "flex";
      if (panelTitle) panelTitle.textContent = "Vídeo do Treinamento";
      if (panelHint) panelHint.textContent = "Dataset V-Librasil";

      this._loadGreetingTrainingVideo(challenge.target);
    }
    document.getElementById("recognition-badge").textContent = "-";
  }


  _updateMotionButtonUI(isPlaying, idleLabel = "Ver Movimento") {
    const iconEl = document.getElementById("motion-icon");
    const textEl = document.getElementById("motion-text");
    const btn = document.getElementById("btn-play-motion");
    if (!btn) return;
    if (isPlaying) {
      btn.classList.add("playing");
      if (iconEl) iconEl.textContent = "⏸";
      if (textEl) textEl.textContent = "Pausar";
    } else {
      btn.classList.remove("playing");
      if (iconEl) iconEl.textContent = "▶";
      if (textEl) textEl.textContent = idleLabel;
    }
  }

  _loadGreetingTrainingVideo(targetSign) {
    const refVid = document.getElementById("greeting-ref-video");
    const badge = document.getElementById("greeting-video-badge");
    if (!refVid) return;
    const fileName = `${targetSign}_Articulador${this.greetingSigner}.mp4`;
    refVid.src = `training-videos/${encodeURIComponent(fileName)}`;
    refVid.playbackRate = this.greetingPlaybackRate;
    if (badge) {
      badge.textContent = `Articulador ${this.greetingSigner} · Dataset de Treinamento`;
    }
    refVid.play().then(() => {
      this._updateMotionButtonUI(true, "Reproduzir");
    }).catch(() => {
      this._updateMotionButtonUI(false, "Reproduzir");
    });
  }
  updateHoldProgress(percent) {
    document.getElementById("hold-progress-fill").style.width = `${percent}%`;
  }

  async predictLoop(generation = this.loopGeneration) {
    if (!this.webcamRunning || generation !== this.loopGeneration) return;

    if (this.handLandmarker && this.videoElement.readyState >= 2) {
      const targetW = this.videoElement.videoWidth || 640;
      const targetH = this.videoElement.videoHeight || 480;
      if (this.canvasElement.width !== targetW || this.canvasElement.height !== targetH) {
        this.canvasElement.width = targetW;
        this.canvasElement.height = targetH;
      } else {
        this.canvasCtx.clearRect(0, 0, targetW, targetH);
      }

      const startTimeMs = performance.now();
      const results = this.handLandmarker.detectForVideo(this.videoElement, startTimeMs);
      if (results.landmarks && results.landmarks.length > 0) {
        const statusDot = document.getElementById("webcam-status-dot");
        const statusText = document.getElementById("webcam-status-text");
        if (statusDot) statusDot.className = "status-dot active";
        if (statusText) statusText.textContent = "Mão Detectada";

        // Draw Skeleton
        for (const landmarks of results.landmarks) {
          this.drawLandmarks(landmarks);
        }

        // Run Classifier
        if (!this.isChallengeResolved) {
          await this.evaluateChallenge(results);
        }
      } else {
        // No hand detected
        const statusDot = document.getElementById("webcam-status-dot");
        const statusText = document.getElementById("webcam-status-text");
        if (statusDot) statusDot.className = "status-dot";
        if (statusText) statusText.textContent = "Aguardando Mão";

        this.holdFrames = Math.max(0, this.holdFrames - 1);
        this.updateHoldProgress(0);
        document.getElementById("recognition-badge").textContent = "-";
      }
    }

    requestAnimationFrame(() => this.predictLoop(generation));
  }

  drawLandmarks(landmarks) {
    const w = this.canvasElement.width;
    const h = this.canvasElement.height;

    // Draw Bones
    this.canvasCtx.strokeStyle = "#00E5FF";
    this.canvasCtx.lineWidth = 3.5;
    for (const [i, j] of this.HAND_CONNECTIONS) {
      const p1 = landmarks[i];
      const p2 = landmarks[j];
      this.canvasCtx.beginPath();
      this.canvasCtx.moveTo(p1.x * w, p1.y * h);
      this.canvasCtx.lineTo(p2.x * w, p2.y * h);
      this.canvasCtx.stroke();
    }

    // Draw Joint Dots
    for (const p of landmarks) {
      this.canvasCtx.fillStyle = "#0052FF";
      this.canvasCtx.beginPath();
      this.canvasCtx.arc(p.x * w, p.y * h, 5, 0, 2 * Math.PI);
      this.canvasCtx.fill();
      this.canvasCtx.strokeStyle = "#FFFFFF";
      this.canvasCtx.lineWidth = 1.5;
      this.canvasCtx.stroke();
    }
  }

  async evaluateChallenge(results) {
    const challenge = this.currentLesson.challenges[this.challengeIndex];
    const primaryHand = results.landmarks[0];
    const handednessList = results.handedness || results.handednesses || [];
    const isLeft = handednessList[0] && handednessList[0][0] && handednessList[0][0].categoryName === "Left";
    if (challenge.type === "alphabet") {
      const features = FeatureExtractor.extractAlphabetFeatures(primaryHand, isLeft);
      const rawPred = await this.onnx.predictAlphabet(features);
      const prediction = FeatureExtractor.refineAlphabetPrediction(primaryHand, isLeft, rawPred);
      document.getElementById("recognition-badge").textContent = prediction.letter;

      // Match target
      if (prediction.letter === challenge.target && prediction.confidence >= 0.40) {
        this.holdFrames++;
        const percent = Math.min(100, Math.round((this.holdFrames / this.REQUIRED_HOLD_FRAMES) * 100));
        this.updateHoldProgress(percent);

        if (this.holdFrames >= this.REQUIRED_HOLD_FRAMES) {
          this.triggerSuccess();
        }
      } else {
        this.holdFrames = Math.max(0, this.holdFrames - 1);
        this.updateHoldProgress((this.holdFrames / this.REQUIRED_HOLD_FRAMES) * 100);
      }
    } else {
      // Greetings handling: combine geometric posture analysis with temporal sequence
      const leftHand = isLeft ? primaryHand : null;
      const rightHand = !isLeft ? primaryHand : null;
      const frame128 = FeatureExtractor.extractFrame128(leftHand, rightHand);
      this.temporalBuffer.push(frame128);
      if (this.temporalBuffer.length > 32) this.temporalBuffer.shift();

      // Analyze instant hand configuration using Alphabet model
      const alphaFeats = FeatureExtractor.extractAlphabetFeatures(primaryHand, isLeft);
      const rawAlphaPred = await this.onnx.predictAlphabet(alphaFeats);
      const alphaPred = FeatureExtractor.refineAlphabetPrediction(primaryHand, isLeft, rawAlphaPred);

      let isMatch = false;
      let displayBadge = challenge.target.substring(0, 2).toUpperCase();

      // Check GRU prediction if buffer has sufficient frames
      if (this.temporalBuffer.length >= 16) {
        const flat = new Float32Array(32 * 128);
        const startOffset = Math.max(0, 32 - this.temporalBuffer.length);
        for (let i = 0; i < this.temporalBuffer.length; i++) {
          flat.set(this.temporalBuffer[i], (startOffset + i) * 128);
        }
        const prediction = await this.onnx.predictGreeting(flat);
        if (prediction.label === challenge.target && prediction.confidence >= 0.35) {
          isMatch = true;
          displayBadge = prediction.label.substring(0, 2).toUpperCase();
        }
      }

      // Semantic posture rules for Libras greetings:
      // "Oi": Hand in letter 'I' (pinky up) or 'O'
      if (challenge.target === "Oi") {
        if (alphaPred.letter === "I" || alphaPred.letter === "O" || alphaPred.letter === "J") {
          isMatch = true;
          displayBadge = "OI";
        }
      }
      // "Sim": Hand in letter 'S' / 'A' (fist)
      else if (challenge.target === "Sim") {
        if (alphaPred.letter === "S" || alphaPred.letter === "A" || alphaPred.letter === "E") {
          isMatch = true;
          displayBadge = "SIM";
        }
      }
      // "Não": Hand pinching with index/middle
      else if (challenge.target === "Não") {
        if (alphaPred.letter === "U" || alphaPred.letter === "N" || alphaPred.letter === "D") {
          isMatch = true;
          displayBadge = "NÃO";
        }
      }
      // "Desculpa": Hand in letter 'Y'
      else if (challenge.target === "Desculpa") {
        if (alphaPred.letter === "Y") {
          isMatch = true;
          displayBadge = "DESC";
        }
      }

      document.getElementById("recognition-badge").textContent = isMatch ? displayBadge : (alphaPred.letter || "-");

      if (isMatch) {
        this.holdFrames += 2;
        const percent = Math.min(100, Math.round((this.holdFrames / this.REQUIRED_HOLD_FRAMES) * 100));
        this.updateHoldProgress(percent);

        if (this.holdFrames >= this.REQUIRED_HOLD_FRAMES) {
          this.triggerSuccess();
        }
      } else {
        this.holdFrames = Math.max(0, this.holdFrames - 1);
        this.updateHoldProgress((this.holdFrames / this.REQUIRED_HOLD_FRAMES) * 100);
      }
    }
  }

  triggerSuccess() {
    this.isChallengeResolved = true;
    this.audio.playSuccess();
    this.xp += 10;
    this.updateStatsUI();

    const drawer = document.getElementById("feedback-drawer");
    drawer.className = "feedback-drawer active success";
    document.getElementById("feedback-title").textContent = "Excelente trabalho!";
    document.getElementById("feedback-subtitle").textContent = "Sinal reconhecido com precisão. +10 XP";
  }

  advanceChallenge() {
    this.challengeIndex++;
    if (this.challengeIndex < this.currentLesson.challenges.length) {
      this.loadChallenge(this.challengeIndex);
    } else {
      // Completed full lesson - Trigger custom in-app modal (zero native popups)
      this.audio.playLevelUp();
      this.completedLessons.add(this.currentLesson.id);
      this.xp += 50;
      this.updateStatsUI();
      this.showCompletionModal(this.currentLesson.title, 50);
    }
  }

  showCompletionModal(lessonTitle, bonusXp) {
    const modal = document.getElementById("duo-modal");
    const titleEl = document.getElementById("duo-modal-title");
    const subEl = document.getElementById("duo-modal-subtitle");

    if (titleEl) titleEl.textContent = "Lição Concluída!";
    if (subEl) subEl.textContent = `Parabéns! Você dominou todos os sinais de "${lessonTitle}"!`;
    if (modal) modal.classList.add("active");
  }
}

window.DuoLibrasApp = DuoLibrasApp;
