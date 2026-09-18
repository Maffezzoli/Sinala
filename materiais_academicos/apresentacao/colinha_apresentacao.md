# Roteiro de Apresentação e Guia dos 10 Artigos (Projeto SINALA)

Este documento acompanha a apresentação oficial **`SINALA_Revisao_Literatura_Academico_Profissional.pptx`** (14 slides). Cada integrante da equipe deve apresentar pelo menos um artigo (tempo sugerido: **3 a 4 minutos por referência**), destacando:
1. O **Modelo Computacional Utilizado** (explícito);
2. Os **Resultados Quantitativos** (métricas e acurácia);
3. Os **Resultados Qualitativos** (vantagens e limitações práticas);
4. O **Papel no Projeto SINALA** (o que herdamos ou superamos).

## Arquitetura implementada atualmente

O SINALA não usa um único classificador. O modo unificado combina dois modelos especialistas:

1. **Dactilologia A–Z:** MLP em PyTorch com 225 features geométricas invariantes (210 distâncias articulares + 15 ângulos 3D), saída de 26 letras.
2. **Sinais inteiros e saudações:** GRU temporal unidirecional em PyTorch, entrada de 32 frames × 128 features, hidden size 64, dropout 0,2 e saída de 9 classes conversacionais.

O `camera-unified` extrai landmarks uma única vez, encaminha a mão estável para o MLP e trajetórias com movimento para a GRU, e reúne os resultados no mesmo `WordBuilder`.

---

## Quadro Rápido dos 10 Artigos e Modelos Utilizados

| Ref. | Autor e Ano | Veículo | Modelo Computacional Utilizado | Métrica Chave | Função no SINALA |
|:---:|---|---|---|:---:|---|
| **1** | Furtado et al. (2023) | IEEE I2MTC | **Descritores de Contorno (CDD) + Distância Euclidiana** | 96,05% | Captura 100% *markerless* em webcam RGB |
| **2** | Feliciano et al. (2023) | IEEE CISTI | **MediaPipe Hands + Perceptron Multicamadas (MLP / Keras)** | 92–98% | Extração de esqueleto para filtrar ruído de fundo |
| **3** | Arcanjo et al. (2024) | SBC WebMedia | **MediaPipe Holistic + Ângulos 3D + FastDTW ($O(N)$)** | 86% LOOCV | **Diretriz inegociável de split por sinalizador** |
| **4** | Castro et al. (2023) | Elsevier ESWA | **Multi-Stream 3D CNN + GAN (Profundidade Sintética)** | 91% (F1 0,90) | Riqueza 3D sem depender de sensor Kinect físico |
| **5** | Rodrigues et al. (2025) | SBC ENIAC | **Benchmark CNNs 2D (ResNet18 vs. VGG/Inception/DenseNet)** | 74,29% | Evidência de que redes 2D exigem camada temporal |
| **6** | Pereira et al. (2026) | SBC WICS | **Rede Siamesa com BERTimbau + MultipleNegativesRankingLoss** | R@1 98,77% | Fundação do Nível 3 de tradução semântica em texto |
| **7** | Silva et al. (2020) | ACM WebMedia | **Two-Stream 3D CNN (Fluxo Espacial RGB + Fluxo Óptico TV-L1)** | 88,9% | Relevância de separar aparência e velocidade |
| **8** | Passos et al. (2021) | IEEE TCSI | **Gait Energy Image (GEI) + PCA/LDA + SVM / LightGBM** | 98,8% | Redução extrema de redundância de dados de vídeo |
| **9** | Rezende et al. (2021) | Springer NCA | **CNNs 2D e 3D Multimodais (MINDS-Libras com RGB-D)** | 93% ± 2% | **Criação do dataset oficial adotado no SINALA** |
| **10** | Rego et al. (2025) | IEEE Access | **LSTM / BiLSTM + FFT Adaptativa + Parâmetros Cinemáticos** | 92,0% | Enriquecimento de trajetórias com derivadas de movimento |

---

## Detalhamento Slide por Slide

### Slide 1 — Capa
- **Título:** *Da detecção de sinais à tradução em tempo real*
- **Subtítulo:** *Revisão de Literatura e posicionamento metodológico do MVP (10 Referências Brasileiras • Modelos Explícitos)*
- **Apresentador:** Daniel Maffezzoli / Grupo SINALA.

---

### Slide 2 — Problema e Estratégia do SINALA
- **Diagnóstico:** 10 milhões de pessoas com deficiência auditiva no Brasil. A Libras possui gramática própria e não aceita tradução literal palavra a palavra. Luvas e sensores corporais são caros, frágeis e desconfortáveis.
- **Roteiro em 3 Níveis:**
  - **Nível 1 (MVP Atual):** Reconhecimento de sinais isolados com GRU temporal e dactilologia com MLP geométrica, usando MediaPipe, validação inter-sinalizador e buffer de estabilidade.
  - **Nível 2:** Segmentação temporal contínua e fusão multimodal (mãos, pose e face).
  - **Nível 3:** Tradução semântica via modelos de linguagem (Libras $\rightarrow$ Português correto) e síntese em tempo real.

---

### Slide 3 — Ref. 1: Furtado, de Oliveira e Shirmohammadi (2023)
- **Identificação:** IEEE I2MTC 2023 • IME / LNCC.
- **Título:** *Interactive and Markerless Visual Recognition of Brazilian Sign Language Alphabet*.
- **MODELO UTILIZADO:** **Descritores de Contorno (CDD) + Distância Euclidiana**.
- **Metodologia:** Câmera RGB convencional de computador; descritores de distância centróide (CDD) para contorno das mãos; dispensa completa de luvas ou marcadores físicos.
- **Quantitativo:** **96,05% de acurácia global** no alfabeto dactilológico; **+6,4%** de ganho sobre cinco métodos da literatura; **+8,6%** sobre gestos genéricos.
- **Qualitativo:** Prova a viabilidade de captura não invasiva em webcams normais. Limitação: cobre apenas letras paradas, sem capturar a trajetória de sinais dinâmicos.
- **O que o SINALA herda:** A premissa de um sistema acessível e barato em webcam comum.

---

### Slide 4 — Ref. 2: Feliciano, Briano, Prudêncio e Alves (2023)
- **Identificação:** IEEE CISTI 2023 • IFPE / UFPE.
- **Título:** *Recognition of Static or Dynamic LIBRAS Words in Complex Background Environments*.
- **MODELO UTILIZADO:** **MediaPipe Hands + Perceptron Multicamadas (MLP / Keras)**.
- **Metodologia:** Pipeline de dois estágios (OpenCV $\rightarrow$ MediaPipe $\rightarrow$ TensorFlow/Keras) para palavras estáticas e dinâmicas de Libras gravadas em ambientes reais com fundos poluídos.
- **Quantitativo:** **92% a 98% de acurácia** em iluminação padrão; **95% preservados sob baixa luminosidade**; recuo para $\approx 82\%$ em fundos muito ruidosos.
- **Qualitativo:** A extração esquelética desacopla a mão da cor da parede. Limitação: movimentos bruscos causam *motion blur* e perda de tracking.
- **O que o SINALA herda:** A validação do MediaPipe e a necessidade do nosso buffer de histerese para absorver oscilações transitórias do esqueleto.

---

### Slide 5 — Ref. 3: Arcanjo, Coelho, Guimarães et al. (2024) [Artigo Central]
- **Identificação:** SBC WebMedia 2024 • PUC Minas.
- **Título:** *Automatic Time-Aware Recognition of Brazilian Sign Language Based on Dynamic Time Warping*.
- **MODELO UTILIZADO:** **MediaPipe Holistic + Ângulos 3D + FastDTW ($O(N)$)**.
- **Metodologia:** Opera sobre a mesma base de 20 classes do SINALA (MINDS-Libras). Extrai 21 marcos por mão e 33 corporais, calcula ângulos trigonométricos entre falanges e alinha sequências temporais via FastDTW linear.
- **Quantitativo (A maior lição metodológica):**
  - **Split Aleatório (75/25):** **98% $\pm$ 1%** de acurácia (ilusão de precisão por decoreba de fisionomia).
  - **LOOCV por Sinalizador:** Despenca para **86% $\pm$ 8%** (e classes ambíguas como *Banheiro* caem para 47%).
- **Qualitativo:** Prova científica de que colocar a mesma pessoa no treino e teste infla artificialmente as métricas.
- **O que o SINALA herda:** A diretriz metodológica inegociável: teste estrito inter-sinalizador para garantir generalização real em novos usuários.

---

### Slide 6 — Ref. 4: Castro, Guerra e Guimarães (2023)
- **Identificação:** Elsevier ESWA 2023 • UFMG (Laboratório MINDS).
- **Título:** *Automatic Translation of Sign Language with Multi-Stream 3D CNN and Generation of Artificial Depth Maps*.
- **MODELO UTILIZADO:** **Multi-Stream 3D CNN + GAN (Profundidade Sintética)**.
- **Metodologia:** Pesquisadores criadores do MINDS-Libras eliminam sensores físicos Kinect gerando mapas de profundidade artificiais com GANs, alimentando uma rede convolucional 3D multi-stream com mãos, face e velocidades.
- **Quantitativo:** **91% $\pm$ 7% de acurácia**, F1 de **0,90 $\pm$ 0,08** em Libras. Ganho estatístico ao adicionar a profundidade sintética.
- **Qualitativo:** Desempata sobreposições de mãos na frente do corpo. Limitação: alto custo computacional de 3D CNNs + GANs, inviável para 30 FPS em CPU modesta.
- **O que o SINALA herda:** A confirmação de que coordenadas $(x, y, z)$ são vitais, mas nós as obtemos diretamente por landmarks vetoriais leves com custo irrisório de CPU.

---

### Slide 7 — Ref. 5: Rodrigues, Barbosa, Fiera e Oliveira (2025)
- **Identificação:** SBC ENIAC 2025 • UNESC.
- **Título:** *Dynamic Sign Recognition in Brazilian Sign Language Through Convolutional Neural Networks*.
- **MODELO UTILIZADO:** **Benchmark de 5 CNNs 2D (ResNet18 vs. AlexNet, VGG16, InceptionV3, DenseNet121)**.
- **Metodologia:** Estudo comparativo de redes convolucionais clássicas aplicadas a sinais dinâmicos da Libras com variação temporal de movimento.
- **Quantitativo:** **ResNet18 líder com 74,29% de acurácia**; modelos maiores como Inception e DenseNet sofreram com *overfitting* ($< 70\%$).
- **Qualitativo:** Evidencia o teto das convoluções 2D estáticas em sinais dinâmicos: tratar vídeo como imagens isoladas perde vetores de aceleração e cronologia.
- **O que o SINALA herda:** Justifica por que o SINALA utiliza obrigatoriamente a camada temporal recorrente (GRU) para superar a barreira dos 74%.

---

### Slide 8 — Ref. 6: Pereira, Santana, Martins e Corrêa (2026)
- **Identificação:** SBC WICS 2026 • UFPel.
- **Título:** *Semantic Vector Space Mapping Between Brazilian Portuguese and Libras Glosses Using Siamese Models*.
- **MODELO UTILIZADO:** **Rede Siamesa com BERTimbau + MultipleNegativesRankingLoss**.
- **Metodologia:** Mapeamento de espaço vetorial semântico compartilhado entre glosas de Libras e sentenças em português brasileiro, treinado via aprendizado contrastivo com *mean pooling*.
- **Quantitativo:** **Recall@1 de 98,77%** e **MRR de 99,38%** no ranking de sentenças em português a partir de glosas.
- **Qualitativo:** Respeita a identidade linguística da Libras, evitando traduções literais truncadas. Limitação: opera na camada textual, dependendo de extrator visual prévio.
- **O que o SINALA herda:** A fundação do Nível 3: nosso classificador visual fornecerá as entradas para alimentar um alinhador semântico como o do grupo da UFPel.

---

### Slide 9 — Ref. 7: da Silva, Araujo, do Rêgo e Brandão (2020)
- **Identificação:** ACM WebMedia 2020 • UFPB.
- **Título:** *A Two-Stream Model Based on 3D Convolutional Neural Networks for the Recognition of Brazilian Sign Language in the Health Context*.
- **MODELO UTILIZADO:** **Two-Stream 3D CNN (Fluxo Espacial RGB + Fluxo Óptico TV-L1)**.
- **Metodologia:** Reconhecimento aplicado a sinais médicos e clínicos de Libras em contexto hospitalar. Processa dois fluxos simultâneos: aparência espacial (frames RGB) e dinâmica de movimento (fluxo óptico denso TV-L1). Avaliação por *leave-one-signer-out*.
- **Quantitativo:** **88,9% de acurácia** em termos hospitalares de Libras; superou a 3D CNN de corrente única em **+5,2%**.
- **Qualitativo:** Prova a importância de desacoplar a informação de aparência estática da mão da dinâmica temporal de velocidade do gesto. Limitação: cálculo de fluxo óptico denso é muito pesado para tempo real leve.
- **O que o SINALA herda:** No SINALA, as derivadas temporais dos próprios marcos anatômicos substituem o fluxo óptico com custo nulo de CPU.

---

### Slide 10 — Ref. 8: Passos, Araujo, Gois e de Lima (2021)
- **Identificação:** IEEE TCSI 2021 • CEFET/RJ / UFF.
- **Título:** *A Gait Energy Image-Based System for Brazilian Sign Language Recognition*.
- **MODELO UTILIZADO:** **Gait Energy Image (GEI) + PCA/LDA + SVM / LightGBM**.
- **Metodologia:** Condensa sequências inteiras de silhuetas temporais em uma única imagem de energia acumulada média ponderada (GEI). Aplica redução de dimensionalidade via PCA + LDA para classificação rápida por SVM ou LightGBM.
- **Quantitativo:** **98,8% de acurácia** em sinais isolados de Libras; **redução de 99,4% no volume de dados brutos** de vídeo para treinamento e inferência.
- **Qualitativo:** Eficiência extrema para baratear o custo computacional. Limitação: a imagem média perde a cronologia interna do gesto (movimentos de ida e volta geram a mesma imagem).
- **O que o SINALA herda:** Redes recorrentes (como a nossa GRU) são indispensáveis para distinguir sinais com trajetórias simétricas ou reversíveis.

---

### Slide 11 — Ref. 9: Rezende, Almeida e Guimarães (2021)
- **Identificação:** Springer NCA 2021 • UFMG / IFMG.
- **Título:** *Development and Validation of a Brazilian Sign Language Database for Human Gesture Recognition* (Artigo canônico do MINDS-Libras!).
- **MODELO UTILIZADO:** **CNNs 2D e 3D Multimodais (InceptionV3 / VGG16 / 3D-CNN com canais RGB e Depth)**.
- **Metodologia:** Criação, anotação e validação experimental da principal base de dados pública de sinais isolados de Libras do Brasil (MINDS-Libras: 1.200 vídeos gravados em estúdio por 12 sinalizadores em 20 classes semânticas). Avalia canais RGB e profundidade física do Kinect v2.
- **Quantitativo:** **93% $\pm$ 2% de acurácia** com fusão multimodal (RGB + profundidade física).
- **Qualitativo:** Estabeleceu o vocabulário oficial, as 20 classes padronizadas e o protocolo de validação que sustentam a literatura brasileira. Limitação: dependia do sensor Kinect v2, que foi descontinuado.
- **O que o SINALA herda:** O dataset oficial e o vocabulário adotados no nosso MVP, extraindo a tridimensionalidade dos marcos puramente a partir de vídeo RGB de webcam via MediaPipe.

---

### Slide 12 — Ref. 10: Rego, de Morais e Almeida (2025)
- **Identificação:** IEEE Access 2025 • UFC / IFCE.
- **Título:** *Brazilian Sign Language Recognition Using Deep Learning Based on Fast Fourier Transform and Kinematic Features*.
- **MODELO UTILIZADO:** **LSTM / BiLSTM + FFT Adaptativa de Janela Deslizante + Cinemática (Velocidade e Aceleração)**.
- **Metodologia:** Enriquecimento de trajetórias articulares esqueléticas com derivadas de velocidade e aceleração e extração de coeficientes espectrais em frequência via FFT adaptativa sobre o dataset MINDS-Libras.
- **Quantitativo:** **92,0% de acurácia e 92% de F1-Score** no MINDS-Libras, superando a LSTM padrão (84%) em **+8,0%** e a BiLSTM pura (89%) em **+3,0%**.
- **Qualitativo:** Comprova que derivadas temporais de velocidade e aceleração enriquecem a discriminação de sinais complexos. Limitação: a FFT de janela deslizante adiciona latência de buffer antes da classificação.
- **O que o SINALA herda:** A memória interna da GRU do SINALA captura esse ritmo temporal das transições com latência de apenas $1{,}4\,\text{ms}$ por predição em CPU.

---

### Slide 13 — Quadro Sinótico dos 10 Artigos
- Tabela comparativa com colunas: **TRABALHO | DOMÍNIO | MODELO UTILIZADO | RESULTADO | CONTRIBUIÇÃO AO SINALA**.
- Permite ao professor e à banca visualizarem instantaneamente a coerência da literatura e o papel de cada arquitetura na construção do SINALA.

---

### Slide 14 — Posicionamento do Projeto SINALA e Conclusão
- **Três Pilares do SINALA:**
  1. *Generalização Real:* Separação estrita inter-sinalizador para medir pessoas nunca vistas.
  2. *Baixo Custo & Tempo Real:* MediaPipe + GRU em CPU comum (30 FPS / 1,4 ms de inferência).
  3. *Evolução Modular:* Transição planejada do Nível 1 ao Nível 3.
- **O Modelo Oficial do SINALA:**
  - `MediaPipe Hands + MLP geométrica (225D) para A–Z + GRU temporal (32×128) para sinais inteiros + WordBuilder`.
- **Agradecimento final.**
