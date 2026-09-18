# Entregas Acadêmicas — Semana 5 e Semana 6 (Projeto SINALA)

Este documento consolida integralmente os requisitos das **Semanas 5 e 6** da disciplina, articulando a parte escrita do artigo científico, o esqueleto (*outline*), a apresentação das referências bibliográficas, o planejamento experimental e a documentação qualitativa do protótipo em bancada.

---

# PARTE 1: SEMANA 5

## 1.1 Apresentação das Referências Visitadas
- **Arquivo Oficial de Apresentação:** `materiais_academicos/apresentacao/SINALA_Revisao_Literatura.pptx`
- **Roteiro Oral de Fala (Colinha):** `materiais_academicos/apresentacao/colinha_apresentacao.md`
- **Diretriz de Apresentação:** Cada integrante da equipe apresenta ao menos um artigo (tempo estimado: **3 a 4 minutos por referência**), cobrindo obrigatoriamente metodologia, resultados quantitativos, resultados qualitativos, limitações e o impacto na engenharia do SINALA.

---

## 1.2 Outline do Artigo (Esqueleto Estrutural das Seções)

O esqueleto foi estruturado segundo as práticas editoriais canônicas de periódicos e simpósios da SBC, IEEE e Elsevier, espelhando a organização metodológica das 10 referências estudadas:

```text
TÍTULO: SINALA: Uma Abordagem Incremental do Reconhecimento de Sinais Isolados de Libras à Tradução em Tempo Real

1. INTRODUÇÃO
   1.1 Contextualização Social e Barreira Comunicacional (da Silva et al., 2020; Passos et al., 2021)
   1.2 Desafios Tecnológicos: Hardware Intrusivo vs. Visão Computacional (Rezende et al., 2021; Rego et al., 2025)
   1.3 Proposta do SINALA e Contribuições Explícitas
   1.4 Organização do Trabalho (Visita Guiada)

2. TRABALHOS RELACIONADOS
   2.1 Nível 1: Reconhecimento de Sinais Isolados de Libras
       - Abordagens sem marcadores em câmera comum (Furtado et al., 2023)
       - Rastreamento articular em fundos complexos (Feliciano et al., 2023)
       - Alinhamento temporal elástico e o viés do split aleatório (Arcanjo et al., 2024)
   2.2 Nível 2: Modelagem Espaço-Temporal e Tradução Semântica
       - Fusão multi-stream e geração de profundidade artificial (Castro et al., 2023)
       - O teto das redes convolucionais 2D em sinais dinâmicos (Rodrigues et al., 2025)
       - Mapeamento semântico de glosas para português via modelos siameses (Pereira et al., 2026)

3. DESCRIÇÃO DO SISTEMA EM DESENVOLVIMENTO
   3.1 Sensores Ópticos de Captura e Aquisição de Imagem (Furtado et al., 2023; Rezende et al., 2021)
   3.2 Plataforma Computacional de Processamento e Execução em Borda (Feliciano et al., 2023; Arcanjo et al., 2024)
   3.3 Protocolos de Comunicação e Pipeline de Tempo Real (UVC, Ring Buffer e Histerese)
   3.4 Montagem Física da Bancada e Condições de Operação

4. MÉTODO PROPOSTO
   4.1 Extração e Normalização de Landmarks Articulares (MediaPipe Hands, 21 pontos x 2 mãos)
   4.2 Aumento de Dados Espaço-Temporal com Preservação Estrita de Máscaras
   4.3 Classificador Sequencial Recorrente (GRU com Dropout e AdamW)
   4.4 Módulo de Estabilização Temporal de Legendas por Consenso (Debouncing)

5. PLANEJAMENTO E AVALIAÇÃO EXPERIMENTAL
   5.1 Configuração Experimental e Dataset de Referência (MINDS-Libras, 20 classes)
   5.2 Protocolo de Partição Inter-Sinalizador Estrito (Signer-Independent)
   5.3 Análise Comparativa: Generalização Real vs. Split Aleatório Viciado
   5.4 Avaliação Qualitativa do Protótipo em Tempo Real (Inspeção Visual da Cadeia de Percepção)
   5.5 Discussão de Resultados e Limitações Práticas

6. CONCLUSÕES E TRABALHOS FUTUROS
   6.1 Síntese das Conquistas do MVP (Nível 1)
   6.2 Roadmap para Reconhecimento Contínuo (Nível 2) e Tradução com LLMs (Nível 3)
```

---

## 1.3 Penúltimo Parágrafo da Introdução (Proposta e Contribuições)

*Texto redigido e inserido no arquivo `materiais_academicos/artigo/main.tex`:*

> "Para responder a essas demandas sem incorrer em promessas prematuras de tradução fluente irrestrita, este artigo propõe o sistema **SINALA**, articulado em torno de uma estratégia de desenvolvimento estritamente incremental. Focando, em seu estágio inicial, no reconhecimento robusto de sinais isolados de Libras sob regime de teste estrito em indivíduos completamente inéditos, o trabalho oferece as seguintes contribuições principais:
> (1) a concepção e implementação de um pipeline de visão computacional *markerless*, leve e modular, capaz de operar em câmeras RGB convencionais a 30 quadros por segundo em CPUs de uso geral, dispensando GPUs dedicadas de alto consumo;
> (2) a modelagem sequencial temporal com uma arquitetura GRU compacta otimizada com técnicas de regularização (AdamW, *label smoothing*, *dropout* e aumento espaço-temporal de dados), preservando a coerência das máscaras anatômicas em mãos ocluídas;
> (3) a formalização de um protocolo de avaliação inter-sinalizador estrito sobre o dataset público MINDS-Libras, demonstrando o impacto do viés de identidade presente em partições aleatórias tradicionais;
> (4) a introdução de um módulo de estabilização temporal por janela deslizante de consenso (*debouncing*), que suprime falsos disparos transitórios e gera legendas estáveis; e
> (5) a definição de um roteiro tecnológico fundamentado na literatura nacional para a transição planejada entre a classificação isolada, a segmentação contínua e a tradução semântica."

---

## 1.4 Último Parágrafo da Introdução (Visita Guiada / "Guided Tour")

*Texto redigido e inserido no arquivo `materiais_academicos/artigo/main.tex`:*

> "O restante deste trabalho está organizado da seguinte forma. A Seção \ref{sec:trabalhos_relacionados} examina a revisão da literatura recente sobre o reconhecimento de sinais e tradução de Libras, situando as abordagens em níveis de complexidade. A Seção \ref{sec:sistema} detalha o sistema em desenvolvimento, apresentando os sensores ópticos de captura, a plataforma computacional de processamento e os protocolos de comunicação adotados. A Seção \ref{sec:metodo} descreve formalmente o método proposto, abrangendo a extração articular, o aumento de dados, o classificador GRU e o buffer de estabilização. A Seção \ref{sec:experimentos} expõe o planejamento experimental, os protocolos de partição inter-sinalizador, a avaliação quantitativa preliminar e a análise qualitativa do protótipo em laboratório. Por fim, a Seção \ref{sec:conclusao} conclui o artigo e delineia os trabalhos futuros."

---

## 1.5 Registro Fotográfico do Protótipo (Avaliação Qualitativa)

Para atender à exigência de fotos/figuras comprovando o funcionamento do protótipo de visão computacional, foram capturados e processados 3 quadros reais do pipeline, armazenados em `materiais_academicos/artigo/figuras/`:

1. **`captura_camera_bruta.jpg`**: Quadro bruto obtido diretamente pelo sensor de imagem RGB da câmera sem qualquer pós-processamento, comprovando o enquadramento superior do sinalizador em plano médio.
2. **`deteccao_landmarks.jpg`**: Saída do detector articular MediaPipe Hands, mostrando os 21 pontos tridimensionais sobrepostos nas articulações das mãos e as conexões do esqueleto anatômico desenhadas em tempo real.
3. **`prototipo_inferencia_hud.jpg`**: Interface completa de execução em tempo real da aplicação (`RealtimeCaptionApp`), exibindo a caixa superior de status com:
   - **Legenda estável:** `"Bala"` (emitida após o consenso temporal do buffer);
   - **Índice de Confiança:** `0.94`;
   - **Vazão Operacional:** `29.8 FPS` (processamento fluido em CPU);
   - **Diagnóstico Articular:** `"Status: 2 mãos detectadas (Estabilidade: OK)"`.

---

# PARTE 2: SEMANA 6

## 2.1 Revisão do Título e da Introdução
- **Título Consolidado:** *SINALA: Uma Abordagem Incremental do Reconhecimento de Sinais Isolados de Libras à Tradução em Tempo Real*.
- **Introdução Polida:** Estruturada em exatamente 4 blocos coesos (Contextualização social com 2 referências $\rightarrow$ Problema tecnológico e motivação com 2 referências $\rightarrow$ Proposta e 5 contribuições explícitas $\rightarrow$ Visita guiada das seções).

---

## 2.2 Seção "Sistema em Desenvolvimento" (Componentes, Sensores e Protocolos)

*Texto implementado na Seção 3 do artigo (`main.tex`):*

### A. Sensores Ópticos de Captura (Hardware)
- **Tipo de Sensor:** Câmera óptica RGB digital convencional (sensor CMOS de obturador rolante), integrada ao dispositivo ou conectada via porta USB.
- **Resolução e Taxa de Amostragem:** $1280 \times 720$ (HD) ou $1920 \times 1080$ (Full HD) a 30 quadros por segundo ($\text{FPS} \approx 30$).
- **Distância Operacional:** Enquadramento frontal a uma distância de $0{,}5\,\text{m}$ a $1{,}2\,\text{m}$ do usuário, cobrindo cabeça, ombros, peito e o espaço tridimensional neutro de sinalização dos braços.
- **Fundamentação na Literatura:** O sensor puramente óptico substitui com sucesso luvas instrumentadas desconfortáveis e sensores inerciais de alto custo, conforme demonstrado por Furtado, de Oliveira e Shirmohammadi (2023) na dactilologia de Libras e por Rezende, Almeida e Guimarães (2021) na criação do corpus MINDS-Libras.

### B. Plataforma Computacional de Processamento (Microcontrolador / CPU)
- **Processador:** CPU multinúcleo comercial de uso geral (arquitetura ARM64 contemporânea, e.g. Apple Silicon M-series, ou processadores x86-64 Intel Core / AMD Ryzen).
- **Sem Dependência de GPU Dedicada:** A rede neural recorrente (GRU) possui apenas 128 unidades ocultas e consome menos de 2 milissegundos por predição em CPU pura.
- **Aceleração Articular:** O motor MediaPipe Tasks / TFLite utiliza operadores vetorizados leves (delegados XNNPACK / Metal), mantendo o consumo térmico e elétrico baixo para dispositivos pessoais, em consonância com Feliciano et al. (2023) e Arcanjo et al. (2024).

### C. Protocolos de Comunicação e Interfaces de Software
1. **Protocolo UVC (*USB Video Class*):** Padrão de comunicação universal para transmissão de vídeo síncrono entre a câmera e o sistema operacional, intermediado via driver V4L2 (Linux) ou AVFoundation (macOS) pela biblioteca OpenCV.
2. **Barramento de Fila Circular Temporal (*Ring Buffer*):** Estrutura de dados assíncrona em memória (*deque*) com capacidade fixa de $T = 32$ amostras temporais. Desacopla a taxa de quadros variável da webcam (25-30 Hz) do ciclo de inferência do classificador sequencial, garantindo que o modelo sempre classifique a janela mais recente do gesto.
3. **Protocolo de Histerese e Debouncing:** Máquina de estados finitos que filtra oscilações transitórias de probabilidade através de consenso temporal em janela deslizante ($K=4$ em $W=5$ quadros, $\tau \ge 0{,}75$, tempo de resfriamento $t_{\text{cool}} = 0{,}8\,\text{s}$).

---

## 2.3 Planejamento Experimental Detalhado

O planejamento experimental foi desenhado para isolar o viés de identidade e validar a capacidade real de generalização da tecnologia:

### 1. Corpus de Dados e Vocabulário
- **Base:** MINDS-Libras (1.200 gravações em vídeo Full HD de 12 sinalizadores).
- **Vocabulário (20 Classes):** *Acontecer, Aluno, Amarelo, América, Aproveitar, Bala, Banco, Banheiro, Barulho, Cinco, Conhecer, Espelho, Esquina, Filho, Maçã, Medo, Ruim, Sapo, Vacina* e *Vontade*.

### 2. Protocolo de Partição Inter-Sinalizador Cego (*Signer-Independent*)
- **Treinamento (303 sequências):** Sinalizadores `01`, `07` e `12`.
- **Validação / Seleção de Hiperparâmetros (100 sequências):** Sinalizador `10`.
- **Teste Cego Não Visto (185 sequências):** Sinalizadores `02` e `03`.
- **Controle Negativo:** Execução paralela sob partição aleatória convencional (75/25) para mensurar exatamente o viés de sobreajuste apontado por Arcanjo et al. (2024).

### 3. Vetorização e Hiperparâmetros de Treinamento
- **Entrada:** Matriz temporal de dimensão $32 \times 128$.
- **Vetor de cada frame:** 21 marcos $(x, y, z)$ da mão esquerda (63) + máscara de presença (1) + 21 marcos da mão direita (63) + máscara de presença (1).
- **Otimizador:** AdamW com taxa de aprendizado inicial $\eta = 5 \times 10^{-4}$ e decaimento de peso $\lambda = 10^{-4}$.
- **Função de Perda:** Entropia Cruzada com suavização de rótulos (*label smoothing*) de $0{,}05$ para mitigar excesso de confiança em classes dominantes.
- **Aumento de Dados no Treino:** Escala aleatória ($0{,}9$ a $1{,}1$), translação espacial com ruído $\sigma = 0{,}01$ e reamostragem elástica temporal ($\pm 15\%$), com zeramento obrigatório de coordenadas sob máscara inativa.

### 4. Métricas de Avaliação
- Acurácia Global em teste cego;
- Macro F1-Score (média aritmética dos F1-scores de todas as 20 classes);
- Matriz de Confusão detalhada para detecção de colapso de classes;
- Latência computacional decomposta: tempo de extração de landmarks (ms) + tempo de inferência GRU (ms) + vazão final em FPS.

---

## 2.4 Montagem do Protótipo em Laboratório e Avaliação Qualitativa

A bancada de testes experimentais foi montada no ambiente de laboratório com os seguintes procedimentos operacionais:

1. **Posicionamento Físico:** Câmera webcam posicionada frontalmente na borda superior do monitor ou em tripé de mesa, ajustada à altura dos olhos do sinalizador a uma distância de aproximadamente $70\,\text{cm}$.
2. **Iluminação da Cena:** Luminosidade de teto padrão de laboratório (fluorescente/LED branca difusa), sem spots direcionados ou refletores profissionais, validando a robustez a cenários cotidianos.
3. **Execução do Sistema:** Disparo da aplicação local pelo comando:
   ```bash
   uv run sinala camera --checkpoint artifacts/sinala_gru_6signers_hardened.pt --model-asset models/hand_landmarker.task
   ```
4. **Inspeção Qualitativa:**
   - O MediaPipe detecta com precisão os 21 marcos articulares por mão a partir do momento em que as mãos entram no campo de visão;
   - O baricentro normalizado elimina variações causadas por pequenas inclinações de postura do usuário;
   - A exibição gráfica HUD na janela OpenCV projeta os ossos anatômicos em azul cobalto e exibe a legenda da classe predita em tempo real com latência imperceptível ao olhar humano;
   - O buffer de histerese impede que sinais dinâmicos comecem exibindo palavras incorretas durante o movimento de subida das mãos.

---

## 2.5 Resumo dos Arquivos Gerados no Repositório

```text
materiais_academicos/
├── apresentacao/
│   ├── SINALA_Revisao_Literatura.pptx   (Slide oficial para apresentação oral)
│   └── colinha_apresentacao.md          (Roteiro de fala cronometrado: 3-4 min por referência)
├── artigo/
│   ├── main.tex                         (Artigo LaTeX completo com Introdução, Sistema e Experimentos)
│   ├── references.bib                   (10 referências acadêmicas com DOI e URL verificados)
│   └── figuras/
│       ├── captura_camera_bruta.jpg     (Foto 1: Quadro bruto da câmera óptica RGB)
│       ├── deteccao_landmarks.jpg       (Foto 2: Esqueleto articular detectado pelo MediaPipe)
│       └── prototipo_inferencia_hud.jpg (Foto 3: Tela final de inferência com legenda e FPS)
└── entregas_semanas_5_e_6.md            (Este guia mestre de consolidação das entregas)
```
