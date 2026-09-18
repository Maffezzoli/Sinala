# SINALA: Sistema de Reconhecimento de Libras em Tempo Real

O **SINALA** é uma solução de Visão Computacional e Aprendizado Profundo projetada para o reconhecimento de Língua Brasileira de Sinais (Libras) em tempo real via webcam convencional, operando **100% em CPU** com latência inferior a 2 ms por inferência.

---

## 1. Arquitetura Geral do Sistema

O pipeline desacopla a captura de pixels da classificação neural por meio de representação geométrica esquelética:

```
[Webcam RGB (30 FPS)]
         │
         ▼
[MediaPipe HandLandmarker] ──► 21 Marcos Articulares 3D por mão (x, y, z)
         │
         ▼
[Normalização Anatômica]  ──► Invariância a pulso, escala e lateralidade (mão esquerda/direita)
         │
         ▼
 ┌───────┴──────────────────────────────┐
 ▼                                      ▼
[Dactilologia: Alfabeto A–Z]         [Sinais Temporais: Palavras e Saudações]
 • Rede: MLP Neural (PyTorch)         • Rede: GRU Recorrente (PyTorch)
 • 225 Features Geométricas           • 128 Features x 32 Frames
 • Acurácia interna: 100,00%          • Buffer de Histerese Temporal
         │                                      │
         └───────────────┬──────────────────────┘
                         ▼
             [HUD / Legenda Contínua]
```

---

## 2. Bases de Dados e Vocabulário

O SINALA integra três fontes de dados complementares e auditadas:

| Domínio | Fonte | Classes | Volume / Amostras | Representação |
|---|---|---|---|---|
| **Dactilologia (Alfabeto)** | *LibraSign* (Kaggle) | **26 classes** (A–Z) | 26.000 amostras (1.000/letra) | CSV vetorial 225D (210 distâncias + 15 ângulos invariantes) |
| **Sinais Léxicos (Palavras)** | *MINDS-Libras* (UFMG) | **20 classes** (*Acontecer, Aluno, Banco...*) | 588 sequências de 6 sinalizadores | Vídeos 30 FPS $\rightarrow$ 32 frames x 128D |
| **Saudações e Cortesia** | *V-Librasil + Acessibilidade Brasil + Spread the Sign* | **9 classes** (*Oi, Nome, Obrigado, Por favor, Cumprimento, Desculpa, Ajudar, Sim, Não*) | 32 sequências de 3 fontes | CSV de landmarks com provenance (`source_id`, `signer_id`, `variant`) |

**Limitações atuais:** o alfabeto tem split aleatório por linha, sem IDs de sinalizador/sessão; a acurácia interna não representa generalização para uma webcam nova. O dataset conversacional agora tem 32 sequências: 27 do V-Librasil, 4 da Acessibilidade Brasil e 1 do Spread the Sign. O holdout por articulador do V-Librasil ainda é baixo, portanto o checkpoint final usa todas as fontes para operação, mas sua generalização não está comprovada.

---

## 3. Normalização e Invariância a Ângulos

Para evitar que o modelo se confunda com diferentes posturas, distâncias e inclinações da mão na webcam:

1. **Translação ao Pulso:** O marco $0$ é usado como origem antes dos cálculos geométricos.
2. **Escala Anatômica:** As distâncias são normalizadas pelo comprimento pulso–base do dedo médio (marco 9).
3. **Features Invariantes:** O classificador usa 210 distâncias par-a-par e 15 ângulos articulares 3D, totalizando 225 features invariantes a translação, escala, rotação e reflexão.

---

## 4. Guia Rápido de Uso (CLI)

O projeto é gerenciado via `uv` e expõe a interface de linha de comando `sinala`:

### Instalação
```bash
# Sincronizar ambiente virtual e dependências
uv sync
```

### 1. Dactilologia (Alfabeto A–Z)
```bash
# 1. Baixar dataset público de 26.000 amostras do alfabeto
uv run sinala download-alphabet --output data/raw/alphabet

# 2. Treinar classificador MLP de dactilologia
uv run sinala train-alphabet --epochs 10 --output-checkpoint artifacts/sinala_alphabet.pt

# 3. Transcrever em tempo real na webcam com o WordBuilder
uv run sinala camera-alphabet --checkpoint artifacts/sinala_alphabet.pt
```

### 2. Sinais Conversacionais e Saudações (múltiplas fontes)
```bash
# Baixar V-Librasil e fontes públicas complementares; extrair 32 sequências
uv run sinala download-greetings --output data/landmarks-greetings

# Treinar com aumento temporal/espacial e holdout do Articulador3
uv run sinala train-greetings \
  --manifest data/landmarks-greetings/manifest.jsonl \
  --checkpoint artifacts/sinala_greetings.pt \
  --epochs 30 \
  --augment-copies 40
```

### 3. Sinais Léxicos (MINDS-Libras)
```bash
# Baixar sinalizadores do MINDS-Libras
uv run sinala download-data --signers 01,02,03,07,10,12 --output data/raw/zips

# Extrair vídeos
uv run sinala extract-data --input data/raw/zips --output data/raw/videos --signers 01,02,03,07,10,12

# Extrair landmarks temporais
uv run sinala extract-landmarks \
  --data-root data/raw/videos \
  --output data/landmarks-20 \
  --classes Acontecer,Aluno,Amarelo,América,Aproveitar,Bala,Banco,Banheiro,Barulho,Cinco,Conhecer,Espelho,Esquina,Filho,Maçã,Medo,Ruim,Sapo,Vacina,Vontade

# Treinar classificador temporal GRU
uv run sinala train \
  --manifest data/landmarks-20/manifest.jsonl \
  --checkpoint artifacts/sinala_gru_6signers.pt \
  --classes Acontecer,Aluno,Amarelo,América,Aproveitar,Bala,Banco,Banheiro,Barulho,Cinco,Conhecer,Espelho,Esquina,Filho,Maçã,Medo,Ruim,Sapo,Vacina,Vontade \
  --train-signers 01,02,03,07 \
  --validation-signers 10 \
  --test-signers 12

# Executar câmera para palavras do vocabulário
uv run sinala camera --checkpoint artifacts/sinala_gru_6signers.pt
```

### 4. Modo Unificado (Palavras / Saudações + Dactilologia na Mesma Câmera)
```bash
# Transcreve tanto sinais inteiros (Oi, Obrigado...) quanto letras soletradas no mesmo feed
uv run sinala camera-unified
```
* **Arbitragem Cinemática:**
  - **Gesto Dinâmico:** Se a mão realiza o sinal dinâmico de *"Oi"*, a GRU temporal captura a trajetória e insere a palavra completa (`"OI "`) no texto.
  - **Mão Estacionária:** Se a mão fica parada na postura de uma letra, o classificador geométrico de dactilologia assume e adiciona letra por letra com debounce.
  - **Controles:** `[ESPAÇO]` separa palavras, `[BACKSPACE]` apaga, `C` limpa tudo e `Q` fecha a janela.

---

## 5. Testes Automatizados

A integridade matemática do pipeline, extratores, aumentador de dados, modelos e buffers é verificada pela suíte de testes:

```bash
uv run pytest
```

Atualmente, **45 testes unitários e de integração** cobrem os módulos críticos do projeto.

---

## 6. Estrutura do Repositório

```text
.
├── artifacts/                  # Checkpoints dos modelos treinados (.pt)
├── data/                       # Datasets locais (raw, landmarks e CSVs)
├── models/                     # Modelos de inferência (hand_landmarker.task)
├── materiais_academicos/       # Apresentação oficial (.pptx), roteiro (.md) e artigo (.tex)
│   ├── apresentacao/
│   └── artigo/
├── src/sinala/
│   ├── config/                 # Configurações do pipeline (Pydantic Settings)
│   ├── data/                   # Downloaders, catalog, dataset e extratores
│   ├── model/                  # Classificadores neurais (GRU, MLP) e treino
│   └── runtime/                # Aplicações de câmera, WordBuilder e buffers
└── tests/                      # 43 testes automatizados (PyTest)
```
