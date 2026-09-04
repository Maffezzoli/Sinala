# Colinha da apresentação SINALA

Apresentação: 9 slides. Referências: 6. Tempo recomendado: 3 minutos por artigo.

## Slide 1 — abertura

Mensagem central: o SINALA começa pelo reconhecimento de sinais isolados de Libras, medido em pessoas não vistas, e evolui posteriormente para reconhecimento contínuo e tradução semântica.

## Slide 2 — problema e níveis

- Libras e português possuem estruturas linguísticas diferentes.
- O problema técnico envolve movimento, oclusão, variação entre sinalizadores e custo de inferência.
- Nível 1: reconhecer sinais isolados.
- Nível 2: segmentar sinais contínuos e incorporar outras modalidades.
- Nível 3: alinhar Libras ao português e produzir legenda ou voz.

Frase de transição: “As seis referências foram organizadas conforme esses três níveis.”

## Slide 3 — Furtado, de Oliveira e Shirmohammadi (2023)

**Artigo:** *Interactive and Markerless Visual Recognition of Brazilian Sign Language Alphabet* — IEEE I2MTC.

**Resumo:** propõe reconhecimento do alfabeto de Libras usando câmera RGB, sem luvas ou sensores corporais.

**Resultado quantitativo:** 96,05% de acurácia reportada; nenhum dispositivo vestível.

**Resultado qualitativo:** demonstra interação natural e baixo custo com câmera convencional.

**Limitação:** o alfabeto é um problema predominantemente estático e mais restrito que palavras ou frases em movimento.

**Por que foi escolhido:** fundamenta o requisito de captura totalmente sem marcadores do SINALA.

**Fala sugerida:** “Este trabalho estabelece a premissa de acessibilidade física do projeto. A câmera comum é suficiente para reconhecer unidades básicas, mas o artigo não resolve a dependência temporal dos sinais dinâmicos.”

## Slide 4 — Feliciano, Briano, Prudêncio e Alves (2023)

**Artigo:** *Recognition of Static or Dynamic LIBRAS Words in Complex Background Environments* — IEEE CISTI.

**Resumo:** combina MediaPipe, OpenCV e classificação neural para tratar sinais estáticos e dinâmicos em fundos complexos.

**Resultado quantitativo disponível:** acurácia superior a 90% na etapa de treinamento; duas categorias de sinais avaliadas, estáticos e dinâmicos.

**Resultado qualitativo:** desloca o problema do alfabeto controlado para ambientes visualmente mais difíceis.

**Limitação:** o resumo público não oferece detalhamento suficiente das métricas de teste; o texto integral deve ser consultado antes de defender números adicionais.

**Por que foi escolhido:** aproxima-se diretamente do pipeline do SINALA por utilizar landmarks como representação intermediária.

**Fala sugerida:** “A contribuição mais útil não é apenas a taxa de treinamento, mas a decisão de separar detecção articular e classificação. Essa mesma separação reduz o volume de dados processado pelo SINALA.”

## Slide 5 — Arcanjo et al. (2024)

**Artigo:** *Automatic Time-Aware Recognition of Brazilian Sign Language Based on Dynamic Time Warping* — SBC WebMedia, PUC Minas.

**Resumo:** extrai landmarks de mãos e corpo, calcula ângulos 3D e usa FastDTW no MINDS-Libras.

**Resultados quantitativos:**

- split aleatório: 98% ± 1%;
- avaliação por sinalizador com LOOCV: 86% ± 8%;
- F1 no LOOCV: 0,87.

**Resultado qualitativo:** comprova que a identidade do sinalizador pode inflar a avaliação quando aparece nos dois conjuntos.

**Limitação:** sinais com configurações semelhantes ainda se confundem quando faltam pistas relativas ao rosto.

**Por que foi escolhido:** é o benchmark mais próximo do SINALA: mesma base, mesmas 20 classes e preocupação com generalização entre pessoas.

**Fala sugerida:** “Este é o artigo central da revisão. O principal resultado não é 98%, mas a queda para 86% quando o protocolo passa a medir pessoas inéditas. Esse é o protocolo cientificamente relevante para nosso projeto.”

## Slide 6 — de Castro, Guerra e Guimarães (2023)

**Artigo:** *Automatic Translation of Sign Language with Multi-Stream 3D CNN and Generation of Artificial Depth Maps* — *Expert Systems with Applications*, UFMG.

**Resumo:** combina fluxos de mãos, face e movimento em uma 3D CNN, acrescentando mapas artificiais de profundidade.

**Resultados quantitativos:** aproximadamente 91% de acurácia média e F1 de 0,90.

**Resultado qualitativo:** a fusão de informações espaciais complementares ajuda a distinguir sinais visualmente ambíguos.

**Limitação:** a arquitetura multi-stream 3D é mais cara que uma sequência compacta de landmarks.

**Por que foi escolhido:** mostra o benefício da informação tridimensional e multimodal, mas também evidencia o compromisso entre precisão e execução local.

**Fala sugerida:** “O SINALA aproveita a ideia de preservar a geometria 3D, mas inicia com uma representação vetorial leve para sustentar inferência em câmera convencional.”

## Slide 7 — Rodrigues, Barbosa, Fiera e Oliveira (2025)

**Artigo:** *Dynamic Sign Recognition in Brazilian Sign Language Through Convolutional Neural Networks* — SBC ENIAC, UNESC.

**Resumo:** compara AlexNet, DenseNet121, InceptionV3, ResNet18 e VGG16 em letras dinâmicas de Libras.

**Resultados quantitativos:** ResNet18 obteve a melhor acurácia, 74,29%, entre cinco CNNs avaliadas.

**Resultado qualitativo:** modelos maiores não garantiram melhor resultado com um conjunto limitado.

**Limitação:** o tratamento do tempo é indireto; isso dificulta representar velocidade e direção do gesto.

**Por que foi escolhido:** justifica avaliar explicitamente modelos sequenciais, como GRU, em vez de aumentar apenas a profundidade da CNN.

**Fala sugerida:** “A conclusão não é que a ResNet18 resolve o problema, mas que existe um teto quando a arquitetura não representa adequadamente a evolução temporal do sinal.”

## Slide 8 — Pereira, Santana, Martins e Corrêa (2026)

**Artigo:** *Semantic Vector Space Mapping Between Brazilian Portuguese and Libras Glosses Using Siamese Models* — SBC WICS, UFPel.

**Resumo:** usa BERTimbau siamês e aprendizado contrastivo para aproximar sentenças em português e glosas de Libras.

**Resultados quantitativos:**

- corpus original: 2.400 pares;
- subconjunto curado: 809 pares;
- Recall@1: 98,77%;
- Recall@5 e Recall@10: 100%;
- MRR: 99,38%.

**Resultado qualitativo:** os embeddings de português e glosas ocupam regiões semânticas sobrepostas.

**Limitação:** negativos relativamente fáceis provavelmente elevam as métricas; o trabalho também começa depois da etapa visual.

**Por que foi escolhido:** representa a etapa de tradução semântica que poderá receber a saída de um reconhecedor visual confiável.

**Fala sugerida:** “Este artigo fecha o roadmap. Reconhecer o gesto não equivale a traduzir. A tradução exige uma etapa semântica entre a sequência em Libras e a sentença natural em português.”

## Slide 9 — síntese

- Captura: webcam sem marcadores.
- Representação: landmarks ou vídeo espaço-temporal.
- Avaliação: pessoas do teste ausentes no treino.
- Arquitetura: considerar custo e quantidade de dados.
- Tradução: manter separadas as etapas visual e semântica.

Encerramento sugerido: “A literatura consolidou o escopo: primeiro, reconhecer sinais isolados com generalização real; depois, incorporar continuidade e semântica. Essa sequência evita prometer tradução antes de resolver de forma confiável a percepção visual.”

## Perguntas prováveis

**Por que não usar split aleatório para atingir 98%?**

Porque Arcanjo et al. mostram que o valor cai de 98% para 86% quando o teste é realizado em sinalizadores não vistos. O split aleatório pode medir identidade do indivíduo em vez de generalização.

**Por que usar GRU?**

Porque sinais dinâmicos dependem da ordem e velocidade dos quadros. Rodrigues et al. mostram desempenho máximo de 74,29% entre cinco CNNs avaliadas, reforçando a necessidade de modelagem temporal explícita.

**O sistema atual já traduz Libras?**

Não. Ele reconhece classes isoladas. Pereira et al. fundamentam uma futura camada semântica para alinhar glosas e português.

**Qual trabalho está mais próximo do SINALA?**

Arcanjo et al. (2024), porque utiliza o MINDS-Libras, MediaPipe, informação temporal e avaliação por sinalizador.
