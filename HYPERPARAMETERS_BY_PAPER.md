# Mapeamento de Hiperparâmetros por Artigo

## Artigo 1: ResearchSquare (Redes Neurais para DQN/DDQN)
**Link:** https://assets-eu.researchsquare.com/files/rs-5939959/v2_covered_b0beacb0-534e-4c72-8a6c-7954fdc6a3ed.pdf

- **layers:** [64, 64]
- **activation:** ReLU
- **BATCH_SIZE:** 64
- **GAMMA:** 0.99
- **EPS_START:** X
- **EPS_END:** X
- **EPS_DECAY:** X
- **TARGET_UPDATE:** X
- **NUM_EPISODES:** X
- **LR:** 0.0003
- **dropout:** X

---

## Artigo 2: arxiv.org/pdf/2011.11850
**Link:** https://arxiv.org/pdf/2011.11850

- **layers:** [128, 128]
- **activation:** ReLU
- **BATCH_SIZE:** 64
- **GAMMA:** X
- **EPS_START:** X
- **EPS_END:** X
- **EPS_DECAY:** X
- **TARGET_UPDATE:** X
- **NUM_EPISODES:** X
- **LR:** 0.001
- **dropout:** X


---

## Artigo 4: dl.acm.org (Short Term Memory DQN)
**Link:** https://dl.acm.org/doi/epdf/10.1145/3633637.3633641

- **layers:** [128, 128]
- **activation:** ReLU
- **BATCH_SIZE:** X
- **GAMMA:** 0.99
- **EPS_START:** 0.2
- **EPS_END:** X
- **EPS_DECAY:** X
- **TARGET_UPDATE:** X
- **NUM_EPISODES:** X
- **LR:** 0.001
- **dropout:** X

---

## Resumo Comparativo

| Parâmetro | Paper 1 | Paper 2 | Paper 3 | Paper 4 |
|-----------|---------|---------|---------|---------|
| layers | [64, 64] | [128, 128] | X | [128, 128] |
| activation | ReLU | ReLU | X | ReLU |
| BATCH_SIZE | 64 | 64 | 32 | X |
| GAMMA | 0.99 | X | 0.99 | 0.99 |
| EPS_START | X | X | X | 0.2 |
| EPS_END | X | X | X | X |
| EPS_DECAY | X | X | X | X |
| TARGET_UPDATE | X | X | 4 | X |
| NUM_EPISODES | X | X | X | X |
| LR | 0.0003 | 0.001 | X | 0.001 |
| dropout | X | X | X | X |

---

## Observações

- **Layers mais comuns:** 64 ou 128 unidades nas camadas ocultas
- **Activation:** ReLU é consenso em todos os papers que especificam
- **BATCH_SIZE:** Varia entre 32-64
- **GAMMA:** Consistentemente 0.99 quando especificado
- **LR:** Varia entre 0.0003 e 0.001
- **Parâmetros não especificados:** EPS_END, EPS_DECAY, NUM_EPISODES, dropout não aparecem nos papers (marcados como X)
