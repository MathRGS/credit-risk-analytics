# 📊 Credit Risk Analytics & Intelligence

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)

Este repositório reúne ferramentas e modelos desenvolvidos em Python voltados para a **Análise de Risco de Crédito** e **Gestão de Inadimplência**. 

O objetivo é aplicar rigor estatístico e teoria econômica para prever o comportamento de carteiras de crédito, unindo a precisão do Machine Learning com a governança necessária em Tesouraria e Auditoria.

---

## 🚀 Performance do Modelo (Benchmark)

Para validar a eficácia do algoritmo em um cenário controlado, utilizamos uma base de dados sintética ("Mock Data") calibrada com premissas econômicas reais. O modelo foi capaz de identificar corretamente os padrões de risco embutidos, atingindo as seguintes métricas de performance:

| Métrica | Resultado | Interpretação de Mercado |
| :--- | :--- | :--- |
| **Gini Coefficient** | **60.64%** | Alta capacidade de discriminação (Padrão Top-Tier) |
| **AUC ROC** | **0.803** | Excelente separabilidade entre classes |
| **Tempo de Treino** | < 1s | Altamente otimizado para volumetria |

> *Nota: O modelo demonstrou consistência lógica ao atribuir PDs de ~10% para faixas de Score >800 e PDs >50% para Scores <600.*

---

## 🎯 Objetivos do Projeto

- **Modelagem Preditiva:** Cálculo de *Probabilidade de Default (PD)* utilizando Regressão Logística.
- **Engenharia Financeira:** Estimativa de *Expected Loss (EL)* vetorial baseada nos pilares de Basileia (PD, LGD e EAD).
- **Automação:** Geração automática de relatórios executivos de risco e exposição de carteira.

---

## 📂 Estrutura do Projeto

O projeto foi arquitetado separando a geração de dados da lógica de negócio (Engine), simulando um ambiente de produção real.

| Arquivo | Função |
| :--- | :--- |
| `risk_engine.py` | **Motor de Risco:** Contém a lógica de Machine Learning, cálculo vetorial de EL e geração de relatórios. |
| `generate_mock_data.py` | **Fábrica de Dados:** Gera bases sintéticas com distribuições estatísticas realistas (Lognormal para renda, etc). |
| `requirements.txt` | Lista de dependências para reprodução do ambiente. |

---

## 🖥️ Exemplo de Output (Relatório Executivo)

Ao executar o pipeline completo, o sistema entrega uma visão consolidada para tomada de decisão:

```text
============================================================
RELATÓRIO EXECUTIVO DE RISCO DE CRÉDITO
============================================================
Volume Analisado (N):     1000 contratos
Exposição Total (EAD):    R$ 42,835,306.34
Perda Esperada (EL):      R$ 9,079,393.49
Índice de Cobertura (EL%):21.20%
Prob. Default Média (PD): 46.52%
------------------------------------------------------------
```
# 🛠 Tech Stack & Metodologias
Linguagem: Python 3.x (Foco em POO e Type Hinting)

Bibliotecas: - Pandas & NumPy: Manipulação vetorial de dados financeiros.

Scikit-Learn: Treinamento de modelos e métricas (ROC/AUC).

Joblib: Persistência de objetos (MLOps).

Conceitos Econômicos: Modelagem de Scoring, Curva de Juros, Teoria de Crédito e Auditoria.

# ⚙️ Como executar
Bash

1. Instale as dependências
pip install -r requirements.txt

2. Gere a massa de dados simulada (Seed Estatística)
python generate_mock_data.py

3. Execute o Motor de Risco
python risk_engine.py

---
👨‍💻 Desenvolvido por Matheus Rocha Economista (UnB) | Especialista em Tesouraria e Auditoria Fiscal | Fullstack Developer
