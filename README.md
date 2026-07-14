# Processamento de Linguagem Natural — Análise de Textos Clínicos

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19m-4BkWEzmgjzVwnCkhaPDuHkMz8zW5w?usp=sharing)

O trabalho implementa um **pipeline completo de PLN aplicado a texto clínico** (doenças e sintomas), percorrendo todas as etapas de um fluxo real de análise textual: desde o pré-processamento linguístico até a comunicação dos resultados. O objetivo é transformar descrições livres de sintomas em representações estruturadas que permitam **buscar, agrupar, classificar e extrair conhecimento** do domínio médico.

Todo o desenvolvimento está no notebook único:

```
pln_analise_textos_clinicos_doencas_sintomas.ipynb
```

> ⚠️ O notebook pesa ~222 MB porque as saídas (gráficos e imagens) estão incorporadas em base64. Recomenda-se abri-lo no Jupyter, VS Code ou Google Colab.

---

## 📊 Dataset

- **Fonte:** [`Shaheer14326/Disease_Symptoms_Dataset`](https://huggingface.co/datasets/Shaheer14326/Disease_Symptoms_Dataset) (Hugging Face), baixado em tempo de execução via `hf://`.
- **Coluna principal utilizada:** `Symptoms` (descrições textuais de sintomas). Colunas auxiliares: `Disease`, `Overview`.
- **Volume:** limitado a **100.000 registros**.
- **Idioma dos dados:** inglês.

Nenhum dado é armazenado localmente: o corpus e os modelos externos são baixados quando o notebook é executado.

---

## 🛠️ Tecnologias

| Categoria | Bibliotecas / Modelos |
| :--- | :--- |
| Manipulação de dados | `pandas`, `numpy` |
| Pré-processamento linguístico | `spaCy` (`en_core_web_sm`), `NLTK` (`PorterStemmer`, stopwords) |
| Representação vetorial | `gensim` — Word2Vec, **BioWordVec** (`BioWordVec_PubMed_MIMICIII_d200`), **GloVe** (`glove-wiki-gigaword-100`) |
| Modelagem e classificação | `scikit-learn` — TF-IDF, NMF, K-Means, PCA, t-SNE, LinearSVC, LogisticRegression |
| Grafos / NER | `networkx`, `spaCy` (NER customizado + displaCy) |
| Visualização | `matplotlib`, `seaborn`, `wordcloud` |

As dependências instaláveis via `pip` estão em [`requirements.txt`](./requirements.txt).

---

## 🚀 Como executar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Baixe o modelo de português/inglês do spaCy (também é baixado automaticamente pelo notebook se ausente):
   ```bash
   python -m spacy download en_core_web_sm
   ```
3. Abra e execute o notebook célula a célula (Jupyter, VS Code ou Colab).

> ▶️ **Executar direto no Google Colab (sem instalar nada):** [abrir notebook no Colab](https://colab.research.google.com/drive/19m-4BkWEzmgjzVwnCkhaPDuHkMz8zW5w?usp=sharing).

**Downloads automáticos durante a execução:** o próprio notebook cuida de baixar o dataset (Hugging Face), as *stopwords* do NLTK, os embeddings **GloVe** (via `gensim.downloader`) e o **BioWordVec** (`BioWordVec_PubMed_MIMICIII_d200.vec.bin`, a partir do NCBI). O BioWordVec é um arquivo grande — garanta conexão e espaço em disco suficientes.

---

## 🧩 Etapas do pipeline

O notebook está organizado em **5 Partes**, cada uma representando uma etapa do fluxo de PLN. Abaixo, o significado de cada etapa e o que é feito nela.

### Parte 1 — Pré-processamento textual (NLTK e spaCy)
Prepara o texto bruto para análise, garantindo qualidade semântica no domínio clínico.
- **1.1** Carregamento e inspeção inicial do corpus.
- **1.2** Tokenização e limpeza (regex + `spaCy.pipe`).
- **1.3** Normalização: lematização + remoção de *stopwords*, incluindo **stopwords médicas customizadas** (`patient`, `disease`, `symptom`, `treatment`…) para não poluir a análise.
- **1.4 / 1.5** Comparação **Lematização (spaCy) vs. Stemming (NLTK)** em qualidade lexical e em **tempo de execução**.
- **1.6** Distribuição de **POS tagging** (classes gramaticais).
- **1.7** Impacto da lematização no **tamanho do vocabulário** (redução de **9.840 → 7.785 lemas**).
- **1.8** **Nuvens de palavras** (WordCloud) comparando as duas técnicas.
- **1.9 / 1.10** Distribuição de termos frequentes e histograma de comprimento dos documentos.
- **1.11** Justificativa das escolhas → adota-se **lematização com spaCy + stopwords customizadas** por preservar o significado morfológico e a interpretabilidade.

### Parte 2 — Representação vetorial e busca textual
Converte o texto em vetores numéricos (*embeddings*) e mede quão bem cada abordagem agrupa os documentos.
- **2.1** Experimento 1: **Word2Vec** (treinado localmente) + **K-Means** + **t-SNE**.
- **2.2** Experimento 2: **BioWordVec** (embeddings pré-treinados em literatura biomédica — PubMed/MIMIC-III) + t-SNE.
- **2.3** Experimento 3: **GloVe** (embeddings de domínio geral — Wikipédia) + t-SNE.
- **2.4** Relatório técnico comparando os três modelos via **Silhouette Score** (Word2Vec ≈ 0,269; BioWordVec ≈ 0,195; GloVe ≈ 0,141).
- **2.5** **Busca semântica** por similaridade do cosseno, com exemplos de consultas e conclusão técnica.

### Parte 3 — Modelagem, classificação e análise de tópicos
Descobre temas latentes no corpus e treina classificadores supervisionados.
- **3.1** **Modelagem de tópicos com NMF** sobre a matriz TF-IDF (`max_features=1500`, **5 tópicos**), extraindo palavras-chave por tópico.
- **3.2** Classificação supervisionada com **SVM** (`LinearSVC`), usando o tópico dominante do NMF como rótulo → **acurácia ≈ 98,5%**.
- **3.3** Classificação supervisionada com **Regressão Logística** (mesmos dados), para comparação.
- **3.4** Relatório técnico: interpretação dos tópicos e comparação dos classificadores (SVM sai como melhor).

### Parte 4 — NER, extração de informação e grafo de conhecimento
Extrai entidades e modela as relações entre doenças e sintomas.
- **4.1** Construção de um **grafo de conhecimento com NetworkX** (bipartido Doença ↔ Sintoma).
- **4.2** Cálculo de **centralidade** (grau e intermediação) para achar os nós mais importantes.
- **4.3** Anotação de exemplos e criação de um **NER customizado com spaCy** (entidade `SYMPTOM`).
- **4.4** **Visualização de entidades com displaCy**.
- **4.5** Resposta a uma **pergunta analítica** usando o grafo (ex.: vizinhos do nó `pain`).

### Parte 5 — Visualização, comunicação e reprodutibilidade
Consolida e comunica os resultados de forma acessível.
- **5.1** **Mapa de visualizações** — inventário de todos os gráficos e onde aparecem.
- **5.2** **Síntese interpretativa** em linguagem acessível, traduzindo os achados técnicos em valor prático.

---

## ✅ Principais conclusões

- **Lematização (spaCy) > Stemming (NLTK)** para texto clínico: preserva o significado e gera palavras reais, ao custo de mais tempo de processamento.
- **Word2Vec local** obteve o melhor Silhouette Score neste corpus, superando embeddings pré-treinados gerais para a tarefa de agrupamento.
- O **NMF** separou o corpus em 5 grupos coerentes de sintomas; sobre esses rótulos, o **SVM** alcançou ~98,5% de acurácia, superando a Regressão Logística.
- O **grafo de conhecimento** revela os sintomas mais centrais (ex.: `pain`) e as relações mais fortes entre doenças e sintomas.

---

## 👤 Autor

**Efrain Colmenares** — Disciplina de Processamento de Linguagem Natural, **INFNET**.
