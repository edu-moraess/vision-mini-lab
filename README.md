# VISION MINI LAB

**Computer Vision Scene Analysis Laboratory**

Análise de cenas estáticas e dinâmicas com detecção de objetos, métricas geométricas, espaciais e temporais, utilizando **YOLO**, **OpenCV** e **Streamlit**.

---

## 🎯 Objetivo

O **Vision Mini Lab** é um laboratório de visão computacional projetado para:
- **Detecção de objetos** (YOLOv8).
- **Análise geométrica** (área, proporção, tamanho relativo).
- **Análise espacial** (distribuição em grade 3x3, densidade, centróide).
- **Rastreamento de objetos** (ID, deslocamento, direção, trajetória).
- **Análise temporal** (evolução de objetos por frame, métricas de confiança ao longo do tempo).
- **Métricas de qualidade** (indicadores baseados em confiança e consistência).

---

## 🧠 Pipeline

Input (Imagem/Vídeo/Câmera)
↓
Detecção YOLO
↓
Análise Geométrica (área, aspecto, tamanho relativo)
↓
Análise Espacial (regiões, densidade, centróide)
↓
Rastreamento (opcional: ID, deslocamento, direção)
↓
Análise Temporal (vídeo: contagem de objetos, confiança por frame)
↓
Geração de Relatório e Visualização


---

## 📊 Métricas Implementadas

### **Geométricas**
- **Bounding Box**: Coordenadas `(x1, y1, x2, y2)`.
- **Dimensões**: Largura, altura, área (pixels e relativa à imagem).
- **Proporção**: *Aspect ratio* (largura/altura).
- **Centro**: Coordenadas absolutas e normalizadas.
- **Distância Relativa**: Baseada no tamanho aparente (sem calibração de câmera).

### **Espaciais**
- **Regiões**: Classificação em grade 3x3 (ex: `SUPERIOR_ESQUERDA`, `CENTRO`).
- **Densidade**: Objetos por milhão de pixels.
- **Centróide Global**: Ponto médio de todos os objetos detectados.
- **Cobertura de União**: Área total coberta por *bounding boxes* (evita dupla contagem em sobreposições).

### **Confiança**
- **Estatísticas**: Média, mediana, desvio padrão.
- **Faixas**: `MUITO_ALTA` (≥90%), `ALTA` (75-89%), `MODERADA` (50-74%), `BAIXA` (<50%).
- **Distribuição**: Gráfico de barras por faixa de confiança.

### **Temporais** (apenas para vídeo)
- **Contagem de Objetos**: Por frame.
- **Confiança Média**: Por frame.
- **Rastreamentos Ativos**: Número de objetos rastreados por frame.
- **Evolução**: Gráfico de linha da contagem de objetos ao longo do tempo.

### **Sobreposição (IoU)**
- **Cálculo Par a Par**: *Intersection over Union* entre objetos.
- **Classificação**: `ALTO` (IoU > 0.6), `MÉDIO` (0.4-0.6), `BAIXO` (<0.4).
- **Lista de Pares**: Objetos com IoU > 0.3.

### **Indicador de Qualidade**
- **Heurística**: Baseado em:
  - **`HIGH`**: Confiança média ≥ 80% e ≥70% das detecções com confiança ≥75%.
  - **`MEDIUM`**: Confiança média ≥ 60% e ≥40% das detecções com confiança ≥75%.
  - **`LOW`**: Caso contrário.

---

## 📦 Instalação

### Pré-requisitos
- Python 3.8+
- Git

### Passos
1. Clone o repositório:
   ```bash
   git clone https://github.com/edu-moraess/vision-mini-lab.git
   cd vision-mini-lab

Crie um ambiente virtual:
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
 2. Instale as dependências:
pip install -r requirements.txt
 3. (Opcional) Configure a GROQ API Key para descrição de frames:
- Crie um arquivo .env na raiz do projeto:
GROQ_API_KEY=sua_chave_aqui
YOLO_MODEL=yolov8n.pt  # ou outro modelo YOLO
🚀 Execução
Inicie o aplicativo Streamlit:
streamlit run app.py
O aplicativo estará disponível em http://localhost:8501.
📂 Estrutura do Projeto

vision-mini-lab/
├── app.py                  # Interface principal (Streamlit)
├── requirements.txt        # Dependências
├── README.md               # Documentação
├── .gitignore
├── src/
│   ├── analyzer.py         # Análise geométrica e classificação
│   ├── detector.py         # Detector YOLO e rastreamento
│   ├── export.py           # Exportação de dados (CSV/JSON)
│   ├── metrics.py          # Métricas de performance
│   ├── processor.py        # Processamento de frames (vídeo/câmera)
│   ├── report.py           # Geração de relatórios
│   ├── spatial.py          # Análise espacial (regiões, densidade, união)
│   ├── temporal.py         # Análise temporal (vídeo)
│   ├── tracking.py         # Métricas de rastreamento
│   ├── video.py            # Utilitários para vídeo
│   └── visualization.py    # Funções de desenho (grade, centros, trajetórias)
├── tests/
│   ├── test_analyzer.py    # Testes para analyzer.py
│   ├── test_core.py        # Testes gerais
│   ├── test_spatial.py     # Testes para spatial.py
│   └── test_tracking.py    # Testes para tracking.py
└── data/
    └── captures/           # Frames capturados

🎛️ Funcionalidades da Interface
Abas Disponíveis
 1. 📷 Imagem:
- Upload de imagens (JPG, PNG, WEBP).
- Detecção de objetos com YOLO.
- Visualização de bounding boxes, grade espacial e centros.
- Relatório detalhado com métricas geométricas, espaciais e de confiança.
- Exportação de dados (CSV/JSON).
 2. 🎬 Vídeo:
- Upload de vídeos (MP4, AVI, MOV, MKV).
- Processamento frame a frame com opção de sampling.
- Visualização de trajetórias (se rastreamento estiver ativo).
- Métricas temporais (evolução de objetos, confiança média).
- Gráfico de linha da contagem de objetos ao longo do tempo.
 3. 📹 Câmera:
- Captura em tempo real da webcam.
- Detecção e rastreamento de objetos.
- Métricas de performance (FPS, contagem de objetos).
 4. 📊 Métricas:
- Resumo da sessão (frames analisados, objetos totais, FPS).
- Métricas temporais (se aplicável).
Configurações (Sidebar)
- Confidence Threshold: Filtro de confiança mínima para detecções.
- Sample Every: Processa 1 frame a cada N frames (para vídeo/câmera).
- Tracking: Ativa/desativa rastreamento de objetos.
- Box Details: Exibe detalhes nas bounding boxes (centro, tamanho, área).
- Mostrar Trajetórias: Desenha trajetórias de objetos rastreados.
- Mostrar Grade Espacial: Exibe grade 3x3 na imagem.
- Mostrar Centros: Marca o centro de cada bounding box.
- Descrever Frame (GROQ): Usa IA para descrever a cena (requer GROQ_API_KEY).
🔬 Limitações
 1. Espaço de Imagem:
- Todas as métricas são em pixels (sem calibração de câmera para mundo real).
- Distância relativa é baseada no tamanho aparente, não em distância física.
- Velocidade é medida em px/frame, não em m/s.
 2. Rastreamento:
- Depende do YOLO e pode falhar em oclusões ou objetos muito rápidos.
- Histórico de trajetória limitado a 10 pontos por objeto.
 3. Desempenho:
- Processamento em tempo real pode ser lento em hardware modestos.
- Recomenda-se usar sample_every > 1 para vídeos em alta resolução.
📄 Licença
Este projeto está licenciado sob a MIT License. Veja o arquivo LICENSE para mais detalhes.
🤝 Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs ou sugerir melhorias via Issues.
- Enviar Pull Requests com novas funcionalidades ou correções.


---
### **Destaques das Melhorias Recentes**
1. **Análise Temporal**:
   - Novo módulo `temporal.py` para rastrear métricas ao longo do tempo (vídeo).
   - Gráfico de evolução da contagem de objetos.

2. **Métricas Espaciais**:
   - **Union Coverage**: Cálculo da área total coberta por *bounding boxes* (evita dupla contagem).
   - **Classificação Relativa**: Tamanho dos objetos em relação à imagem (`MUITO_PEQUENO`, `PEQUENO`, etc.).

3. **Rastreamento**:
   - Métricas de deslocamento, direção e trajetória.
   - Visualização de trajetórias na aba **Vídeo**.

4. **Relatório**:
   - Inclusão de **área relativa** e **union coverage** no relatório.
   - Exportação em **CSV/JSON** com todos os campos.

5. **Testes**:
   - Novos testes para `spatial.py` e `tracking.py` (24 testes passando).
