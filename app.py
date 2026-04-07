import streamlit as st
import time
from model_utils import SentimentClassifier
import os

# Configuração da página - Dark Tech Theme
st.set_page_config(
    page_title="SENTINEL-X | Análise de Sentimento",
    page_icon="🦾",
    layout="centered"
)

# Estilo CSS Cyberpunk / High-Tech
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'JetBrains+Mono', monospace;
    }
    .stApp {
        background: radial-gradient(circle at top, #1e293b 0%, #0f172a 100%);
    }
    
    /* Input Area */
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
        font-family: 'JetBrains+Mono', monospace;
    }
    
    /* Button */
    .stButton button {
        background-color: #38bdf8 !important;
        color: #0f172a !important;
        border: none !important;
        font-weight: bold !important;
        width: 100%;
        transition: 0.3s;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .stButton button:hover {
        background-color: #7dd3fc !important;
        box-shadow: 0 0 15px #38bdf8;
    }

    /* Cards Glassmorphism */
    .sentiment-card {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin: 20px 0;
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .positive-glow {
        border-left: 5px solid #22c55e;
        color: #4ade80;
    }
    .negative-glow {
        border-left: 5px solid #ef4444;
        color: #f87171;
    }
    
    .tech-title {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .tech-subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }

    /* Metric */
    .metric-container {
        display: flex;
        justify-content: space-between;
        background: #1e293b;
        padding: 10px 20px;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_classifier():
    """Carrega o modelo apenas uma vez e guarda em cache."""
    classifier = SentimentClassifier()
    model_path = 'sentiment_model.joblib'
    vectorizer_path = 'vectorizer.joblib'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        classifier.load(model_path, vectorizer_path)
        return classifier
    return None

# Top Bar / Header
st.markdown('<h1 class="tech-title">SENTINEL-X</h1>', unsafe_allow_html=True)
st.markdown('<p class="tech-subtitle">SISTEMA AVANÇADO DE CLASSIFICAÇÃO TEXTUAL</p>', unsafe_allow_html=True)

classifier = load_classifier()

if classifier is None:
    st.error("ERRO DE SISTEMA: ARQUIVOS_MIA_NAO_ENCONTRADOS")
    st.info("Execute: `python train_model.py` para calibrar o núcleo.")
else:
    # Main Interaction
    st.markdown("### 🧬 ANALISADOR DE ENTRADA")
    text_input = st.text_area("Insira o fragmento de texto para análise:", placeholder="Aguardando entrada de dados...", height=120)
    
    if st.button("EXECUTAR ANÁLISE"):
        if text_input.strip() == "":
            st.warning("ERRO: NENHUMA_ENTRADA_DETECTADA")
        else:
            with st.spinner('PROCESSANDO VETORES DE DADOS...'):
                time.sleep(0.8) # Efeito de processamento
                result = classifier.predict(text_input)
                
                sentiment = result['sentiment']
                confidence = result['confidence']
                
                # Exibição do Resultado
                if sentiment == 'Positivo':
                    st.markdown(f"""
                        <div class="sentiment-card positive-glow">
                            <h2 style="margin:0;">[+] RESPOSTA POSITIVA</h2>
                            <p style="color: #94a3b8;">Sinais detectados: Otimismo, Satisfação, Aprovação.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="sentiment-card negative-glow">
                            <h2 style="margin:0;">[-] RESPOSTA NEGATIVA</h2>
                            <p style="color: #94a3b8;">Sinais detectados: Insatisfação, Reclamação, Pessimismo.</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Metrics Container
                st.markdown(f"""
                    <div class="metric-container">
                        <span style="color: #94a3b8;">PERCENTUAL DE CONFIANÇA:</span>
                        <span style="color: #38bdf8; font-weight: bold;">{confidence:.2%}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                st.progress(confidence)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #475569; font-size: 0.7rem;">VERSÃO DO SISTEMA: 2.1.0-ESTÁVEL | FONTE: DATASET_SENTIMENT140</p>', unsafe_allow_html=True)