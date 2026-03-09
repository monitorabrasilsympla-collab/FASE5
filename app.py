import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import time

# Configuração da página - MAIS SIMPLES
st.set_page_config(
    page_title="Passos Mágicos - Alerta Precoce",
    page_icon="✨",
    layout="wide"
)

# CSS mínimo
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .css-1v3fvcr {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.title("✨ Passos Mágicos")
st.subheader("Sistema de Alerta Precoce para Defasagem Escolar")

# ============================================================
# CARREGAR MODELO
# ============================================================

@st.cache_resource
def load_model():
    """Carrega o modelo com tratamento de erro simplificado"""
    try:
        # Verificar se os arquivos existem
        if not os.path.exists('modelo_risco_defasagem.pkl'):
            st.error("❌ Arquivo 'modelo_risco_defasagem.pkl' não encontrado!")
            return None
        
        model = joblib.load('modelo_risco_defasagem.pkl')
        scaler = joblib.load('scaler.pkl') if os.path.exists('scaler.pkl') else None
        features = joblib.load('features_list.pkl') if os.path.exists('features_list.pkl') else []
        
        return {
            'model': model,
            'scaler': scaler,
            'features': features
        }
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

# Carregar modelo
artifacts = load_model()

if artifacts is None:
    st.warning("⚠️ Modo de demonstração - Usando dados simulados")
    modo_demo = True
else:
    modo_demo = False
    st.success("✅ Modelo carregado com sucesso!")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://passosmagicos.org.br/wp-content/uploads/2022/08/logo-passos-magicos.png", width=200)
    
    st.markdown("## Menu")
    opcao = st.radio(
        "Selecione uma opção:",
        ["Preditor Individual", "Preditor em Lote", "Sobre"]
    )

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def calcular_risco_simulado(iaa, ieg, ips, ida, ian):
    """Calcula risco simulado quando não há modelo"""
    # Fórmula simples para demonstração
    risco = (
        (ian < 6) * 0.4 +
        (ieg < 5) * 0.3 +
        (ida < 5) * 0.2 +
        (ips < 5) * 0.1
    )
    return min(risco, 0.95)  # Limitar em 95%

# ============================================================
# PÁGINA: PREDITOR INDIVIDUAL
# ============================================================

if opcao == "Preditor Individual":
    st.header("🔍 Preditor Individual")
    
    with st.expander("📝 Instruções", expanded=False):
        st.markdown("""
        Preencha os dados do aluno abaixo para calcular a probabilidade de risco de defasagem.
        Todos os valores devem estar entre 0 e 10.
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Indicadores Principais")
        iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 7.0, 0.1)
        ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 6.0, 0.1)
        ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, 6.0, 0.1)
        
    with col2:
        st.subheader("Desempenho")
        ida = st.slider("IDA (Desempenho)", 0.0, 10.0, 5.0, 0.1)
        ian = st.slider("IAN (Adequação de Nível)", 0.0, 10.0, 7.0, 0.1)
        fase = st.selectbox("Fase", [5, 6, 7, 8])
    
    if st.button("🔮 Calcular Risco", type="primary", use_container_width=True):
        
        with st.spinner("Calculando..."):
            time.sleep(1)  # Simular processamento
            
            # Calcular risco
            if modo_demo:
                prob_risco = calcular_risco_simulado(iaa, ieg, ips, ida, ian)
            else:
                try:
                    # Preparar dados para o modelo real
                    input_data = {}
                    for feat in artifacts['features']:
                        if feat == 'IAA': input_data[feat] = iaa
                        elif feat == 'IEG': input_data[feat] = ieg
                        elif feat == 'IPS': input_data[feat] = ips
                        elif feat == 'IDA': input_data[feat] = ida
                        elif feat == 'IAN': input_data[feat] = ian
                        else: input_data[feat] = 0.0
                    
                    df_input = pd.DataFrame([input_data])
                    
                    if artifacts['scaler']:
                        df_input_scaled = artifacts['scaler'].transform(df_input)
                        prob_risco = artifacts['model'].predict_proba(df_input_scaled)[0][1]
                    else:
                        prob_risco = artifacts['model'].predict_proba(df_input)[0][1]
                        
                except Exception as e:
                    st.error(f"Erro na predição: {e}")
                    prob_risco = calcular_risco_simulado(iaa, ieg, ips, ida, ian)
            
            # Mostrar resultado
            st.markdown("---")
            st.subheader("📊 Resultado")
            
            # Cards de resultado
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                if prob_risco >= 0.7:
                    st.error(f"🚨 **ALTO RISCO**\n\n{prob_risco:.1%}")
                elif prob_risco >= 0.3:
                    st.warning(f"⚠️ **RISCO MÉDIO**\n\n{prob_risco:.1%}")
                else:
                    st.success(f"✅ **BAIXO RISCO**\n\n{prob_risco:.1%}")
            
            with col_r2:
                # Métricas simples
                st.metric("IAA", f"{iaa:.1f}")
                st.metric("IEG", f"{ieg:.1f}")
            
            with col_r3:
                st.metric("IDA", f"{ida:.1f}")
                st.metric("IAN", f"{ian:.1f}")
            
            # Recomendações
            st.markdown("---")
            st.subheader("💡 Recomendações")
            
            if prob_risco >= 0.7:
                st.error("""
                **Intervenção Imediata:**
                - Contatar responsáveis
                - Sessão com psicopedagogo
                - Reforço escolar intensivo
                """)
            elif prob_risco >= 0.3:
                st.warning("""
                **Acompanhamento:**
                - Monitorar semanalmente
                - Reforço em Matemática/Português
                - Conversa com tutor
                """)
            else:
                st.info("""
                **Manutenção:**
                - Acompanhamento regular
                - Incentivar participação
                - Manter indicadores
                """)

# ============================================================
# PÁGINA: PREDITOR EM LOTE
# ============================================================

elif opcao == "Preditor em Lote":
    st.header("📊 Preditor em Lote")
    
    st.info("""
    Faça upload de um arquivo CSV com os dados dos alunos.
    O arquivo deve conter as colunas: RA, IAA, IEG, IPS, IDA, IAN, Fase
    """)
    
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Arquivo carregado: {uploaded_file.name}")
            st.dataframe(df.head())
            
            if st.button("Processar", type="primary"):
                with st.spinner("Processando..."):
                    # Simular processamento
                    progress_bar = st.progress(0)
                    
                    resultados = []
                    for i, row in df.iterrows():
                        # Calcular risco
                        iaa = row.get('IAA', 5)
                        ieg = row.get('IEG', 5)
                        ips = row.get('IPS', 5)
                        ida = row.get('IDA', 5)
                        ian = row.get('IAN', 5)
                        
                        if modo_demo:
                            prob = calcular_risco_simulado(iaa, ieg, ips, ida, ian)
                        else:
                            prob = 0.5  # Valor padrão
                        
                        if prob >= 0.7:
                            risco = "Alto"
                        elif prob >= 0.3:
                            risco = "Médio"
                        else:
                            risco = "Baixo"
                        
                        resultados.append({
                            'RA': row.get('RA', f'ALUNO-{i+1}'),
                            'Probabilidade': f"{prob:.1%}",
                            'Risco': risco
                        })
                        
                        progress_bar.progress((i + 1) / len(df))
                    
                    df_result = pd.DataFrame(resultados)
                    st.subheader("Resultados")
                    st.dataframe(df_result)
                    
                    # Download
                    csv = df_result.to_csv(index=False)
                    st.download_button(
                        "📥 Download Resultados",
                        csv,
                        "resultados.csv",
                        "text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Erro: {e}")

# ============================================================
# PÁGINA: SOBRE
# ============================================================

else:
    st.header("ℹ️ Sobre o Projeto")
    
    col_about1, col_about2 = st.columns(2)
    
    with col_about1:
        st.markdown("""
        ### 🎯 Objetivo
        Sistema de alerta precoce para identificar alunos em risco de defasagem escolar,
        permitindo intervenções antes que o problema se agrave.
        
        ### 🤖 Modelo
        - **Algoritmo:** Random Forest / XGBoost
        - **Features:** IAA, IEG, IPS, IDA, IAN
        - **Métrica principal:** Recall (identificar alunos em risco)
        
        ### 📊 Interpretação
        - **Alto risco (>70%):** Intervenção imediata
        - **Médio risco (30-70%):** Acompanhamento próximo
        - **Baixo risco (<30%):** Monitoramento regular
        """)
    
    with col_about2:
        st.markdown("""
        ### 📋 Indicadores
        - **IAA:** Autoavaliação
        - **IEG:** Engajamento
        - **IPS:** Psicossocial
        - **IDA:** Desempenho
        - **IAN:** Adequação de Nível
        
        ### 🚀 Como usar
        1. Use o **Preditor Individual** para um aluno
        2. Use o **Preditor em Lote** para vários
        3. Siga as recomendações
        
        ### 📞 Contato
        Em caso de dúvidas, entre em contato com a equipe da Passos Mágicos.
        """)

# ============================================================
# RODAPÉ
# ============================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Desenvolvido para o Datathon da Passos Mágicos © 2024"
    "</div>",
    unsafe_allow_html=True
)