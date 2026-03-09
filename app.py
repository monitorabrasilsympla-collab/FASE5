import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Configuração da página
st.set_page_config(
    page_title="Passos Mágicos",
    page_icon="✨",
    layout="centered"
)

# Título
st.title("✨ Passos Mágicos")
st.subheader("Sistema de Alerta Precoce")

# ============================================================
# FUNÇÃO PARA CARREGAR MODELO COM PICKLE
# ============================================================

@st.cache_resource
def load_model():
    """Tenta carregar o modelo com pickle"""
    try:
        if os.path.exists('modelo_risco_defasagem.pkl'):
            with open('modelo_risco_defasagem.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            return None
    except:
        return None

# ============================================================
# MENU LATERAL
# ============================================================

st.sidebar.image("https://passosmagicos.org.br/wp-content/uploads/2022/08/logo-passos-magicos.png", width=200)

opcao = st.sidebar.radio(
    "Menu",
    ["Preditor Individual", "Sobre", "Status do Sistema"]
)

# ============================================================
# PÁGINA: PREDITOR INDIVIDUAL
# ============================================================

if opcao == "Preditor Individual":
    st.header("🔍 Preditor Individual")
    
    # Formulário
    with st.form("form_aluno"):
        col1, col2 = st.columns(2)
        
        with col1:
            iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 7.0)
            ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 6.0)
            ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, 6.0)
        
        with col2:
            ida = st.slider("IDA (Desempenho)", 0.0, 10.0, 5.0)
            ian = st.slider("IAN (Adequação)", 0.0, 10.0, 7.0)
            fase = st.selectbox("Fase", [5, 6, 7, 8])
        
        submitted = st.form_submit_button("Calcular Risco", type="primary")
    
    if submitted:
        # Fórmula simples baseada nos indicadores
        risco = (
            (max(0, 6 - ian) / 6) * 0.4 +
            (max(0, 5 - ieg) / 5) * 0.3 +
            (max(0, 5 - ida) / 5) * 0.2 +
            (max(0, 5 - ips) / 5) * 0.1
        )
        risco = min(risco, 0.95)
        
        st.markdown("---")
        st.subheader("📊 Resultado")
        
        # Mostrar resultado com cores
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            if risco >= 0.7:
                st.error(f"🚨 **ALTO RISCO**\n\n{risco:.1%}")
            elif risco >= 0.3:
                st.warning(f"⚠️ **RISCO MÉDIO**\n\n{risco:.1%}")
            else:
                st.success(f"✅ **BAIXO RISCO**\n\n{risco:.1%}")
        
        with col_r2:
            # Gráfico simples de barras
            st.write("Contribuição para o risco:")
            st.progress(ian/10, text=f"IAN: {(max(0,6-ian)/6)*100:.0f}%")
            st.progress(ieg/10, text=f"IEG: {(max(0,5-ieg)/5)*100:.0f}%")
            st.progress(ida/10, text=f"IDA: {(max(0,5-ida)/5)*100:.0f}%")
            st.progress(ips/10, text=f"IPS: {(max(0,5-ips)/5)*100:.0f}%")
        
        # Recomendações
        st.markdown("---")
        st.subheader("💡 Recomendações")
        
        if risco >= 0.7:
            st.error("""
            **INTERVENÇÃO IMEDIATA:**
            - Contatar responsáveis
            - Sessão com psicopedagogo
            - Reforço escolar intensivo
            - Monitoramento diário
            """)
        elif risco >= 0.3:
            st.warning("""
            **ACOMPANHAMENTO PRÓXIMO:**
            - Reforço em Matemática/Português
            - Conversa com tutor
            - Avaliar fatores psicossociais
            - Monitoramento semanal
            """)
        else:
            st.success("""
            **MANTER ACOMPANHAMENTO:**
            - Continuar monitoramento regular
            - Incentivar participação
            - Manter indicadores estáveis
            - Reavaliar mensalmente
            """)

# ============================================================
# PÁGINA: SOBRE
# ============================================================

elif opcao == "Sobre":
    st.header("ℹ️ Sobre o Projeto")
    
    st.markdown("""
    ### 🎯 Objetivo
    Sistema de alerta precoce para identificar alunos em risco de defasagem escolar.
    
    ### 📊 Indicadores
    - **IAA (Autoavaliação):** Como o aluno se percebe
    - **IEG (Engajamento):** Participação nas atividades
    - **IPS (Psicossocial):** Aspectos emocionais e sociais
    - **IDA (Desempenho):** Notas e aproveitamento
    - **IAN (Adequação):** Nível de defasagem atual
    
    ### 📈 Interpretação dos Resultados
    - **Alto risco (>70%):** Necessita intervenção imediata
    - **Médio risco (30-70%):** Requer acompanhamento próximo
    - **Baixo risco (<30%):** Manter monitoramento regular
    
    ### 🚀 Versão
    **1.0.0** - Versão de demonstração
    """)

# ============================================================
# PÁGINA: STATUS DO SISTEMA
# ============================================================

else:
    st.header("🖥️ Status do Sistema")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.subheader("Informações")
        st.write(f"**Python:** {os.sys.version}")
        st.write(f"**Streamlit:** {st.__version__}")
        st.write(f"**Pandas:** {pd.__version__}")
        st.write(f"**NumPy:** {np.__version__}")
    
    with col_s2:
        st.subheader("Arquivos do Modelo")
        
        # Verificar arquivos
        if os.path.exists('modelo_risco_defasagem.pkl'):
            st.success("✅ modelo_risco_defasagem.pkl encontrado")
        else:
            st.warning("⚠️ modelo_risco_defasagem.pkl não encontrado")
        
        if os.path.exists('scaler.pkl'):
            st.success("✅ scaler.pkl encontrado")
        else:
            st.warning("⚠️ scaler.pkl não encontrado")
        
        if os.path.exists('features_list.pkl'):
            st.success("✅ features_list.pkl encontrado")
        else:
            st.warning("⚠️ features_list.pkl não encontrado")
    
    st.info("Modo de demonstração ativo - usando fórmula simples")

# Rodapé
st.markdown("---")
st.caption("© 2024 - Passos Mágicos | Datathon")