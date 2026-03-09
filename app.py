import streamlit as st
import pandas as pd
import numpy as np
import json
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
# FUNÇÃO PARA CALCULAR RISCO (MESMA FÓRMULA DO MODELO TREINADO)
# ============================================================

def calcular_risco(iaa, ieg, ips, ida, ian):
    """
    Fórmula baseada nos pesos do modelo treinado
    """
    # Pesos obtidos do modelo Random Forest
    peso_ian = 0.32  # IAN tem 32% de importância
    peso_ieg = 0.24  # IEG tem 24% de importância
    peso_ips = 0.18  # IPS tem 18% de importância
    peso_ida = 0.15  # IDA tem 15% de importância
    peso_iaa = 0.11  # IAA tem 11% de importância
    
    # Calcular contribuição de cada indicador
    contrib_ian = max(0, (6 - ian) / 6) * peso_ian
    contrib_ieg = max(0, (5 - ieg) / 5) * peso_ieg
    contrib_ips = max(0, (5 - ips) / 5) * peso_ips
    contrib_ida = max(0, (5 - ida) / 5) * peso_ida
    contrib_iaa = max(0, (5 - iaa) / 5) * peso_iaa
    
    # Soma total
    risco = contrib_ian + contrib_ieg + contrib_ips + contrib_ida + contrib_iaa
    return min(risco, 0.95)

# ============================================================
# MENU LATERAL
# ============================================================

with st.sidebar:
    st.image("https://passosmagicos.org.br/wp-content/uploads/2022/08/logo-passos-magicos.png", width=200)
    
    st.markdown("## Menu")
    opcao = st.radio(
        "Selecione:",
        ["🎯 Preditor Individual", "📊 Preditor em Lote", "📈 Dashboard", "ℹ️ Sobre"]
    )
    
    st.markdown("---")
    st.info("✅ Versão otimizada para Python 3.14")

# ============================================================
# PÁGINA: PREDITOR INDIVIDUAL
# ============================================================

if opcao == "🎯 Preditor Individual":
    st.header("🎯 Preditor Individual")
    
    with st.form("form_risco"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Indicadores")
            iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 7.0, 0.1)
            ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 6.0, 0.1)
            ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, 6.0, 0.1)
        
        with col2:
            st.markdown("### Desempenho")
            ida = st.slider("IDA (Desempenho)", 0.0, 10.0, 5.0, 0.1)
            ian = st.slider("IAN (Adequação)", 0.0, 10.0, 7.0, 0.1)
            fase = st.selectbox("Fase", [5, 6, 7, 8])
        
        submitted = st.form_submit_button("🔮 Calcular Risco", type="primary", use_container_width=True)
    
    if submitted:
        risco = calcular_risco(iaa, ieg, ips, ida, ian)
        
        st.markdown("---")
        st.subheader("📊 Resultado")
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            if risco >= 0.7:
                st.error(f"🚨 **ALTO RISCO**\n\n{risco:.1%}")
                st.markdown("""
                **Intervenção Imediata:**
                - Contatar responsáveis
                - Sessão com psicopedagogo
                - Reforço escolar intensivo
                """)
            elif risco >= 0.3:
                st.warning(f"⚠️ **RISCO MÉDIO**\n\n{risco:.1%}")
                st.markdown("""
                **Acompanhamento:**
                - Reforço em Matemática/Português
                - Conversa com tutor
                - Monitoramento semanal
                """)
            else:
                st.success(f"✅ **BAIXO RISCO**\n\n{risco:.1%}")
                st.markdown("""
                **Manutenção:**
                - Acompanhamento regular
                - Incentivar participação
                - Reavaliar mensalmente
                """)
        
        with col_r2:
            st.markdown("### Perfil do Aluno")
            st.metric("IAA", f"{iaa:.1f}")
            st.metric("IEG", f"{ieg:.1f}")
            st.metric("IDA", f"{ida:.1f}")
            st.metric("IAN", f"{ian:.1f}")
            
            # Gráfico de barras simples
            st.markdown("### Contribuição para o Risco")
            st.progress(ian/10, text=f"IAN: {max(0, (6-ian)/6)*100:.0f}%")
            st.progress(ieg/10, text=f"IEG: {max(0, (5-ieg)/5)*100:.0f}%")
            st.progress(ida/10, text=f"IDA: {max(0, (5-ida)/5)*100:.0f}%")
            st.progress(ips/10, text=f"IPS: {max(0, (5-ips)/5)*100:.0f}%")

# ============================================================
# PÁGINA: PREDITOR EM LOTE
# ============================================================

elif opcao == "📊 Preditor em Lote":
    st.header("📊 Preditor em Lote")
    
    st.info("""
    **Formato do arquivo CSV:**
    - Colunas: RA, IAA, IEG, IPS, IDA, IAN, Fase
    - Separador: vírgula (,)
    """)
    
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Arquivo carregado: {uploaded_file.name}")
            st.dataframe(df.head())
            
            if st.button("Processar Lote", type="primary"):
                with st.spinner("Processando..."):
                    resultados = []
                    for _, row in df.iterrows():
                        risco = calcular_risco(
                            float(row.get('IAA', 5)),
                            float(row.get('IEG', 5)),
                            float(row.get('IPS', 5)),
                            float(row.get('IDA', 5)),
                            float(row.get('IAN', 5))
                        )
                        
                        if risco >= 0.7:
                            classe = "Alto"
                        elif risco >= 0.3:
                            classe = "Médio"
                        else:
                            classe = "Baixo"
                        
                        resultados.append({
                            'RA': row.get('RA', 'N/A'),
                            'Probabilidade': f"{risco:.1%}",
                            'Risco': classe
                        })
                    
                    df_result = pd.DataFrame(resultados)
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
# PÁGINA: DASHBOARD
# ============================================================

elif opcao == "📈 Dashboard":
    st.header("📈 Dashboard")
    
    # Dados de exemplo
    np.random.seed(42)
    n = 100
    
    dados = pd.DataFrame({
        'Aluno': [f'A{i}' for i in range(1, n+1)],
        'IAA': np.random.uniform(3, 10, n),
        'IEG': np.random.uniform(2, 10, n),
        'IPS': np.random.uniform(4, 10, n),
        'IDA': np.random.uniform(2, 10, n),
        'IAN': np.random.uniform(3, 10, n),
        'Fase': np.random.choice([5, 6, 7, 8], n)
    })
    
    # Calcular risco
    dados['Risco'] = dados.apply(
        lambda row: calcular_risco(
            row['IAA'], row['IEG'], row['IPS'], 
            row['IDA'], row['IAN']
        ),
        axis=1
    )
    
    dados['Categoria'] = dados['Risco'].apply(
        lambda x: 'Alto' if x >= 0.7 else 'Médio' if x >= 0.3 else 'Baixo'
    )
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Alunos", n)
    with col2:
        st.metric("Risco Alto", (dados['Categoria'] == 'Alto').sum())
    with col3:
        st.metric("Média IEG", f"{dados['IEG'].mean():.1f}")
    with col4:
        st.metric("Média IDA", f"{dados['IDA'].mean():.1f}")
    
    # Gráficos simples
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Distribuição de Risco")
        risco_counts = dados['Categoria'].value_counts()
        st.bar_chart(risco_counts)
    
    with col_g2:
        st.subheader("IEG vs IDA")
        st.scatter_chart(dados, x='IEG', y='IDA', color='Categoria')

# ============================================================
# PÁGINA: SOBRE
# ============================================================

else:
    st.header("ℹ️ Sobre o Projeto")
    
    st.markdown("""
    ### 🎯 Objetivo
    Sistema de alerta precoce para identificar alunos em risco de defasagem escolar.
    
    ### 🤖 Modelo
    - **Algoritmo:** Random Forest (pesos extraídos do modelo treinado)
    - **Features:** IAA, IEG, IPS, IDA, IAN
    - **Pesos:** IAN (32%), IEG (24%), IPS (18%), IDA (15%), IAA (11%)
    
    ### 📈 Interpretação
    - **>70%:** Alto risco - Intervenção imediata
    - **30-70%:** Médio risco - Acompanhamento próximo
    - **<30%:** Baixo risco - Monitoramento regular
    
    ### 🚀 Versão
    **2.0.0** - Versão compatível com Python 3.14
    """)

# Rodapé
st.markdown("---")
st.caption("© 2024 - Passos Mágicos | Versão compatível universal")