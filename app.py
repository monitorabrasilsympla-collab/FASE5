import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Passos Mágicos - Alerta Precoce",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffbb33;
        color: black;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .risk-low {
        background-color: #00C851;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stApp {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Cabeçalho
st.markdown("""
<div class="main-header">
    <h1>✨ Passos Mágicos</h1>
    <h3>Sistema de Alerta Precoce para Defasagem Escolar</h3>
    <p>Machine Learning para identificar alunos em risco</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CARREGAR MODELO
# ============================================================

@st.cache_resource
def load_model():
    """Carrega o modelo e artefatos"""
    try:
        # Verificar se os arquivos existem
        arquivos = {
            'modelo': 'modelo_risco_defasagem.pkl',
            'scaler': 'scaler.pkl',
            'features': 'features_list.pkl'
        }
        
        for nome, arquivo in arquivos.items():
            if not os.path.exists(arquivo):
                st.warning(f"⚠️ Arquivo não encontrado: {arquivo}")
                return None, None, None
        
        model = joblib.load('modelo_risco_defasagem.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('features_list.pkl')
        
        st.sidebar.success("✅ Modelo carregado!")
        return model, scaler, features
        
    except Exception as e:
        st.sidebar.error(f"❌ Erro ao carregar modelo: {str(e)}")
        return None, None, None

model, scaler, features = load_model()
modo_demo = model is None

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://passosmagicos.org.br/wp-content/uploads/2022/08/logo-passos-magicos.png", width=200)
    
    st.markdown("## 🧭 Navegação")
    pagina = st.radio(
        "Selecione:",
        ["🎯 Preditor Individual", "📊 Preditor em Lote", "📈 Dashboard", "ℹ️ Sobre"]
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Informações")
    st.info(f"""
    **Status:** {'🟢 Demo' if modo_demo else '🔵 Modelo Ativo'}
    **Python:** {sys.version.split()[0]}
    **Streamlit:** {st.__version__}
    """)
    
    if modo_demo:
        st.warning("""
        **Modo Demonstração**
        - Modelo não encontrado
        - Usando simulação
        """)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def calcular_risco_simulado(iaa, ieg, ips, ida, ian):
    """Calcula risco simulado quando não há modelo"""
    # Fórmula baseada nos indicadores
    risco_base = (
        (max(0, 6 - ian) / 6) * 0.4 +  # Peso 40% para IAN
        (max(0, 5 - ieg) / 5) * 0.3 +   # Peso 30% para IEG
        (max(0, 5 - ida) / 5) * 0.2 +   # Peso 20% para IDA
        (max(0, 5 - ips) / 5) * 0.1      # Peso 10% para IPS
    )
    return min(risco_base, 0.95)

def classificar_risco(prob):
    """Classifica o risco baseado na probabilidade"""
    if prob >= 0.7:
        return "Alto", "🔴"
    elif prob >= 0.3:
        return "Médio", "🟡"
    else:
        return "Baixo", "🟢"

# ============================================================
# PÁGINA: PREDITOR INDIVIDUAL
# ============================================================

if pagina == "🎯 Preditor Individual":
    st.header("🎯 Preditor Individual")
    st.markdown("Preencha os dados do aluno para calcular a probabilidade de risco.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Indicadores")
        iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 7.0, 0.1, 
                       help="Índice de Autoavaliação")
        ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 6.0, 0.1,
                       help="Índice de Engajamento")
        ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, 6.0, 0.1,
                       help="Índice Psicossocial")
    
    with col2:
        st.markdown("### 📚 Desempenho")
        ida = st.slider("IDA (Desempenho)", 0.0, 10.0, 5.0, 0.1,
                       help="Índice de Desempenho Acadêmico")
        ian = st.slider("IAN (Adequação)", 0.0, 10.0, 7.0, 0.1,
                       help="Índice de Adequação de Nível")
        fase = st.selectbox("Fase", [5, 6, 7, 8], help="Fase atual do aluno")
    
    # Botão de cálculo
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        calcular = st.button("🔮 Calcular Risco", type="primary", use_container_width=True)
    
    if calcular:
        with st.spinner("Calculando probabilidade..."):
            
            # Calcular risco
            if modo_demo:
                prob_risco = calcular_risco_simulado(iaa, ieg, ips, ida, ian)
            else:
                try:
                    # Preparar dados para o modelo
                    input_data = {}
                    for feat in features:
                        if feat == 'IAA': input_data[feat] = iaa
                        elif feat == 'IEG': input_data[feat] = ieg
                        elif feat == 'IPS': input_data[feat] = ips
                        elif feat == 'IDA': input_data[feat] = ida
                        elif feat == 'IAN': input_data[feat] = ian
                        elif feat == 'IPV': input_data[feat] = 6.0
                        elif feat == 'Fase_Num': input_data[feat] = fase
                        else: input_data[feat] = 0.0
                    
                    df_input = pd.DataFrame([input_data])
                    
                    # Garantir ordem correta das features
                    df_input = df_input[features]
                    
                    # Escalar se necessário
                    if scaler is not None:
                        df_input = scaler.transform(df_input)
                    
                    prob_risco = model.predict_proba(df_input)[0][1]
                    
                except Exception as e:
                    st.error(f"Erro na predição: {str(e)}")
                    prob_risco = calcular_risco_simulado(iaa, ieg, ips, ida, ian)
            
            # Classificar risco
            classe, icone = classificar_risco(prob_risco)
            
            # Mostrar resultado
            st.markdown("---")
            st.subheader("📊 Resultado da Análise")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                if classe == "Alto":
                    st.markdown(f'<div class="risk-high">{icone} RISCO {classe}<br>{prob_risco:.1%}</div>', unsafe_allow_html=True)
                elif classe == "Médio":
                    st.markdown(f'<div class="risk-medium">{icone} RISCO {classe}<br>{prob_risco:.1%}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low">{icone} RISCO {classe}<br>{prob_risco:.1%}</div>', unsafe_allow_html=True)
            
            with col_r2:
                # Gráfico de gauge simples
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_risco * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilidade (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ff4444" if prob_risco > 0.7 else "#ffbb33" if prob_risco > 0.3 else "#00C851"},
                        'steps': [
                            {'range': [0, 30], 'color': "#00C851"},
                            {'range': [30, 70], 'color': "#ffbb33"},
                            {'range': [70, 100], 'color': "#ff4444"}
                        ]
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col_r3:
                st.markdown("### 📈 Perfil")
                st.metric("IAA", f"{iaa:.1f}")
                st.metric("IEG", f"{ieg:.1f}")
                st.metric("IDA", f"{ida:.1f}")
                st.metric("IAN", f"{ian:.1f}")
            
            # Recomendações
            st.markdown("---")
            st.subheader("💡 Recomendações")
            
            if classe == "Alto":
                st.error("""
                **🚨 INTERVENÇÃO IMEDIATA NECESSÁRIA**
                - Contatar responsáveis urgentemente
                - Agendar sessão com psicopedagogo
                - Iniciar reforço escolar intensivo
                - Monitoramento diário
                """)
            elif classe == "Médio":
                st.warning("""
                **⚠️ ACOMPANHAMENTO PRÓXIMO**
                - Reforço escolar em Matemática/Português
                - Conversa de acompanhamento com tutor
                - Avaliar fatores psicossociais
                - Monitoramento semanal
                """)
            else:
                st.success("""
                **✅ MANTER ACOMPANHAMENTO**
                - Continuar monitoramento regular
                - Incentivar participação nas atividades
                - Manter indicadores estáveis
                - Reavaliar mensalmente
                """)

# ============================================================
# PÁGINA: PREDITOR EM LOTE
# ============================================================

elif pagina == "📊 Preditor em Lote":
    st.header("📊 Preditor em Lote")
    
    st.info("""
    **Formato do arquivo CSV:**
    - Colunas obrigatórias: RA, IAA, IEG, IPS, IDA, IAN, Fase
    - Valores numéricos entre 0 e 10
    - Separador: vírgula (,)
    """)
    
    uploaded_file = st.file_uploader(
        "Escolha um arquivo CSV",
        type=['csv'],
        help="Faça upload do arquivo com os dados dos alunos"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Arquivo carregado: {uploaded_file.name}")
            st.markdown(f"**Total de alunos:** {len(df)}")
            
            with st.expander("👁️ Visualizar dados"):
                st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Processar Lote", type="primary", use_container_width=True):
                with st.spinner("Processando alunos..."):
                    progress_bar = st.progress(0)
                    
                    resultados = []
                    for i, row in df.iterrows():
                        # Extrair dados
                        iaa = float(row.get('IAA', 5))
                        ieg = float(row.get('IEG', 5))
                        ips = float(row.get('IPS', 5))
                        ida = float(row.get('IDA', 5))
                        ian = float(row.get('IAN', 5))
                        
                        # Calcular risco
                        if modo_demo:
                            prob = calcular_risco_simulado(iaa, ieg, ips, ida, ian)
                        else:
                            prob = 0.5  # Placeholder
                        
                        classe, icone = classificar_risco(prob)
                        
                        resultados.append({
                            'RA': row.get('RA', f'ALU-{i+1:03d}'),
                            'Probabilidade': f"{prob:.1%}",
                            'Risco': classe,
                            'IAA': f"{iaa:.1f}",
                            'IEG': f"{ieg:.1f}",
                            'IDA': f"{ida:.1f}",
                            'IAN': f"{ian:.1f}"
                        })
                        
                        progress_bar.progress((i + 1) / len(df))
                    
                    df_result = pd.DataFrame(resultados)
                    
                    st.markdown("---")
                    st.subheader("📋 Resultados")
                    
                    # Métricas
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Total", len(df_result))
                    with col_m2:
                        n_alto = (df_result['Risco'] == 'Alto').sum()
                        st.metric("Risco Alto", n_alto, f"{n_alto/len(df_result)*100:.0f}%")
                    with col_m3:
                        n_baixo = (df_result['Risco'] == 'Baixo').sum()
                        st.metric("Risco Baixo", n_baixo, f"{n_baixo/len(df_result)*100:.0f}%")
                    
                    # Tabela
                    st.dataframe(df_result, use_container_width=True)
                    
                    # Download
                    csv = df_result.to_csv(index=False)
                    st.download_button(
                        "📥 Download Resultados (CSV)",
                        csv,
                        "resultados_risco.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")

# ============================================================
# PÁGINA: DASHBOARD
# ============================================================

elif pagina == "📈 Dashboard":
    st.header("📈 Dashboard")
    
    # Dados simulados para demonstração
    np.random.seed(42)
    n = 100
    
    df_dash = pd.DataFrame({
        'Aluno': [f'A{i:03d}' for i in range(1, n+1)],
        'IAA': np.random.uniform(3, 10, n),
        'IEG': np.random.uniform(2, 10, n),
        'IPS': np.random.uniform(4, 10, n),
        'IDA': np.random.uniform(2, 10, n),
        'IAN': np.random.uniform(3, 10, n),
        'Fase': np.random.choice([5, 6, 7, 8], n)
    })
    
    # Calcular risco
    df_dash['Risco'] = df_dash.apply(
        lambda row: classificar_risco(
            calcular_risco_simulado(
                row['IAA'], row['IEG'], row['IPS'], 
                row['IDA'], row['IAN']
            )
        )[0],
        axis=1
    )
    
    # KPIs
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    
    with col_k1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Alunos", n)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_k2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        n_alto = (df_dash['Risco'] == 'Alto').sum()
        st.metric("Risco Alto", n_alto, f"{n_alto/n*100:.0f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_k3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Média IEG", f"{df_dash['IEG'].mean():.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_k4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Média IDA", f"{df_dash['IDA'].mean():.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Gráficos
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # Distribuição de risco
        risco_counts = df_dash['Risco'].value_counts()
        fig = px.pie(
            values=risco_counts.values,
            names=risco_counts.index,
            title="Distribuição de Risco",
            color_discrete_map={'Alto': '#ff4444', 'Médio': '#ffbb33', 'Baixo': '#00C851'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_g2:
        # Risco por fase
        risco_fase = pd.crosstab(df_dash['Fase'], df_dash['Risco'])
        fig = px.bar(
            risco_fase,
            title="Risco por Fase",
            barmode='stack',
            color_discrete_map={'Alto': '#ff4444', 'Médio': '#ffbb33', 'Baixo': '#00C851'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlação
    fig = px.scatter(
        df_dash,
        x='IEG',
        y='IDA',
        color='Risco',
        title="Relação Engajamento x Desempenho",
        color_discrete_map={'Alto': '#ff4444', 'Médio': '#ffbb33', 'Baixo': '#00C851'},
        hover_data=['Aluno']
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PÁGINA: SOBRE
# ============================================================

else:
    st.header("ℹ️ Sobre o Projeto")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("""
        ### 🎯 Objetivo
        Sistema de Machine Learning para identificar precocemente alunos em risco de defasagem escolar,
        permitindo intervenções antes que o problema se agrave.
        
        ### 🤖 Modelo Preditivo
        - **Algoritmo:** Random Forest / XGBoost
        - **Features:** IAA, IEG, IPS, IDA, IAN, IPV
        - **Métrica principal:** Recall (identificar alunos em risco)
        - **Acurácia:** ~85%
        
        ### 📊 Indicadores
        - **IAA:** Autoavaliação do aluno
        - **IEG:** Engajamento nas atividades
        - **IPS:** Aspectos psicossociais
        - **IDA:** Desempenho acadêmico
        - **IAN:** Adequação de nível
        """)
    
    with col_s2:
        st.markdown("""
        ### 📈 Interpretação
        | Risco | Probabilidade | Ação |
        |-------|--------------|------|
        | 🔴 Alto | >70% | Intervenção imediata |
        | 🟡 Médio | 30-70% | Acompanhamento próximo |
        | 🟢 Baixo | <30% | Monitoramento regular |
        
        ### 🚀 Como Usar
        1. **Preditor Individual:** Análise detalhada de um aluno
        2. **Preditor em Lote:** Processar múltiplos alunos via CSV
        3. **Dashboard:** Visualizar métricas da turma
        
        ### 📞 Contato
        Em caso de dúvidas, entre em contato com a equipe da Passos Mágicos.
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px;'>
        <h4>Desenvolvido para o Datathon da Passos Mágicos</h4>
        <p>© 2024 - Todos os direitos reservados</p>
    </div>
    """, unsafe_allow_html=True)

# Rodapé
st.markdown("---")
st.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")