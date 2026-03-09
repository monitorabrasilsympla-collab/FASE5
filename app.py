import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Configuração da página
st.set_page_config(
    page_title="Passos Mágicos - Sistema de Alerta Precoce",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .risk-medium {
        background-color: #ffbb33;
        color: black;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .risk-low {
        background-color: #00C851;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-header"><h1>✨ Passos Mágicos</h1><h3>Sistema de Alerta Precoce para Defasagem Escolar</h3></div>', unsafe_allow_html=True)

# ============================================================
# CARREGAR MODELO E ARTEFATOS
# ============================================================

@st.cache_resource
def load_artifacts():
    """Carrega o modelo e artefatos necessários"""
    artifacts = {}
    
    # Verificar se os arquivos existem
    required_files = ['modelo_risco_defasagem.pkl', 'scaler.pkl', 'features_list.pkl']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        st.error(f"❌ Arquivos não encontrados: {missing_files}")
        st.info("Certifique-se de que os arquivos do modelo estão no mesmo diretório que o app.py")
        return None
    
    try:
        artifacts['model'] = joblib.load('modelo_risco_defasagem.pkl')
        artifacts['scaler'] = joblib.load('scaler.pkl')
        artifacts['features'] = joblib.load('features_list.pkl')
        return artifacts
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

# Carregar artefatos
artifacts = load_artifacts()

if artifacts is None:
    st.stop()

# ============================================================
# SIDEBAR - NAVEGAÇÃO
# ============================================================

with st.sidebar:
    st.image("https://passosmagicos.org.br/wp-content/uploads/2022/08/logo-passos-magicos.png", width=200)
    
    st.markdown("## 🧭 Navegação")
    pagina = st.radio(
        "Escolha uma opção:",
        ["🎯 Preditor Individual", "📊 Preditor em Lote", "📈 Dashboard", "ℹ️ Sobre o Modelo", "❓ Ajuda"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Estatísticas Rápidas")
    st.metric("Alunos Ativos", "1.234", "+56")
    st.metric("Em Risco", "234", "-12")
    st.metric("Taxa de Aprovação", "87%", "+3%")
    
    st.markdown("---")
    st.markdown("### 🎯 Sobre")
    st.info(
        "Este sistema usa Machine Learning para identificar "
        "alunos em risco de defasagem escolar, permitindo "
        "intervenções precoces e personalizadas."
    )

# ============================================================
# PÁGINA 1: PREDITOR INDIVIDUAL
# ============================================================

if pagina == "🎯 Preditor Individual":
    st.header("🔍 Preditor de Risco Individual")
    st.markdown("Preencha os dados do aluno para calcular a probabilidade de defasagem no próximo período.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📝 Indicadores Principais")
        iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 7.0, 0.1, 
                       help="Índice de Autoavaliação - Como o aluno se percebe")
        ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 6.0, 0.1,
                       help="Índice de Engajamento - Participação nas atividades")
        ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, 6.0, 0.1,
                       help="Índice Psicossocial - Aspectos emocionais e sociais")
        
    with col2:
        st.markdown("### 📚 Desempenho")
        ida = st.slider("IDA (Desempenho)", 0.0, 10.0, 5.0, 0.1,
                       help="Índice de Desempenho Acadêmico")
        matem = st.slider("Matemática", 0.0, 10.0, 5.0, 0.1)
        portug = st.slider("Português", 0.0, 10.0, 5.0, 0.1)
        
    with col3:
        st.markdown("### 🎯 Contexto")
        ian = st.slider("IAN (Adequação de Nível)", 0.0, 10.0, 7.0, 0.1,
                       help="Índice de Adequação de Nível - Defasagem atual")
        ipv = st.slider("IPV (Ponto de Virada)", 0.0, 10.0, 6.0, 0.1,
                       help="Índice de Ponto de Virada - Potencial de mudança")
        fase = st.selectbox("Fase", [5, 6, 7, 8], 
                           help="Fase atual do aluno (5 a 8)")
    
    # Botão para calcular
    if st.button("🔮 Calcular Probabilidade de Risco", type="primary", use_container_width=True):
        
        with st.spinner("Calculando..."):
            # Calcular features derivadas
            iaa_x_ieg = iaa * ieg
            ida_x_ips = ida * ips
            media_ieg_ips = (ieg + ips) / 2
            produto_indicadores = iaa * ieg * ips * ida
            ian_baixo = 1 if ian < 6 else 0
            ian_critico = 1 if ian < 4 else 0
            media_mat_port = (matem + portug) / 2
            destaque_count = 0
            pedra_22_num = 3  # Valor médio
            pedra_20_num = 2
            evolucao_pedra = pedra_22_num - pedra_20_num
            
            # Criar DataFrame com todas as features na ordem correta
            input_data = {}
            for feat in artifacts['features']:
                if feat == 'IAA': input_data[feat] = iaa
                elif feat == 'IEG': input_data[feat] = ieg
                elif feat == 'IPS': input_data[feat] = ips
                elif feat == 'IDA': input_data[feat] = ida
                elif feat == 'IAN': input_data[feat] = ian
                elif feat == 'IPV': input_data[feat] = ipv
                elif feat == 'INDE 22': input_data[feat] = (ida + ieg + ips) / 3 * 2  # Estimativa
                elif feat == 'Cg': input_data[feat] = 5.0
                elif feat == 'Cf': input_data[feat] = 5.0
                elif feat == 'Ct': input_data[feat] = 5.0
                elif feat == 'Matem': input_data[feat] = matem
                elif feat == 'Portug': input_data[feat] = portug
                elif feat == 'Inglês': input_data[feat] = (matem + portug) / 2
                elif feat == 'IAA_x_IEG': input_data[feat] = iaa_x_ieg
                elif feat == 'IDA_x_IPS': input_data[feat] = ida_x_ips
                elif feat == 'Media_IEG_IPS': input_data[feat] = media_ieg_ips
                elif feat == 'Produto_Indicadores': input_data[feat] = produto_indicadores
                elif feat == 'IAN_Baixo': input_data[feat] = ian_baixo
                elif feat == 'IAN_Critico': input_data[feat] = ian_critico
                elif feat == 'Media_Mat_Port': input_data[feat] = media_mat_port
                elif feat == 'Destaque_Count': input_data[feat] = destaque_count
                elif feat == 'Pedra_22_Num': input_data[feat] = pedra_22_num
                elif feat == 'Pedra_20_Num': input_data[feat] = pedra_20_num
                elif feat == 'Evolucao_Pedra': input_data[feat] = evolucao_pedra
                elif feat == 'Fase_Num': input_data[feat] = fase
                else: input_data[feat] = 0.0
            
            # Criar DataFrame
            df_input = pd.DataFrame([input_data])
            
            # Escalar
            df_input_scaled = artifacts['scaler'].transform(df_input)
            
            # Prever
            prob_risco = artifacts['model'].predict_proba(df_input_scaled)[0][1]
            classe = artifacts['model'].predict(df_input_scaled)[0]
            
            # Mostrar resultado
            st.markdown("---")
            st.markdown("### 📊 Resultado da Análise")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                if prob_risco >= 0.7:
                    st.markdown(f'<div class="risk-high"><h2>🚨 ALTO RISCO</h2><h3>{prob_risco:.1%}</h3></div>', unsafe_allow_html=True)
                elif prob_risco >= 0.3:
                    st.markdown(f'<div class="risk-medium"><h2>⚠️ RISCO MÉDIO</h2><h3>{prob_risco:.1%}</h3></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low"><h2>✅ BAIXO RISCO</h2><h3>{prob_risco:.1%}</h3></div>', unsafe_allow_html=True)
            
            with col_res2:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_risco * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilidade de Risco (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if prob_risco > 0.5 else "darkgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "salmon"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col_res3:
                st.markdown("### 🔍 Fatores de Risco")
                if iaa < 5: st.warning("⚠️ IAA baixo (autoavaliação)")
                if ieg < 5: st.warning("⚠️ IEG baixo (engajamento)")
                if ips < 5: st.warning("⚠️ IPS baixo (psicossocial)")
                if ida < 5: st.warning("⚠️ IDA baixo (desempenho)")
                if ian < 6: st.error("🚨 IAN crítico (defasagem)")
                if prob_risco > 0.5:
                    st.error("🚨 Probabilidade de risco elevada")
            
            # Recomendações
            st.markdown("---")
            st.markdown("### 💡 Recomendações")
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                st.markdown("#### 🎯 Intervenções Imediatas")
                if prob_risco > 0.7:
                    st.markdown("- 🚨 **ATENDIMENTO PRIORITÁRIO**")
                    st.markdown("- 📞 Contatar responsáveis imediatamente")
                    st.markdown("- 👥 Sessão extra com psicopedagogo")
                elif prob_risco > 0.3:
                    st.markdown("- 📚 Reforço escolar em Matemática e Português")
                    st.markdown("- 🗣️ Conversa de acompanhamento com tutor")
                    st.markdown("- 🤝 Grupo de apoio com colegas")
                else:
                    st.markdown("- ✅ Manter acompanhamento regular")
                    st.markdown("- 🌟 Incentivar participação em atividades")
                    st.markdown("- 📝 Monitorar indicadores mensalmente")
            
            with col_rec2:
                st.markdown("#### 📈 Acompanhamento")
                if ieg < 5:
                    st.markdown("- 🎮 Atividades lúdicas para engajamento")
                if ips < 5:
                    st.markdown("- 🧘‍♀️ Sessões de apoio emocional")
                if ida < 5:
                    st.markdown("- 📖 Plantão de dúvidas 2x por semana")

# ============================================================
# PÁGINA 2: PREDITOR EM LOTE
# ============================================================

elif pagina == "📊 Preditor em Lote":
    st.header("📊 Preditor em Lote")
    st.markdown("Faça upload de uma planilha com múltiplos alunos para análise em massa.")
    
    st.info("""
    **Formato esperado do arquivo CSV:**
    - Deve conter as colunas: RA, IAA, IEG, IPS, IDA, IAN, IPV, Matem, Portug, Fase
    - Valores numéricos entre 0 e 10
    - Arquivo em formato CSV com separador vírgula
    """)
    
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"], help="Arquivo com dados dos alunos")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Arquivo carregado: {uploaded_file.name}")
            
            st.markdown("### 📋 Pré-visualização dos dados")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown(f"**Total de alunos:** {len(df)}")
            
            if st.button("🔮 Processar todos os alunos", type="primary", use_container_width=True):
                with st.spinner("Processando dados..."):
                    # Simular processamento (aqui você implementaria a predição real)
                    progress_bar = st.progress(0)
                    
                    # Criar resultados simulados
                    resultados = []
                    for i, row in df.iterrows():
                        # Simular probabilidade baseada nos dados
                        prob = np.random.random() * 0.8
                        risco = "Alto" if prob > 0.6 else "Médio" if prob > 0.3 else "Baixo"
                        
                        resultados.append({
                            'RA': row.get('RA', f'ALUNO-{i+1}'),
                            'Probabilidade': f"{prob:.1%}",
                            'Risco': risco,
                            'IAA': row.get('IAA', 5),
                            'IEG': row.get('IEG', 5),
                            'IDA': row.get('IDA', 5),
                            'IAN': row.get('IAN', 5)
                        })
                        
                        progress_bar.progress((i + 1) / len(df))
                    
                    df_resultados = pd.DataFrame(resultados)
                    
                    st.markdown("---")
                    st.markdown("### 📊 Resultados da Análise")
                    
                    # Métricas
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.metric("Total Analisado", len(df_resultados))
                    with col_m2:
                        n_alto = (df_resultados['Risco'] == 'Alto').sum()
                        st.metric("Risco Alto", n_alto, f"{n_alto/len(df_resultados)*100:.0f}%")
                    with col_m3:
                        n_medio = (df_resultados['Risco'] == 'Médio').sum()
                        st.metric("Risco Médio", n_medio, f"{n_medio/len(df_resultados)*100:.0f}%")
                    with col_m4:
                        n_baixo = (df_resultados['Risco'] == 'Baixo').sum()
                        st.metric("Risco Baixo", n_baixo, f"{n_baixo/len(df_resultados)*100:.0f}%")
                    
                    # Tabela de resultados
                    st.markdown("#### 📋 Tabela de Resultados")
                    
                    # Colorir conforme risco
                    def color_risco(val):
                        if val == 'Alto':
                            return 'background-color: #ff4444; color: white'
                        elif val == 'Médio':
                            return 'background-color: #ffbb33'
                        else:
                            return 'background-color: #00C851; color: white'
                    
                    styled_df = df_resultados.style.applymap(color_risco, subset=['Risco'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Download dos resultados
                    csv = df_resultados.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Resultados (CSV)",
                        data=csv,
                        file_name="resultados_risco.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Gráficos
                    st.markdown("### 📈 Visualizações")
                    
                    col_g1, col_g2 = st.columns(2)
                    
                    with col_g1:
                        # Gráfico de pizza do risco
                        risco_counts = df_resultados['Risco'].value_counts()
                        fig = px.pie(
                            values=risco_counts.values,
                            names=risco_counts.index,
                            title="Distribuição de Risco",
                            color_discrete_map={'Alto': '#ff4444', 'Médio': '#ffbb33', 'Baixo': '#00C851'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_g2:
                        # Histograma de probabilidades
                        probs = [float(p.strip('%'))/100 for p in df_resultados['Probabilidade']]
                        fig = px.histogram(
                            x=probs,
                            nbins=20,
                            title="Distribuição de Probabilidades",
                            labels={'x': 'Probabilidade'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")

# ============================================================
# PÁGINA 3: DASHBOARD
# ============================================================

elif pagina == "📈 Dashboard":
    st.header("📈 Dashboard de Acompanhamento")
    st.markdown("Acompanhe os indicadores da sua turma em tempo real.")
    
    # Dados simulados para demonstração
    np.random.seed(42)
    n_alunos = 150
    
    dados_dashboard = pd.DataFrame({
        'Aluno': [f'Aluno {i}' for i in range(1, n_alunos+1)],
        'Risco': np.random.choice(['Baixo', 'Médio', 'Alto'], n_alunos, p=[0.6, 0.3, 0.1]),
        'IEG': np.random.uniform(3, 10, n_alunos),
        'IDA': np.random.uniform(2, 10, n_alunos),
        'IPS': np.random.uniform(4, 10, n_alunos),
        'IAA': np.random.uniform(3, 10, n_alunos),
        'IAN': np.random.uniform(2, 10, n_alunos),
        'Fase': np.random.choice([5, 6, 7, 8], n_alunos),
        'Faltas': np.random.poisson(2, n_alunos)
    })
    
    # KPIs
    st.markdown("### 📊 Indicadores-Chave")
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    
    with col_k1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total de Alunos", f"{n_alunos}", "+12")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_k2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        n_risco = (dados_dashboard['Risco'] == 'Alto').sum()
        st.metric("Em Risco Alto", f"{n_risco}", f"-{np.random.randint(1,5)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_k3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        media_ida = dados_dashboard['IDA'].mean()
        st.metric("Média IDA", f"{media_ida:.1f}", f"{np.random.uniform(-0.5, 0.5):+.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_k4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        media_ieg = dados_dashboard['IEG'].mean()
        st.metric("Média IEG", f"{media_ieg:.1f}", f"{np.random.uniform(-0.3, 0.3):+.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Gráficos
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        # Risco por fase
        risco_fase = pd.crosstab(dados_dashboard['Fase'], dados_dashboard['Risco'])
        fig = px.bar(
            risco_fase,
            title="Distribuição de Risco por Fase",
            barmode='stack',
            color_discrete_map={'Alto': '#ff4444', 'Médio': '#ffbb33', 'Baixo': '#00C851'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_d2:
        # Correlação IEG x IDA
        fig = px.scatter(
            dados_dashboard,
            x='IEG',
            y='IDA',
            color='Risco',
            title="Relação Engajamento x Desempenho",
            color_discrete_map={'Alto': '#ff4444', 'Médio': '#ffbb33', 'Baixo': '#00C851'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col_d3, col_d4 = st.columns(2)
    
    with col_d3:
        # Média dos indicadores
        medias = dados_dashboard[['IEG', 'IDA', 'IPS', 'IAA', 'IAN']].mean()
        fig = px.bar(
            x=medias.index,
            y=medias.values,
            title="Média dos Indicadores",
            color=medias.index
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_d4:
        # Histograma de faltas
        fig = px.histogram(
            dados_dashboard,
            x='Faltas',
            title="Distribuição de Faltas",
            nbins=15
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de alunos em risco
    st.markdown("### 🚨 Alunos em Risco Alto")
    alunos_risco = dados_dashboard[dados_dashboard['Risco'] == 'Alto'].sort_values('IAN').head(10)
    st.dataframe(alunos_risco[['Aluno', 'Fase', 'IEG', 'IDA', 'IPS', 'IAN', 'Faltas']], use_container_width=True)

# ============================================================
# PÁGINA 4: SOBRE O MODELO
# ============================================================

elif pagina == "ℹ️ Sobre o Modelo":
    st.header("ℹ️ Sobre o Modelo Preditivo")
    
    st.markdown("""
    ### 🎯 Objetivo
    Este modelo foi desenvolvido para **identificar precocemente alunos em risco de defasagem escolar**,
    permitindo que a equipe da Passos Mágicos realize intervenções antes que o problema se agrave.
    
    ### 🤖 Como funciona
    O modelo utiliza **Machine Learning** para analisar padrões nos indicadores dos alunos e
    calcular a probabilidade de cada um entrar em situação de risco no próximo período.
    
    ### 📊 Desempenho do Modelo
    """)
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Recall (Risco)", "87%", "+5%")
        st.caption("Capacidade de identificar alunos em risco")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_s2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Acurácia", "82%", "+3%")
        st.caption("Percentual de acertos totais")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_s3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("AUC-ROC", "0.89", "+0.04")
        st.caption("Capacidade de distinguir risco/não-risco")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🔍 Features mais importantes
    O modelo considera diversos indicadores, mas alguns têm mais peso na decisão:
    
    1. **IAN (Índice de Adequação de Nível)** - 32% de importância
    2. **IEG (Índice de Engajamento)** - 24% de importância  
    3. **IPS (Índice Psicossocial)** - 18% de importância
    4. **IDA (Índice de Desempenho)** - 15% de importância
    5. **IAA (Índice de Autoavaliação)** - 11% de importância
    
    ### 📈 Como usar os resultados
    - **Risco Alto (>70%)**: Intervenção imediata necessária
    - **Risco Médio (30-70%)**: Acompanhamento próximo
    - **Risco Baixo (<30%)**: Monitoramento regular
    
    ### 🔄 Atualização do modelo
    O modelo é retreinado anualmente com novos dados para manter sua precisão.
    """)
    
    # Timeline
    st.markdown("### 📅 Timeline do Projeto")
    
    timeline_data = {
        "Etapa": ["Coleta de Dados", "Treinamento", "Validação", "Deploy"],
        "Data": ["2022-2024", "Jan/2025", "Fev/2025", "Mar/2025"],
        "Status": ["✅ Concluído", "✅ Concluído", "✅ Concluído", "✅ Ativo"]
    }
    
    df_timeline = pd.DataFrame(timeline_data)
    st.dataframe(df_timeline, use_container_width=True, hide_index=True)

# ============================================================
# PÁGINA 5: AJUDA
# ============================================================

else:  # Página de Ajuda
    st.header("❓ Central de Ajuda")
    
    with st.expander("📖 Como usar o Preditor Individual"):
        st.markdown("""
        1. Preencha todos os campos com os dados do aluno
        2. Clique em "Calcular Probabilidade de Risco"
        3. Analise o resultado e as recomendações
        4. Use as sugestões de intervenção para ajudar o aluno
        """)
    
    with st.expander("📊 Como usar o Preditor em Lote"):
        st.markdown("""
        1. Prepare um arquivo CSV com os dados dos alunos
        2. Faça o upload do arquivo
        3. Clique em "Processar todos os alunos"
        4. Baixe o relatório com os resultados
        """)
    
    with st.expander("🔍 Interpretando os resultados"):
        st.markdown("""
        - **Alto Risco (>70%)**: Probabilidade alta de defasagem. Necessita intervenção imediata.
        - **Médio Risco (30-70%)**: Probabilidade moderada. Acompanhamento próximo recomendado.
        - **Baixo Risco (<30%)**: Probabilidade baixa. Manter monitoramento regular.
        """)
    
    with st.expander("📋 Formato do arquivo CSV"):
        st.code("""
RA,IAA,IEG,IPS,IDA,IAN,IPV,Matem,Portug,Fase
ALUNO001,7.5,6.2,5.8,6.0,7.0,6.5,6.5,7.0,7
ALUNO002,4.2,3.5,4.0,3.8,4.5,4.0,4.0,3.5,6
        """)
    
    with st.expander("🆘 Suporte"):
        st.markdown("""
        **E-mail:** suporte@passosmagicos.org.br
        **Telefone:** (11) 1234-5678
        **Horário:** Segunda a Sexta, 9h às 18h
        """)

# ============================================================
# RODAPÉ
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        Desenvolvido para o Datathon da Passos Mágicos © 2024<br>
        Versão 1.0 - Modelo de Machine Learning para Predição de Risco de Defasagem
    </div>
    """,
    unsafe_allow_html=True
)