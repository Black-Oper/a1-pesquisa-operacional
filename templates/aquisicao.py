import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_folium import folium_static
from data.loader import load_real_data
from visualization.maps import create_route_map
from datetime import datetime

def pagina_aquisicao_preparo():
    st.markdown('<div class="main-header">AQUISIÇÃO E PREPARO DE DADOS - Pesquisa Operacional</div>', unsafe_allow_html=True)
    
    df = load_real_data()
    
    if df is None:
        st.error("Não foi possível carregar os dados. Verifique se o arquivo está na pasta correta.")
        return
    
    st.markdown('<div class="section-header">📊 Contexto, Origem e Problema</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>🏙️ Contexto - Logística Urbana em Curitiba</h4>
        <p>Este projeto analisa dados reais de pontos de coleta de lixo na cidade de <strong>Curitiba</strong> 
        para otimizar as rotas dos veículos de coleta. A eficiência nesse processo impacta diretamente:</p>
        <ul>
            <li>💰 Custos operacionais (combustível, manutenção, tempo)</li>
            <li>🌱 Impacto ambiental (emissões de CO₂)</li>
            <li>🎯 Qualidade do serviço público de limpeza urbana</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-label">🎯 OBJETIVO PRINCIPAL</div>
        <div style="font-size: 1rem; color: #e0e0e0;">
        Minimizar a distância total percorrida pelos veículos de coleta, atendendo a todas as restrições operacionais e janelas de tempo.
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>📁 Origem dos Dados - Dataset Real</h4>
    <p>Dataset real contendo <strong>201 pontos de coleta</strong> em Curitiba, incluindo o depósito central no CIC.</p>
    <p><strong>Estrutura do dataset:</strong> 201 registros × 9 colunas com informações completas de localização, demanda, tempo de serviço e restrições operacionais.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📋 Estrutura do Dataset")
    colunas_info = pd.DataFrame({
        'Coluna': ['id_ponto', 'bairro', 'latitude', 'longitude', 'demanda_kg', 
                   'tempo_servico_min', 'janela_inicio', 'janela_fim', 'prioridade'],
        'Descrição': [
            'Identificador único (0 = depósito)',
            'Bairro de localização',
            'Coordenada geográfica - latitude',
            'Coordenada geográfica - longitude',
            'Quantidade de resíduos (kg)',
            'Tempo de serviço necessário (minutos)',
            'Início da janela de tempo para coleta',
            'Fim da janela de tempo para coleta',
            'Prioridade (1-3, 3 = mais urgente)'
        ],
        'Tipo': ['int', 'string', 'float', 'float', 'int', 'int', 'time', 'time', 'int']
    })
    st.dataframe(colunas_info, use_container_width=True)
    
    st.markdown('<div class="section-header">📈 Métricas e Estatísticas do Dataset</div>', unsafe_allow_html=True)
    
    total_pontos = len(df) - 1
    demanda_total = df['demanda_kg'].sum()
    bairros_unicos = df['bairro'].nunique() - 1
    tempo_total_servico = df['tempo_servico_min'].sum()
    prioridade_alta = len(df[df['prioridade'] == 3])
    capacidade_minima = demanda_total / 5
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📍 PONTOS DE COLETA</div>
            <div class="metric-value">{total_pontos}</div>
            <div style="font-size: 0.8rem; color: #90caf9;">Total excluindo depósito</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚖️ DEMANDA TOTAL</div>
            <div class="metric-value">{demanda_total:,.0f} kg</div>
            <div style="font-size: 0.8rem; color: #90caf9;">Peso total a ser coletado</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🏘️ BAIRROS</div>
            <div class="metric-value">{bairros_unicos}</div>
            <div style="font-size: 0.8rem; color: #90caf9;">Bairros atendidos</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⏱️ TEMPO TOTAL</div>
            <div class="metric-value">{tempo_total_servico} min</div>
            <div style="font-size: 0.8rem; color: #90caf9;">Tempo de serviço total</div>
        </div>
        """, unsafe_allow_html=True)
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🚨 PRIORIDADE ALTA</div>
            <div class="metric-value">{prioridade_alta}</div>
            <div style="font-size: 0.8rem; color: #90caf9;">Pontos com prioridade 3</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        demanda_media = df[df['id_ponto'] > 0]['demanda_kg'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 DEMANDA MÉDIA</div>
            <div class="metric-value">{demanda_media:.0f} kg</div>
            <div style="font-size: 0.8rem; color: #90caf9;">Por ponto de coleta</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col7:
        tempo_medio = df[df['id_ponto'] > 0]['tempo_servico_min'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⏰ TEMPO MÉDIO</div>
            <div class="metric-value">{tempo_medio:.1f} min</div>
            <div style="font-size: 0.8rem; color: #90caf9;">Por ponto de coleta</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col8:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🚛 CAPACIDADE MÍNIMA</div>
            <div class="metric-value">{capacidade_minima:.0f} kg</div>
            <div style="font-size: 0.8rem; color: #90caf9;">Por veículo (estimado)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">📋 Tabela Filtrável do Dataset</div>', unsafe_allow_html=True)
    
    df_display = df.copy()
    
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        bairros_disponiveis = ['Todos'] + sorted(df_display['bairro'].unique().tolist())
        bairro_selecionado = st.selectbox('🏘️ Filtrar por Bairro:', bairros_disponiveis)
    
    with col_filter2:
        prioridades_disponiveis = ['Todas', '1 - Baixa', '2 - Média', '3 - Alta']
        prioridade_selecionada = st.selectbox('🎯 Filtrar por Prioridade:', prioridades_disponiveis)
    
    with col_filter3:
        demanda_min = int(df_display['demanda_kg'].min())
        demanda_max = int(df_display['demanda_kg'].max())
        demanda_range = st.slider('⚖️ Filtrar por Demanda (kg):', 
                                  demanda_min, demanda_max, 
                                  (demanda_min, demanda_max))
    
    df_filtrado = df_display.copy()
    
    if bairro_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['bairro'] == bairro_selecionado]
    
    if prioridade_selecionada != 'Todas':
        prioridade_valor = int(prioridade_selecionada[0])
        df_filtrado = df_filtrado[df_filtrado['prioridade'] == prioridade_valor]
    
    df_filtrado = df_filtrado[(df_filtrado['demanda_kg'] >= demanda_range[0]) & 
                              (df_filtrado['demanda_kg'] <= demanda_range[1])]
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("📍 Pontos Filtrados", len(df_filtrado))
    
    with col_stat2:
        st.metric("⚖️ Demanda Total", f"{df_filtrado['demanda_kg'].sum():,.0f} kg")
    
    with col_stat3:
        st.metric("⏱️ Tempo Total", f"{df_filtrado['tempo_servico_min'].sum()} min")
    
    with col_stat4:
        if len(df_filtrado) > 0:
            st.metric("📊 Demanda Média", f"{df_filtrado['demanda_kg'].mean():.0f} kg")
        else:
            st.metric("📊 Demanda Média", "N/A")
    
    df_tabela = df_filtrado[[
        'id_ponto', 'bairro', 'latitude', 'longitude', 
        'demanda_kg', 'tempo_servico_min', 'janela_inicio', 
        'janela_fim', 'prioridade'
    ]].copy()
    
    df_tabela.columns = [
        'ID', 'Bairro', 'Latitude', 'Longitude', 
        'Demanda (kg)', 'Tempo Serviço (min)', 'Janela Início', 
        'Janela Fim', 'Prioridade'
    ]
    
    df_tabela['Prioridade'] = df_tabela['Prioridade'].map({
        1: '1 - Baixa',
        2: '2 - Média',
        3: '3 - Alta'
    })
    
    st.dataframe(
        df_tabela,
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    csv = df_tabela.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download dos Dados Filtrados (CSV)",
        data=csv,
        file_name=f'dados_filtrados_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv',
    )
    
    st.markdown('<div class="section-header">📊 Visualizações dos Dados</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏘️ Distribuição por Bairro (Top 10)")
        bairros_count = df[df['bairro'] != 'CIC (Depósito)']['bairro'].value_counts().head(10)
        fig_bairros = px.bar(
            bairros_count, 
            x=bairros_count.values, 
            y=bairros_count.index,
            orientation='h',
            labels={'x': 'Número de Pontos', 'y': 'Bairro'},
            color=bairros_count.values,
            color_continuous_scale='blues'
        )
        fig_bairros.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white')
        st.plotly_chart(fig_bairros, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Distribuição de Prioridades")
        prioridade_count = df[df['id_ponto'] > 0]['prioridade'].value_counts().sort_index()
        fig_prioridade = px.pie(
            prioridade_count, 
            values=prioridade_count.values, 
            names=['Baixa', 'Média', 'Alta'],
            title="Distribuição por Nível de Prioridade",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig_prioridade.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white')
        st.plotly_chart(fig_prioridade, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("⚖️ Distribuição de Demanda")
        fig_demanda = px.histogram(
            df[df['id_ponto'] > 0], 
            x='demanda_kg',
            nbins=20,
            title="Distribuição da Demanda por Ponto",
            labels={'demanda_kg': 'Demanda (kg)'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_demanda.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white')
        st.plotly_chart(fig_demanda, use_container_width=True)
    
    with col4:
        st.subheader("⏱️ Distribuição do Tempo de Serviço")
        fig_tempo = px.box(
            df[df['id_ponto'] > 0], 
            y='tempo_servico_min',
            title="Distribuição do Tempo de Serviço",
            labels={'tempo_servico_min': 'Tempo de Serviço (min)'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_tempo.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white')
        st.plotly_chart(fig_tempo, use_container_width=True)
    
    st.subheader("🗺️ Mapa de Localização dos Pontos de Coleta - Curitiba")
    
    mapa = create_route_map(df)
    folium_static(mapa, width=1200, height=500)
    
    st.markdown("""
    <div class="info-box">
    <h4>🎯 Problema a ser Tratado - VRPTW</h4>
    <p>Os dados descrevem naturalmente um <strong>Problema de Roteamento de Veículos com Janelas de Tempo 
    (Vehicle Routing Problem with Time Windows - VRPTW)</strong>.</p>
    
    <h5>🔍 Características do VRPTW em Curitiba:</h5>
    <ul>
        <li><strong>Depósito único</strong> no CIC (Centro Industrial de Curitiba)</li>
        <li><strong>200 pontos de coleta</strong> distribuídos por diversos bairros</li>
        <li><strong>Janelas de tempo específicas</strong> para cada ponto</li>
        <li><strong>Demandas variadas</strong> de 853kg a 3.480kg por ponto</li>
        <li><strong>Prioridades diferenciadas</strong> para atendimento</li>
        <li><strong>Restrições de capacidade</strong> dos caminhões</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🔍 Mapeamento para um Problema de Otimização (Branch and Bound)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <p>O objetivo é determinar um conjunto de rotas de custo mínimo (distância ou tempo) para uma frota de veículos 
    (caminhões de lixo), de forma que:</p>
    
    <ol>
        <li><strong>Cada rota comece e termine no depósito CIC (Ponto 0)</strong></li>
        <li><strong>Todos os 200 pontos de coleta sejam visitados exatamente uma vez</strong></li>
        <li><strong>A demanda total de uma rota não exceda a capacidade do caminhão</strong></li>
        <li><strong>O serviço em cada ponto seja realizado dentro da janela de tempo especificada</strong></li>
        <li><strong>Pontos de prioridade mais alta sejam atendidos preferencialmente</strong></li>
    </ol>
    
    <p>O <strong>Branch and Bound (B&B)</strong> é um algoritmo de solução exata para problemas de otimização 
    combinatória NP-difíceis como o VRPTW. Ele explora sistematicamente o espaço de soluções através de 
    ramificação (branch) e poda (bound) de subproblemas.</p>
    </div>
    """, unsafe_allow_html=True)