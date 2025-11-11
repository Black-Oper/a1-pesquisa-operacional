import streamlit as st
import pandas as pd
import time
import unittest
from io import StringIO
from data.loader import load_real_data, time_to_minutes
from models.solver import VRPTWSolver
from tests.test_solver import TestVRPTWSolver
from visualization.tree import create_tree_visualization
import plotly.graph_objects as go

def pagina_implementacao_algoritmo():
    st.markdown('<div class="main-header">IMPLEMENTAÇÃO DO BRANCH AND BOUND</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>📁 Seleção da Base de Dados</h4>
    <p>Escolha entre a base de dados padrão ou faça upload de sua própria base de dados (formato CSV).</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_data1, col_data2 = st.columns([1, 1])
    
    with col_data1:
        data_source = st.radio(
            "Selecione a origem dos dados:",
            ["Base de Dados Padrão (Curitiba)", "Upload de Arquivo Personalizado"],
            help="Escolha a base de dados padrão ou faça upload de seu próprio arquivo CSV"
        )
    
    with col_data2:
        st.markdown("#### 📥 Template de Exemplo")
        template_csv = """id_ponto,bairro,latitude,longitude,demanda_kg,tempo_servico_min,janela_inicio,janela_fim,prioridade
0,Depósito Central,-25.4500,-49.3000,0,0,00:00,23:59,0
1,Bairro A,-25.4450,-49.2950,1200,30,06:00,12:00,1
2,Bairro B,-25.4400,-49.2900,1500,35,06:30,13:00,2
3,Bairro C,-25.4350,-49.2850,2000,40,07:00,14:00,3
4,Bairro D,-25.4300,-49.2800,1800,35,07:30,15:00,2
5,Bairro E,-25.4250,-49.2750,1600,30,08:00,16:00,1"""
        
        st.download_button(
            label="⬇️ Baixar Template CSV",
            data=template_csv,
            file_name="template_base_dados.csv",
            mime="text/csv",
            help="Baixe este template para criar sua própria base de dados"
        )
    
    df = None
    
    if data_source == "Base de Dados Padrão (Curitiba)":
        df = load_real_data()
        if df is None:
            st.error("Não foi possível carregar os dados padrão.")
            return
        st.success(f"✅ Base de dados padrão carregada: {len(df)} pontos de coleta")
    
    else:
        with col_data2:
            uploaded_file = st.file_uploader(
                "Faça upload do arquivo CSV",
                type=['csv'],
                help="O arquivo deve conter as colunas: id_ponto, bairro, latitude, longitude, demanda_kg, tempo_servico_min, janela_inicio, janela_fim, prioridade"
            )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                required_columns = ['id_ponto', 'latitude', 'longitude', 'demanda_kg', 
                                  'tempo_servico_min', 'janela_inicio', 'janela_fim']
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Colunas obrigatórias faltando: {', '.join(missing_columns)}")
                    st.info("""
                    **Colunas obrigatórias:**
                    - id_ponto (int): ID do ponto (0 = depósito)
                    - latitude (float): Coordenada de latitude
                    - longitude (float): Coordenada de longitude
                    - demanda_kg (int/float): Demanda em kg
                    - tempo_servico_min (int): Tempo de serviço em minutos
                    - janela_inicio (string): Horário de início (formato HH:MM)
                    - janela_fim (string): Horário de fim (formato HH:MM)
                    - prioridade (int, opcional): Nível de prioridade
                    - bairro (string, opcional): Nome do bairro
                    """)
                    return
                
                df['janela_inicio_min'] = df['janela_inicio'].apply(time_to_minutes)
                df['janela_fim_min'] = df['janela_fim'].apply(time_to_minutes)
                
                if 'prioridade' not in df.columns:
                    df['prioridade'] = 1
                if 'bairro' not in df.columns:
                    df['bairro'] = 'Não especificado'
                
                st.success(f"✅ Arquivo carregado com sucesso: {len(df)} pontos de coleta")
                
                with st.expander("📋 Visualizar Preview dos Dados"):
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total de Pontos", len(df))
                    with col2:
                        st.metric("Demanda Total", f"{df['demanda_kg'].sum():,.0f} kg")
                    with col3:
                        st.metric("Tempo Total", f"{df['tempo_servico_min'].sum()} min")
                    with col4:
                        if 'prioridade' in df.columns:
                            st.metric("Prioridade Alta", len(df[df['prioridade'] == 3]))
                
            except Exception as e:
                st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
                return
        else:
            st.info("⬆️ Faça upload de um arquivo CSV para continuar")
            return
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
    <h4>🎯 Configuração do Algoritmo Branch and Bound</h4>
    <p>Configure os parâmetros de execução do algoritmo e acompanhe em tempo real o progresso da otimização.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vehicle_capacity = st.slider("Capacidade do Veículo (kg)", 1000, 10000, 5000, 100)
        st.info("ℹ️ Número de veículos: Ilimitado (determinado automaticamente)")
    
    with col2:
        time_limit = st.slider("Tempo Limite (segundos)", 10, 600, 60, 10)
        search_strategy = st.selectbox("Estratégia de Busca", ["best-first", "depth-first"])
    
    with col3:
        max_nodes = st.number_input("Número Máximo de Nós", 100, 10000, 1000)
        run_tests = st.checkbox("Executar Testes Unitários", value=True)
    
    max_vehicles = 50
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 Executar Branch and Bound", type="primary", use_container_width=True):
            with st.spinner("Executando algoritmo Branch and Bound..."):
                solver = VRPTWSolver(df, vehicle_capacity, max_vehicles, time_limit)
                
                start_time = time.time()
                solution, metrics = solver.solve(search_strategy)
                execution_time = time.time() - start_time
                
                greedy_routes, greedy_cost = solver._greedy_heuristic()
                
                st.session_state.solver = solver
                st.session_state.solution = solution
                st.session_state.metrics = metrics
                st.session_state.greedy_solution = greedy_routes
                st.session_state.greedy_cost = greedy_cost
                st.session_state.execution_time = execution_time
    
    with col2:
        if st.button("🧪 Executar Testes Unitários", use_container_width=True) and run_tests:
            with st.spinner("Executando testes unitários..."):
                test_output = StringIO()
                runner = unittest.TextTestRunner(stream=test_output, verbosity=2)
                suite = unittest.TestLoader().loadTestsFromTestCase(TestVRPTWSolver)
                result = runner.run(suite)
                
                st.session_state.test_results = result
                st.session_state.test_output = test_output.getvalue()
    
    if 'metrics' in st.session_state:
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Resultados da Execução</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nós Expandidos", st.session_state.metrics['nodes_expanded'])
        with col2:
            st.metric("Profundidade Máxima", st.session_state.metrics['max_depth'])
        with col3:
            st.metric("Tempo Execução", f"{st.session_state.metrics['execution_time']:.2f}s")
        with col4:
            st.metric("Soluções Encontradas", st.session_state.metrics['solutions_found'])
        
        st.markdown("### 🔄 Comparação com Heurística Gulosa")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <div class="metric-label">BRANCH AND BOUND</div>
            <div class="metric-value">{:.2f} km</div>
            <div style="font-size: 0.8rem; color: #90caf9;">Custo total da rota</div>
            </div>
            """.format(st.session_state.metrics['best_cost']), unsafe_allow_html=True)
            
            st.write("**Rotas Otimizadas:**")
            for i, route in enumerate(st.session_state.solution):
                st.write(f"Veículo {i+1}: {' → '.join(map(str, route))}")
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <div class="metric-label">HEURÍSTICA GULOSA</div>
            <div class="metric-value">{:.2f} km</div>
            <div style="font-size: 0.8rem; color: #90caf9;">Custo total da rota</div>
            </div>
            """.format(st.session_state.greedy_cost), unsafe_allow_html=True)
            
            improvement = ((st.session_state.greedy_cost - st.session_state.metrics['best_cost']) / st.session_state.greedy_cost) * 100
            st.markdown(f"""
            <div class="success-box">
            <h4>🎯 Melhoria do Branch and Bound</h4>
            <div style="font-size: 1.5rem; font-weight: bold; text-align: center;">
            {improvement:.1f}% melhor
            </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="section-header">🌳 Visualização da Árvore de Busca</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4>📊 Estrutura da Árvore Branch and Bound</h4>
        <p>Esta visualização mostra a árvore de busca explorada pelo algoritmo Branch and Bound. 
        Cada nó representa um estado da busca, e as cores indicam o tipo de nó:</p>
        <ul>
        <li><strong style="color: #1f77b4;">Azul:</strong> Nó raiz (estado inicial)</li>
        <li><strong style="color: lightblue;">Azul claro:</strong> Nós explorados durante a busca</li>
        <li><strong style="color: green;">Verde:</strong> Nós que representam soluções completas</li>
        <li><strong style="color: red;">Vermelho:</strong> Nós podados (bound pior que melhor solução)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if 'solver' in st.session_state and st.session_state.solver.tree_nodes:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("### ⚙️ Configurações")
                show_stats = st.checkbox("Mostrar Estatísticas", value=True)
                
            with col1:
                fig = create_tree_visualization(st.session_state.solver)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Árvore vazia ou não disponível para visualização.")
            
            if show_stats:
                st.markdown("### 📈 Estatísticas da Árvore")
                
                solver = st.session_state.solver
                total_nodes = len(solver.tree_nodes)
                solution_nodes = sum(1 for n in solver.tree_nodes if n.is_solution)
                pruned_nodes = sum(1 for n in solver.tree_nodes if n.is_pruned)
                explored_nodes = total_nodes - solution_nodes - pruned_nodes
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total de Nós", total_nodes)
                with col2:
                    st.metric("Nós Explorados", explored_nodes)
                with col3:
                    st.metric("Soluções", solution_nodes)
                with col4:
                    st.metric("Nós Podados", pruned_nodes)
                
                if total_nodes > 0:
                    st.markdown("### 📊 Distribuição de Nós")
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Explorados', 'Soluções', 'Podados'],
                        values=[explored_nodes, solution_nodes, pruned_nodes],
                        marker=dict(colors=['lightblue', 'green', 'red']),
                        hole=0.3
                    )])
                    
                    fig_pie.update_layout(
                        plot_bgcolor='#1e1e1e',
                        paper_bgcolor='#1e1e1e',
                        font=dict(color='white'),
                        height=400
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                st.markdown("### 📊 Distribuição por Nível")
                
                level_data = {}
                for node in solver.tree_nodes:
                    level = node.level
                    if level not in level_data:
                        level_data[level] = {'total': 0, 'pruned': 0, 'solutions': 0}
                    level_data[level]['total'] += 1
                    if node.is_pruned:
                        level_data[level]['pruned'] += 1
                    if node.is_solution:
                        level_data[level]['solutions'] += 1
                
                if level_data:
                    levels = sorted(level_data.keys())
                    totals = [level_data[l]['total'] for l in levels]
                    pruned = [level_data[l]['pruned'] for l in levels]
                    solutions = [level_data[l]['solutions'] for l in levels]
                    
                    fig_levels = go.Figure()
                    fig_levels.add_trace(go.Bar(x=levels, y=totals, name='Total', marker_color='lightblue'))
                    fig_levels.add_trace(go.Bar(x=levels, y=pruned, name='Podados', marker_color='red'))
                    fig_levels.add_trace(go.Bar(x=levels, y=solutions, name='Soluções', marker_color='green'))
                    
                    fig_levels.update_layout(
                        barmode='group',
                        xaxis_title='Nível',
                        yaxis_title='Número de Nós',
                        plot_bgcolor='#1e1e1e',
                        paper_bgcolor='#1e1e1e',
                        font=dict(color='white'),
                        height=400
                    )
                    
                    st.plotly_chart(fig_levels, use_container_width=True)
        else:
            st.info("⚠️ Execute o algoritmo para visualizar a árvore de busca.")
    
    if 'test_output' in st.session_state:
        st.markdown("---")
        st.markdown('<div class="section-header">🧪 Resultados dos Testes Unitários</div>', unsafe_allow_html=True)
        
        st.code(st.session_state.test_output, language='text')
        
        if st.session_state.test_results.wasSuccessful():
            st.success("✅ Todos os testes passaram!")
        else:
            st.error("❌ Alguns testes falharam.")