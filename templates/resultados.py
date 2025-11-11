import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_folium import folium_static
from data.loader import load_real_data
from models.solver import VRPTWSolver
from visualization.maps import create_route_map
from visualization.tree import create_tree_visualization

def pagina_resultados_analise():
    st.markdown('<div class="main-header">RESULTADOS E ANÁLISE</div>', unsafe_allow_html=True)
    
    if 'metrics' not in st.session_state:
        st.info("ℹ️ Execute o algoritmo na página de implementação para ver os resultados.")
        return
    
    df = load_real_data()
    if df is None:
        st.error("Não foi possível carregar os dados.")
        return
    
    st.markdown("""
    <div class="info-box">
    <h4>📈 Análise de Sensibilidade e Robustez</h4>
    <p>Analise o impacto dos parâmetros na qualidade da solução e na performance do algoritmo.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Análise de Sensibilidade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        capacity_range = st.slider("Faixa de Capacidade para Análise", 1000, 10000, (3000, 7000), 500)
        if st.button("Analisar Sensibilidade à Capacidade"):
            with st.spinner("Analisando sensibilidade..."):
                capacities = range(capacity_range[0], capacity_range[1] + 500, 500)
                costs = []
                
                for capacity in capacities:
                    solver = VRPTWSolver(df, capacity, 5, 30)
                    solution, metrics = solver.solve()
                    costs.append(metrics['best_cost'])
                
                fig = px.line(x=capacities, y=costs, 
                            title="Sensibilidade à Capacidade do Veículo",
                            labels={'x': 'Capacidade (kg)', 'y': 'Custo Total (km)'})
                fig.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white')
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Métricas de Performance")
        
        metrics_data = {
            'Metrica': ['Nós Expandidos', 'Profundidade Máxima', 'Tempo Execução', 
                        'Soluções Encontradas', 'Nós Podados', 'Custo Total'],
            'Valor': [
                st.session_state.metrics['nodes_expanded'],
                st.session_state.metrics['max_depth'],
                f"{st.session_state.metrics['execution_time']:.2f}s",
                st.session_state.metrics['solutions_found'],
                st.session_state.metrics['pruned_nodes'],
                f"{st.session_state.metrics['best_cost']:.2f} km"
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.markdown("### 🗺️ Visualização das Rotas Otimizadas")
    
    if 'solution' in st.session_state:
        mapa = create_route_map(df, st.session_state.solution)
        folium_static(mapa, width=1200, height=500)
    
    st.markdown("---")
    st.markdown("### 🌳 Árvore de Busca Branch and Bound")
    
    if 'solver' in st.session_state and st.session_state.solver.tree_nodes:
        tab1, tab2 = st.tabs(["Visualização da Árvore", "Análise Detalhada"])
        
        with tab1:
            fig = create_tree_visualization(st.session_state.solver)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Árvore vazia ou não disponível para visualização.")
        
        with tab2:
            solver = st.session_state.solver
            
            st.markdown("#### 📋 Informações Detalhadas")
            
            node_data = []
            for node in solver.tree_nodes[:50]:
                route_str = '→'.join(map(str, node.get_current_route()))
                node_data.append({
                    'ID': node.node_id,
                    'Nível': node.level,
                    'Custo': f"{node.cost:.2f}",
                    'Bound': f"{node.bound:.2f}",
                    'Rota Atual': route_str,
                    'Visitados': f"{len(node.visited)}/{solver.n_points-1}",
                    'Status': 'Solução' if node.is_solution else ('Podado' if node.is_pruned else 'Explorado')
                })
            
            if node_data:
                df_nodes = pd.DataFrame(node_data)
                st.dataframe(df_nodes, use_container_width=True, hide_index=True)
                
                if len(solver.tree_nodes) > 50:
                    st.caption(f"Mostrando 50 de {len(solver.tree_nodes)} nós. A árvore completa está na visualização acima.")
    else:
        st.info("⚠️ A árvore de busca não está disponível. Execute o algoritmo novamente para visualizar.")