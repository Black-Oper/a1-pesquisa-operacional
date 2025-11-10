import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, atan2
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import heapq
import random
from typing import List, Tuple, Dict, Optional
import unittest
import sys
from io import StringIO

# Configuração da página
st.set_page_config(
    page_title="Otimização de Rotas - Coleta de Lixo Curitiba",
    page_icon="🗑️",
    layout="wide"
)

# CSS personalizado com tema escuro para cards
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #2b2b2b;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #2d2d2d 0%, #404040 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin: 0.8rem 0;
        color: #ffffff;
        border: 1px solid #404040;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4fc3f7;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #b0b0b0;
        margin-bottom: 0.5rem;
    }
    .stDataFrame {
        background-color: #1e1e1e;
        border-radius: 0.5rem;
    }
    /* Estilo para a tabela de budget em markdown */
    .stMarkdown table {
        width: 100%;
        background-color: #2d2d2d;
        color: #ffffff;
        border-radius: 0.5rem;
    }
    .stMarkdown th {
        background-color: #3a3a3a;
        color: #4fc3f7;
        padding: 0.5rem;
    }
    .stMarkdown td {
        padding: 0.5rem;
        border-bottom: 1px solid #444;
    }
    .math-formula {
        background-color: #1a1a1a;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        color: #ffffff;
    }
    .algorithm-box {
        background-color: #2d2d2d;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #444;
        margin: 1rem 0;
        color: #ffffff;
    }
    .success-box {
        background-color: #1b5e20;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #ffffff;
    }
    .warning-box {
        background-color: #ff6f00;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #ffffff;
    }
    .node-box {
        background-color: #333333;
        padding: 0.5rem;
        margin: 0.2rem;
        border-radius: 0.3rem;
        border: 1px solid #555;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Carregar dados reais
@st.cache_data
def load_real_data():
    """Carrega os dados reais do arquivo CSV"""
    try:
        df = pd.read_csv('rota_coleta_curitiba (1).csv')
        
        # Converter horários para minutos desde meia-noite
        def time_to_minutes(time_str):
            try:
                if ':' in time_str:
                    hours, minutes = map(int, time_str.split(':'))
                    return hours * 60 + minutes
                else:
                    return 0
            except:
                return 0
        
        df['janela_inicio_min'] = df['janela_inicio'].apply(time_to_minutes)
        df['janela_fim_min'] = df['janela_fim'].apply(time_to_minutes)
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

# =============================================================================
# IMPLEMENTAÇÃO DO BRANCH AND BOUND
# =============================================================================

class VRPTWNode:
    """Nó da árvore de busca do Branch and Bound para VRPTW"""
    
    _node_counter = 0  # Contador global para IDs únicos
    
    def __init__(self, level=0, cost=0, bound=0, visited=None, routes=None, 
                 current_route_idx=0, vehicle_load=0, vehicle_time=0, parent=None):
        self.level = level  # Profundidade na árvore
        self.cost = cost    # Custo acumulado (distância total)
        self.bound = bound  # Limite inferior (bound)
        self.visited = visited or set()  # Pontos visitados globalmente
        self.routes = routes or [[0]]  # Lista de rotas (cada rota é uma lista de pontos)
        self.current_route_idx = current_route_idx  # Índice da rota atual
        self.vehicle_load = vehicle_load  # Carga atual do veículo na rota atual
        self.vehicle_time = vehicle_time  # Tempo atual do veículo na rota atual
        self.parent = parent  # Nó pai para visualização da árvore
        
        # Atribuir ID único
        VRPTWNode._node_counter += 1
        self.node_id = VRPTWNode._node_counter
        
        self.is_solution = False  # Indica se é uma solução completa
        self.is_pruned = False    # Indica se foi podado
        
    def __lt__(self, other):
        # Para fila de prioridade (menor bound primeiro)
        return self.bound < other.bound
    
    def get_current_route(self):
        """Retorna a rota atual"""
        return self.routes[self.current_route_idx]
    
    def get_last_point(self):
        """Retorna o último ponto da rota atual"""
        return self.get_current_route()[-1]
    
    @classmethod
    def reset_counter(cls):
        """Reset do contador de nós"""
        cls._node_counter = 0

class VRPTWSolver:
    """Solver Branch and Bound para VRPTW"""
    
    def __init__(self, df, vehicle_capacity=5000, max_vehicles=5, time_limit=300):
        self.df = df
        self.n_points = len(df)
        self.vehicle_capacity = vehicle_capacity
        self.max_vehicles = max_vehicles
        self.time_limit = time_limit
        self.KM_PARA_MINUTOS_FATOR = 2.0
        
        # Calcular matriz de distâncias
        self.dist_matrix = self._calculate_distance_matrix()
        
        # Métricas de execução
        self.nodes_expanded = 0
        self.max_depth = 0
        self.start_time = 0
        self.solutions_found = 0
        self.pruned_nodes = 0
        
        # Melhor solução encontrada
        self.best_cost = float('inf')
        self.best_solution = None
        
        # Armazenar estrutura da árvore para visualização
        self.tree_nodes = []  # Lista de todos os nós explorados
        self.max_tree_nodes = 500  # Limitar para não sobrecarregar a visualização
        
    def _calculate_distance_matrix(self):
        """Calcula matriz de distâncias usando Haversine"""
        n = len(self.df)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    lat1, lon1 = radians(self.df.iloc[i]['latitude']), radians(self.df.iloc[i]['longitude'])
                    lat2, lon2 = radians(self.df.iloc[j]['latitude']), radians(self.df.iloc[j]['longitude'])
                    
                    dlon = lon2 - lon1
                    dlat = lat2 - lat1
                    
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    dist_matrix[i][j] = 6371 * c  # km
                    
        return dist_matrix
    
    def _calculate_bound(self, node):
        """Calcula o limite inferior para um nó usando MST dos pontos não visitados"""
        bound = node.cost
        
        # Para nós não completos, adicionar estimativa para visitar nós restantes
        if len(node.visited) < self.n_points - 1:  # -1 porque o depósito (0) não conta
            unvisited = set(range(1, self.n_points)) - node.visited
            
            if unvisited:
                last_point = node.get_last_point()
                
                # Custo mínimo de sair do último ponto para algum nó não visitado
                min_exit_cost = min([self.dist_matrix[last_point][j] for j in unvisited], default=0)
                bound += min_exit_cost
                
                # Estimativa MST (Minimum Spanning Tree) para pontos não visitados
                if len(unvisited) > 1:
                    mst_cost = 0
                    unvisited_list = list(unvisited)
                    for i in range(len(unvisited_list)):
                        min_edge = float('inf')
                        for j in range(len(unvisited_list)):
                            if i != j:
                                min_edge = min(min_edge, self.dist_matrix[unvisited_list[i]][unvisited_list[j]])
                        if min_edge < float('inf'):
                            mst_cost += min_edge
                    bound += mst_cost * 0.5  # Fator de aproximação do MST
                
                # Custo mínimo para retornar ao depósito de qualquer ponto não visitado
                min_return_cost = min([self.dist_matrix[j][0] for j in unvisited], default=0)
                bound += min_return_cost
        
        return bound
    
    def _calculate_travel_time(self, point_a_idx, point_b_idx):
        return self.dist_matrix[point_a_idx][point_b_idx] * self.KM_PARA_MINUTOS_FATOR
    
    def _is_feasible(self, node, next_point):
        """Verifica se adicionar next_point à rota atual é viável"""
        if next_point in node.visited:
            return False
            
        # Verificar capacidade
        demand = self.df.iloc[next_point]['demanda_kg']
        if node.vehicle_load + demand > self.vehicle_capacity:
            return False
            
        # Verificar janela de tempo
        last_point = node.get_last_point()
        travel_time = self._calculate_travel_time(last_point, next_point)
        arrival_time = node.vehicle_time + travel_time
        
        time_window_start = self.df.iloc[next_point]['janela_inicio_min']
        time_window_end = self.df.iloc[next_point]['janela_fim_min']
        
        # Chegou antes da janela? Pode esperar
        # Chegou depois do fim da janela? Inviável
        if arrival_time > time_window_end:
            return False
            
        return True
    
    def _can_start_new_route(self, node):
        """Verifica se é possível iniciar uma nova rota"""
        return len(node.routes) < self.max_vehicles
    
    def _update_best_solution(self, node):
        """Atualiza a melhor solução encontrada"""
        # Finalizar todas as rotas retornando ao depósito
        complete_routes = []
        total_cost = 0
        
        for route in node.routes:
            if len(route) > 1:  # Só adicionar se tem mais que o depósito
                # Adicionar retorno ao depósito se ainda não foi adicionado
                if route[-1] != 0:
                    complete_route = route + [0]
                    # Calcular custo da rota
                    route_cost = sum(self.dist_matrix[complete_route[i]][complete_route[i+1]] 
                                   for i in range(len(complete_route)-1))
                    total_cost += route_cost
                    complete_routes.append(complete_route)
                else:
                    complete_routes.append(route)
                    route_cost = sum(self.dist_matrix[route[i]][route[i+1]] 
                                   for i in range(len(route)-1))
                    total_cost += route_cost
        
        if total_cost < self.best_cost:
            self.best_cost = total_cost
            self.best_solution = complete_routes
            return True
        return False
    
    def _greedy_heuristic(self):
        """Heurística gulosa para solução inicial"""
        unvisited = set(range(1, self.n_points))
        routes = []
        current_route = [0]  # Começa no depósito
        current_load = 0
        current_time = 0
        
        while unvisited:
            best_point = None
            best_cost = float('inf')
            
            for point in unvisited:
                demand = self.df.iloc[point]['demanda_kg']
                travel_time = self._calculate_travel_time(current_route[-1], point)
                arrival_time = current_time + travel_time
                time_window_end = self.df.iloc[point]['janela_fim_min']
                
                if (current_load + demand <= self.vehicle_capacity and 
                    arrival_time <= time_window_end):
                    
                    cost = self.dist_matrix[current_route[-1]][point]
                    if cost < best_cost:
                        best_cost = cost
                        best_point = point
            
            if best_point is None:
                # Voltar ao depósito e começar nova rota
                if len(current_route) > 1:
                    current_route.append(0)
                    routes.append(current_route)
                current_route = [0]
                current_load = 0
                current_time = 0
            else:
                current_route.append(best_point)
                current_load += self.df.iloc[best_point]['demanda_kg']
                travel_time = self._calculate_travel_time(current_route[-2], best_point)
                arrival_time = current_time + travel_time
                service_start = max(arrival_time, self.df.iloc[best_point]['janela_inicio_min'])
                current_time = service_start + self.df.iloc[best_point]['tempo_servico_min']
                unvisited.remove(best_point)
        
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)
            
        total_cost = sum(self.dist_matrix[route[i]][route[i+1]] for route in routes for i in range(len(route)-1))
        return routes, total_cost
    
    def solve(self, search_strategy='best-first'):
        """Resolve o VRPTW usando Branch and Bound"""
        self.start_time = time.time()
        self.nodes_expanded = 0
        self.max_depth = 0
        self.solutions_found = 0
        self.pruned_nodes = 0
        self.best_cost = float('inf')
        self.best_solution = None
        self.tree_nodes = []  # Reset da árvore
        
        # Reset do contador de nós
        VRPTWNode.reset_counter()
        
        # Solução inicial com heurística gulosa
        best_routes, best_cost = self._greedy_heuristic()
        self.best_solution = best_routes
        self.best_cost = best_cost
        
        print(f"Solução inicial (gulosa): custo = {best_cost:.2f} km")
        
        # Nó raiz
        root = VRPTWNode()
        root.bound = self._calculate_bound(root)
        
        # Adicionar nó raiz à árvore
        if len(self.tree_nodes) < self.max_tree_nodes:
            self.tree_nodes.append(root)
        
        # Fila de prioridade
        queue = []
        if search_strategy == 'best-first':
            heapq.heappush(queue, root)
        else:  # DFS
            queue = [root]
        
        while queue and (time.time() - self.start_time) < self.time_limit:
            if search_strategy == 'best-first':
                node = heapq.heappop(queue)
            else:
                node = queue.pop()
                
            self.nodes_expanded += 1
            self.max_depth = max(self.max_depth, node.level)
            
            # Poda: se o bound já é pior que a melhor solução, não expandir
            if node.bound >= self.best_cost:
                self.pruned_nodes += 1
                node.is_pruned = True
                continue
            
            # Verificar se é solução completa (todos os pontos visitados)
            if len(node.visited) == self.n_points - 1:  # -1 porque o depósito não conta
                self.solutions_found += 1
                node.is_solution = True
                self._update_best_solution(node)
                continue
            
            # Expandir nó: considerar adicionar pontos à rota atual ou iniciar nova rota
            expanded = False
            
            # Opção 1: Adicionar próximo ponto à rota atual
            for next_point in range(1, self.n_points):
                if next_point not in node.visited and self._is_feasible(node, next_point):
                    expanded = True
                    
                    # Criar novo nó
                    new_visited = node.visited.copy()
                    new_visited.add(next_point)
                    
                    # Copiar rotas e adicionar ponto à rota atual
                    new_routes = [route.copy() for route in node.routes]
                    new_routes[node.current_route_idx].append(next_point)
                    
                    # Calcular tempo de viagem
                    last_point = node.get_last_point()
                    travel_time = self._calculate_travel_time(last_point, next_point)
                    arrival_time = node.vehicle_time + travel_time
                    
                    # Dados do próximo ponto
                    point_data = self.df.iloc[next_point]
                    time_window_start = point_data['janela_inicio_min']
                    service_time = point_data['tempo_servico_min']
                    
                    # Calcular tempo de início do serviço (pode haver espera)
                    service_start_time = max(arrival_time, time_window_start)
                    
                    # Novo tempo (após serviço)
                    new_time = service_start_time + service_time
                    
                    # Novo custo e carga
                    new_cost = node.cost + self.dist_matrix[last_point][next_point]
                    new_load = node.vehicle_load + point_data['demanda_kg']
                    
                    # Criar novo nó
                    new_node = VRPTWNode(
                        level=node.level + 1,
                        cost=new_cost,
                        visited=new_visited,
                        routes=new_routes,
                        current_route_idx=node.current_route_idx,
                        vehicle_load=new_load,
                        vehicle_time=new_time,
                        parent=node  # Adicionar referência ao pai
                    )
                    
                    new_node.bound = self._calculate_bound(new_node)
                    
                    # Só adicionar à fila se o bound for promissor
                    if new_node.bound < self.best_cost:
                        if search_strategy == 'best-first':
                            heapq.heappush(queue, new_node)
                        else:
                            queue.append(new_node)
                        
                        # Adicionar à árvore de visualização
                        if len(self.tree_nodes) < self.max_tree_nodes:
                            self.tree_nodes.append(new_node)
                    else:
                        self.pruned_nodes += 1
                        new_node.is_pruned = True
                        if len(self.tree_nodes) < self.max_tree_nodes:
                            self.tree_nodes.append(new_node)
            
            # Opção 2: Finalizar rota atual e iniciar nova (se houver pontos não visitados e veículos disponíveis)
            if len(node.visited) < self.n_points - 1 and self._can_start_new_route(node):
                # Copiar rotas
                new_routes = [route.copy() for route in node.routes]
                
                # Finalizar rota atual retornando ao depósito
                if new_routes[node.current_route_idx][-1] != 0:
                    last_point = new_routes[node.current_route_idx][-1]
                    new_routes[node.current_route_idx].append(0)
                    route_return_cost = self.dist_matrix[last_point][0]
                else:
                    route_return_cost = 0
                
                # Iniciar nova rota
                new_routes.append([0])
                new_route_idx = len(new_routes) - 1
                
                # Criar novo nó para nova rota
                new_node = VRPTWNode(
                    level=node.level + 1,
                    cost=node.cost + route_return_cost,
                    visited=node.visited.copy(),
                    routes=new_routes,
                    current_route_idx=new_route_idx,
                    vehicle_load=0,
                    vehicle_time=0,
                    parent=node  # Adicionar referência ao pai
                )
                
                new_node.bound = self._calculate_bound(new_node)
                
                if new_node.bound < self.best_cost:
                    if search_strategy == 'best-first':
                        heapq.heappush(queue, new_node)
                    else:
                        queue.append(new_node)
                    
                    # Adicionar à árvore de visualização
                    if len(self.tree_nodes) < self.max_tree_nodes:
                        self.tree_nodes.append(new_node)
                else:
                    self.pruned_nodes += 1
                    new_node.is_pruned = True
                    if len(self.tree_nodes) < self.max_tree_nodes:
                        self.tree_nodes.append(new_node)
            
            # Se não foi possível expandir de forma alguma, contar como poda
            if not expanded and not self._can_start_new_route(node):
                self.pruned_nodes += 1
        
        execution_time = time.time() - self.start_time
        
        # Calcular gap ótimo
        optimal_gap = 0
        if self.best_cost < float('inf') and best_cost > 0:
            optimal_gap = ((self.best_cost - best_cost) / best_cost) * 100
        
        metrics = {
            'nodes_expanded': self.nodes_expanded,
            'max_depth': self.max_depth,
            'execution_time': execution_time,
            'solutions_found': self.solutions_found,
            'pruned_nodes': self.pruned_nodes,
            'best_cost': self.best_cost,
            'optimal_gap': optimal_gap,
            'initial_heuristic_cost': best_cost
        }
        
        return self.best_solution, metrics

# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO DA ÁRVORE
# =============================================================================

def create_tree_visualization(solver):
    """Cria visualização da árvore de busca Branch and Bound usando Plotly"""
    
    if not solver.tree_nodes:
        return None
    
    # Preparar dados para visualização
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    edge_x = []
    edge_y = []
    
    # Calcular posições dos nós usando layout em níveis
    level_counts = {}
    level_positions = {}
    
    # Contar nós por nível
    for node in solver.tree_nodes:
        level = node.level
        if level not in level_counts:
            level_counts[level] = 0
            level_positions[level] = 0
        level_counts[level] += 1
    
    # Calcular posições
    node_positions = {}
    for node in solver.tree_nodes:
        level = node.level
        # Posição Y baseada no nível (invertido para raiz no topo)
        y = -level
        
        # Posição X distribuída uniformemente no nível
        total_in_level = level_counts[level]
        position_in_level = level_positions[level]
        level_positions[level] += 1
        
        # Espaçamento horizontal
        if total_in_level > 1:
            x = (position_in_level - (total_in_level - 1) / 2) * 2
        else:
            x = 0
        
        node_positions[node.node_id] = (x, y)
        
        # Adicionar coordenadas do nó
        node_x.append(x)
        node_y.append(y)
        
        # Texto do nó
        route_str = '→'.join(map(str, node.get_current_route()))
        text = f"ID: {node.node_id}<br>"
        text += f"Nível: {node.level}<br>"
        text += f"Custo: {node.cost:.2f}<br>"
        text += f"Bound: {node.bound:.2f}<br>"
        text += f"Rota: {route_str}<br>"
        text += f"Visitados: {len(node.visited)}/{solver.n_points-1}"
        node_text.append(text)
        
        # Cor do nó baseada no estado
        if node.is_solution:
            node_color.append('green')  # Solução
        elif node.is_pruned:
            node_color.append('red')     # Podado
        elif node.level == 0:
            node_color.append('blue')    # Raiz
        else:
            node_color.append('lightblue')  # Normal
        
        # Adicionar arestas para o pai
        if node.parent and node.parent.node_id in node_positions:
            parent_x, parent_y = node_positions[node.parent.node_id]
            edge_x.extend([parent_x, x, None])
            edge_y.extend([parent_y, y, None])
    
    # Criar trace das arestas
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Criar trace dos nós
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[str(n.node_id) for n in solver.tree_nodes],
        textposition="middle center",
        textfont=dict(size=8, color='white'),
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=20,
            line=dict(width=2, color='white')
        ),
        showlegend=False
    )
    
    # Criar figura
    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            # Mova o texto e a fonte para dentro da propriedade 'title'
            title=dict(
                text='Árvore de Busca Branch and Bound',
                font=dict(size=16) # Define o tamanho da fonte aqui
            ),
            # title='Árvore de Busca Branch and Bound', <-- Remova esta linha
            # titlefont_size=16,                     <-- Remova esta linha
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            font=dict(color='white'),
            height=600
        ))
    
    # Adicionar legendas
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='blue'),
        showlegend=True,
        name='Raiz'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='lightblue'),
        showlegend=True,
        name='Nó Explorado'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='green'),
        showlegend=True,
        name='Solução'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='red'),
        showlegend=True,
        name='Podado'
    ))
    
    return fig

# =============================================================================
# TESTES UNITÁRIOS
# =============================================================================

class TestVRPTWSolver(unittest.TestCase):
    """Testes unitários para o solver VRPTW"""
    
    def setUp(self):
        # Dados de teste simplificados
        data = {
            'id_ponto': [0, 1, 2],
            'latitude': [-25.5, -25.4, -25.45],
            'longitude': [-49.3, -49.2, -49.25],
            'demanda_kg': [0, 1000, 1500],
            'tempo_servico_min': [0, 30, 45],
            'janela_inicio_min': [0, 480, 540],
            'janela_fim_min': [1440, 1020, 1080],
            'prioridade': [0, 1, 2]
        }
        self.df = pd.DataFrame(data)
        self.solver = VRPTWSolver(self.df, vehicle_capacity=3000, max_vehicles=2)
    
    def test_distance_matrix(self):
        """Testa cálculo da matriz de distâncias"""
        dist_matrix = self.solver._calculate_distance_matrix()
        self.assertEqual(dist_matrix.shape, (3, 3))
        self.assertEqual(dist_matrix[0][0], 0)
        
    def test_bound_calculation(self):
        """Testa cálculo do bound"""
        node = VRPTWNode()
        bound = self.solver._calculate_bound(node)
        self.assertGreaterEqual(bound, 0)
        
    def test_feasibility_check(self):
        """Testa verificação de viabilidade"""
        node = VRPTWNode(visited=set(), vehicle_load=0, vehicle_time=0, routes=[[0]])
        feasible = self.solver._is_feasible(node, 1)
        self.assertTrue(feasible)

# =============================================================================
# PÁGINAS DO DASHBOARD
# =============================================================================

def pagina_aquisicao_preparo():
    st.markdown('<div class="main-header">AQUISIÇÃO E PREPARO DE DADOS - Pesquisa Operacional</div>', unsafe_allow_html=True)
    
    # Carregar dados
    df = load_real_data()
    
    if df is None:
        st.error("Não foi possível carregar os dados. Verifique se o arquivo está na pasta correta.")
        return
    
    # Contexto, Origem e Problema
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
    
    # Origem dos Dados
    st.markdown("""
    <div class="info-box">
    <h4>📁 Origem dos Dados - Dataset Real</h4>
    <p>Dataset real contendo <strong>201 pontos de coleta</strong> em Curitiba, incluindo o depósito central no CIC.</p>
    <p><strong>Estrutura do dataset:</strong> 201 registros × 9 colunas com informações completas de localização, demanda, tempo de serviço e restrições operacionais.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabela de descrição das colunas
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
    
    # Métricas principais com design escuro
    st.markdown('<div class="section-header">📈 Métricas e Estatísticas do Dataset</div>', unsafe_allow_html=True)
    
    # Calcular métricas
    total_pontos = len(df) - 1  # Excluindo depósito
    demanda_total = df['demanda_kg'].sum()
    bairros_unicos = df['bairro'].nunique() - 1  # Excluindo depósito
    tempo_total_servico = df['tempo_servico_min'].sum()
    prioridade_alta = len(df[df['prioridade'] == 3])
    capacidade_minima = demanda_total / 5  # Estimativa para 5 caminhões
    
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
    
    # Segunda linha de métricas
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
    
    # Tabela Filtrável do Dataset
    st.markdown('<div class="section-header">📋 Tabela Filtrável do Dataset</div>', unsafe_allow_html=True)
    
    # Criar cópias dos dados para exibição
    df_display = df.copy()
    
    # Criar filtros na barra lateral
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        # Filtro por bairro
        bairros_disponiveis = ['Todos'] + sorted(df_display['bairro'].unique().tolist())
        bairro_selecionado = st.selectbox('🏘️ Filtrar por Bairro:', bairros_disponiveis)
    
    with col_filter2:
        # Filtro por prioridade
        prioridades_disponiveis = ['Todas', '1 - Baixa', '2 - Média', '3 - Alta']
        prioridade_selecionada = st.selectbox('🎯 Filtrar por Prioridade:', prioridades_disponiveis)
    
    with col_filter3:
        # Filtro por demanda
        demanda_min = int(df_display['demanda_kg'].min())
        demanda_max = int(df_display['demanda_kg'].max())
        demanda_range = st.slider('⚖️ Filtrar por Demanda (kg):', 
                                  demanda_min, demanda_max, 
                                  (demanda_min, demanda_max))
    
    # Aplicar filtros
    df_filtrado = df_display.copy()
    
    if bairro_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['bairro'] == bairro_selecionado]
    
    if prioridade_selecionada != 'Todas':
        prioridade_valor = int(prioridade_selecionada[0])
        df_filtrado = df_filtrado[df_filtrado['prioridade'] == prioridade_valor]
    
    df_filtrado = df_filtrado[(df_filtrado['demanda_kg'] >= demanda_range[0]) & 
                              (df_filtrado['demanda_kg'] <= demanda_range[1])]
    
    # Exibir estatísticas dos dados filtrados
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
    
    # Preparar dados para exibição
    df_tabela = df_filtrado[[
        'id_ponto', 'bairro', 'latitude', 'longitude', 
        'demanda_kg', 'tempo_servico_min', 'janela_inicio', 
        'janela_fim', 'prioridade'
    ]].copy()
    
    # Renomear colunas para melhor visualização
    df_tabela.columns = [
        'ID', 'Bairro', 'Latitude', 'Longitude', 
        'Demanda (kg)', 'Tempo Serviço (min)', 'Janela Início', 
        'Janela Fim', 'Prioridade'
    ]
    
    # Adicionar label de prioridade
    df_tabela['Prioridade'] = df_tabela['Prioridade'].map({
        1: '1 - Baixa',
        2: '2 - Média',
        3: '3 - Alta'
    })
    
    # Exibir tabela com opção de download
    st.dataframe(
        df_tabela,
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # Botão de download dos dados filtrados
    csv = df_tabela.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download dos Dados Filtrados (CSV)",
        data=csv,
        file_name=f'dados_filtrados_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv',
    )
    
    # Visualizações
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
    
    # Gráficos adicionais
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
    
    # Mapa de localização
    st.subheader("🗺️ Mapa de Localização dos Pontos de Coleta - Curitiba")
    
    # Criar mapa centrado em Curitiba
    mapa = folium.Map(location=[-25.4284, -49.2733], zoom_start=11)
    
    # Adicionar pontos ao mapa
    for _, row in df.iterrows():
        if row['id_ponto'] == 0:
            # Depósito - cor diferente
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=f"🚛 DEPÓSITO CENTRAL CIC\nBairro: {row['bairro']}",
                tooltip="Depósito Central CIC",
                icon=folium.Icon(color='red', icon='home', prefix='fa')
            ).add_to(mapa)
        else:
            # Pontos de coleta - cor baseada na prioridade
            cores = {1: 'green', 2: 'orange', 3: 'red'}
            cor = cores.get(row['prioridade'], 'blue')
            
            folium.CircleMarker(
                [row['latitude'], row['longitude']],
                radius=6,
                popup=(
                    f"Ponto: {row['id_ponto']}<br>"
                    f"Bairro: {row['bairro']}<br>"
                    f"Demanda: {row['demanda_kg']}kg<br>"
                    f"Prioridade: {row['prioridade']}<br>"
                    f"Janela: {row['janela_inicio']} - {row['janela_fim']}"
                ),
                tooltip=f"Ponto {row['id_ponto']} - {row['bairro']}",
                color=cor,
                fill=True,
                fillColor=cor,
                fillOpacity=0.7
            ).add_to(mapa)
    
    folium_static(mapa, width=1200, height=500)
    
    # Problema a ser Tratado
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
    
    # Mapeamento para Branch and Bound
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

def pagina_modelagem_matematica():
    st.markdown('<div class="main-header">MODELAGEM MATEMÁTICA - VRPTW</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>🎯 Definição Formal do Problema VRPTW</h4>
    <p>O Problema de Roteamento de Veículos com Janelas de Tempo (VRPTW) é formalmente definido como um grafo 
    direcionado G = (V, A) onde:</p>
    <ul>
        <li><strong>V = {0, 1, 2, ..., n}</strong> é o conjunto de vértices (0 = depósito, 1...n = pontos de coleta)</li>
        <li><strong>A</strong> é o conjunto de arcos representando os trajetos possíveis entre pontos</li>
        <li>Cada arco (i,j) possui um custo c_ij (distância ou tempo)</li>
        <li>Cada vértice i possui demanda d_i, tempo de serviço s_i e janela temporal [a_i, b_i]</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # 2.1 Definição Formal do Modelo
    st.markdown('<div class="section-header">📐 2.1 Definição Formal do Modelo</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>🔢 Conjuntos e Parâmetros</h4>
        
        <strong>Conjuntos:</strong>
        <ul>
            <li>V = {0, 1, 2, ..., n} → vértices</li>
            <li>K = {1, 2, ..., m} → veículos</li>
            <li>A = {(i,j) | i,j ∈ V, i ≠ j} → arcos</li>
        </ul>
        
        <strong>Parâmetros:</strong>
        <ul>
            <li>c_ij → custo do arco (i,j)</li>
            <li>d_i → demanda no vértice i</li>
            <li>s_i → tempo de serviço no vértice i</li>
            <li>[a_i, b_i] → janela temporal do vértice i</li>
            <li>Q → capacidade do veículo</li>
            <li>T → tempo máximo de rota</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>🎯 Variáveis de Decisão</h4>
        
        <strong>Variáveis binárias:</strong>
        <ul>
            <li>x_ijk = 1 se veículo k percorre o arco (i,j), 0 caso contrário</li>
        </ul>
        
        <strong>Variáveis contínuas:</strong>
        <ul>
            <li>t_ik → tempo de início do serviço no vértice i pelo veículo k</li>
            <li>l_ik → carga do veículo k ao sair do vértice i</li>
        </ul>
        
        <strong>Função Objetivo:</strong>
        <div class="math-formula">
        Minimizar ∑∑∑ c_ij · x_ijk<br>
        k∈K i∈V j∈V
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Restrições
    st.markdown("""
    <div class="info-box">
    <h4>⚖️ Sistema de Restrições</h4>
    
    <strong>1. Restrições de Fluxo:</strong>
    <div class="math-formula">
    ∑∑ x_ijk = 1,   ∀ i ∈ V\\{0}  (cada ponto visitado uma vez)<br>
    k∈K j∈V
    </div>
    
    <strong>2. Conservação de Fluxo:</strong>
    <div class="math-formula">
    ∑ x_ihk - ∑ x_hjk = 0,   ∀ h ∈ V\\{0}, ∀ k ∈ K<br>
    i∈V         j∈V
    </div>
    
    <strong>3. Restrição de Capacidade:</strong>
    <div class="math-formula">
    l_jk ≥ l_ik + d_j - Q(1 - x_ijk),   ∀ i,j ∈ V, ∀ k ∈ K<br>
    0 ≤ l_ik ≤ Q,   ∀ i ∈ V, ∀ k ∈ K
    </div>
    
    <strong>4. Restrições de Janela Temporal:</strong>
    <div class="math-formula">
    t_jk ≥ t_ik + s_i + t_ij - M(1 - x_ijk),   ∀ i,j ∈ V, ∀ k ∈ K<br>
    a_i ≤ t_ik ≤ b_i,   ∀ i ∈ V, ∀ k ∈ K
    </div>
    
    <strong>5. Restrições de Depósito:</strong>
    <div class="math-formula">
    ∑ x_0jk = 1,   ∀ k ∈ K  (cada veículo sai do depósito)<br>
    j∈V<br>
    ∑ x_i0k = 1,   ∀ k ∈ K  (cada veículo retorna ao depósito)<br>
    i∈V
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2.2 Hipótese de Relaxação
    st.markdown('<div class="section-header">🧮 2.2 Hipótese de Relaxação</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>📉 Relaxação Linear</h4>
        
        <strong>Problema Original (MIP):</strong>
        <div class="math-formula">
        x_ijk ∈ {0, 1}
        </div>
        
        <strong>Problema Relaxado (LP):</strong>
        <div class="math-formula">
        0 ≤ x_ijk ≤ 1
        </div>
        
        <p><strong>Justificativa:</strong> A relaxação linear transforma o problema de programação inteira mista 
        em um problema de programação linear, permitindo o uso de métodos eficientes como o Simplex.</p>
        
        <strong>Bound Inferior:</strong>
        <div class="math-formula">
        LB = Z_LP ≤ Z_MIP
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>🎲 Relaxação Lagrangiana</h4>
        
        <strong>Função Lagrangiana:</strong>
        <div class="math-formula">
        L(λ) = min [∑∑∑ c_ij·x_ijk + λ·(∑∑ x_ijk - 1)]<br>
        sujeito a outras restrições
        </div>
        
        <strong>Problema Dual Lagrangiano:</strong>
        <div class="math-formula">
        Z_D = max L(λ)<br>
        λ ≥ 0
        </div>
        
        <p><strong>Vantagens:</strong></p>
        <ul>
            <li>Fornece bounds mais justos que a relaxação linear</li>
            <li>Explora a estrutura decomponível do problema</li>
            <li>Permite soluções factíveis através de heurísticas</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 2.3 Critérios de Poda e Condição de Parada
    st.markdown('<div class="section-header">🌳 2.3 Critérios de Poda e Estratégia de Busca</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>✂️ Critérios de Poda</h4>
        
        <strong>1. Poda por Inviabilidade:</strong>
        <ul>
            <li>Solução viola restrições de capacidade</li>
            <li>Solução viola janelas temporais</li>
            <li>Demanda excede capacidade residual</li>
        </ul>
        
        <strong>2. Poda por Optimalidade:</strong>
        <ul>
            <li>Solução atual é inteira e factível</li>
            <li>Valor da função objetivo não pode ser melhorado</li>
        </ul>
        
        <strong>3. Poda por Bound:</strong>
        <div class="math-formula">
        LB(nó) ≥ UB   →   PODA
        </div>
        <p>onde UB é o melhor valor factível conhecido</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>🔍 Estratégia de Busca</h4>
        
        <strong>Busca em Profundidade (DFS):</strong>
        <ul>
            <li>Explora ramificações até encontrar solução factível</li>
            <li>Menor consumo de memória</li>
            <li>Backtracking sistemático</li>
        </ul>
        
        <strong>Critério de Ramificação:</strong>
        <div class="math-formula">
        Variável x_ijk com valor fracionário mais próximo de 0.5
        </div>
        
        <strong>Condição de Parada:</strong>
        <ul>
            <li>Todos os nós foram explorados ou podados</li>
            <li>Tempo máximo de execução atingido</li>
            <li>Gap de optimalidade ≤ ε</li>
            <li>Número máximo de iterações atingido</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Algoritmo Branch and Bound
    st.markdown("""
    <div class="algorithm-box">
    <h4>📝 Algoritmo Branch and Bound para VRPTW</h4>
    
    <strong>Entrada:</strong> Grafo G, parâmetros do problema<br>
    <strong>Saída:</strong> Solução ótima ou melhor solução encontrada<br><br>
    
    <strong>1. Inicialização:</strong><br>
    &nbsp;&nbsp;UB ← ∞ (melhor solução factível)<br>
    &nbsp;&nbsp;L ← {nó raiz} (lista de nós ativos)<br><br>
    
    <strong>2. Enquanto L ≠ ∅:</strong><br>
    &nbsp;&nbsp;2.1 Selecionar nó n de L (estratégia DFS)<br>
    &nbsp;&nbsp;2.2 Resolver relaxação linear de n → LB(n)<br>
    &nbsp;&nbsp;2.3 Se LB(n) ≥ UB: PODA por bound<br>
    &nbsp;&nbsp;2.4 Se solução é inteira e factível:<br>
    &nbsp;&nbsp;&nbsp;&nbsp;UB ← min(UB, LB(n))<br>
    &nbsp;&nbsp;2.5 Senão se solução é factível:<br>
    &nbsp;&nbsp;&nbsp;&nbsp;Ramificar em novos nós<br>
    &nbsp;&nbsp;&nbsp;&nbsp;Adicionar nós a L<br><br>
    
    <strong>3. Retornar UB</strong>
    </div>
    """, unsafe_allow_html=True)

def pagina_implementacao_algoritmo():
    st.markdown('<div class="main-header">IMPLEMENTAÇÃO DO BRANCH AND BOUND</div>', unsafe_allow_html=True)
    
    # Opção de escolher a base de dados
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
    
    # Botão para download do template
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
    
    else:  # Upload de Arquivo Personalizado
        with col_data2:
            uploaded_file = st.file_uploader(
                "Faça upload do arquivo CSV",
                type=['csv'],
                help="O arquivo deve conter as colunas: id_ponto, bairro, latitude, longitude, demanda_kg, tempo_servico_min, janela_inicio, janela_fim, prioridade"
            )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validar colunas obrigatórias
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
                
                # Converter janelas de tempo
                def time_to_minutes(time_str):
                    try:
                        if ':' in str(time_str):
                            hours, minutes = map(int, str(time_str).split(':'))
                            return hours * 60 + minutes
                        else:
                            return 0
                    except:
                        return 0
                
                df['janela_inicio_min'] = df['janela_inicio'].apply(time_to_minutes)
                df['janela_fim_min'] = df['janela_fim'].apply(time_to_minutes)
                
                # Adicionar colunas opcionais se não existirem
                if 'prioridade' not in df.columns:
                    df['prioridade'] = 1
                if 'bairro' not in df.columns:
                    df['bairro'] = 'Não especificado'
                
                st.success(f"✅ Arquivo carregado com sucesso: {len(df)} pontos de coleta")
                
                # Mostrar preview dos dados
                with st.expander("📋 Visualizar Preview dos Dados"):
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Estatísticas básicas
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
    
    # Parâmetros de configuração
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
    
    # Definir número máximo de veículos como um valor fixo alto (ilimitado na prática)
    max_vehicles = 50  # Valor fixo suficiente para qualquer cenário
    
    # Execução do algoritmo
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 Executar Branch and Bound", type="primary", use_container_width=True):
            with st.spinner("Executando algoritmo Branch and Bound..."):

                solver = VRPTWSolver(df, vehicle_capacity, max_vehicles, time_limit)
                
                # Executar algoritmo
                start_time = time.time()
                solution, metrics = solver.solve(search_strategy)
                execution_time = time.time() - start_time
                
                greedy_routes, greedy_cost = solver._greedy_heuristic()
                
                # Armazenar resultados na sessão
                st.session_state.solver = solver
                st.session_state.solution = solution
                st.session_state.metrics = metrics
                st.session_state.greedy_solution = greedy_routes
                st.session_state.greedy_cost = greedy_cost
                st.session_state.execution_time = execution_time
    
    with col2:
        if st.button("🧪 Executar Testes Unitários", use_container_width=True) and run_tests:
            with st.spinner("Executando testes unitários..."):
                # Capturar output dos testes
                test_output = StringIO()
                runner = unittest.TextTestRunner(stream=test_output, verbosity=2)
                suite = unittest.TestLoader().loadTestsFromTestCase(TestVRPTWSolver)
                result = runner.run(suite)
                
                st.session_state.test_results = result
                st.session_state.test_output = test_output.getvalue()
    
    # Mostrar resultados se disponíveis
    if 'metrics' in st.session_state:
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Resultados da Execução</div>', unsafe_allow_html=True)
        
        # Métricas de execução
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nós Expandidos", st.session_state.metrics['nodes_expanded'])
        with col2:
            st.metric("Profundidade Máxima", st.session_state.metrics['max_depth'])
        with col3:
            st.metric("Tempo Execução", f"{st.session_state.metrics['execution_time']:.2f}s")
        with col4:
            st.metric("Soluções Encontradas", st.session_state.metrics['solutions_found'])
        
        # Comparação com heurística gulosa
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
        
        # Visualização da Árvore de Busca
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
            # Configuração de visualização
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("### ⚙️ Configurações")
                show_stats = st.checkbox("Mostrar Estatísticas", value=True)
                
            with col1:
                # Criar e mostrar visualização da árvore
                fig = create_tree_visualization(st.session_state.solver)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Árvore vazia ou não disponível para visualização.")
            
            # Estatísticas da árvore
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
                
                # Gráfico de pizza da distribuição de nós
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
                
                # Informações por nível
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
                    fig_levels.add_trace(go.Bar(
                        x=levels, y=totals,
                        name='Total',
                        marker_color='lightblue'
                    ))
                    fig_levels.add_trace(go.Bar(
                        x=levels, y=pruned,
                        name='Podados',
                        marker_color='red'
                    ))
                    fig_levels.add_trace(go.Bar(
                        x=levels, y=solutions,
                        name='Soluções',
                        marker_color='green'
                    ))
                    
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
    
    # Mostrar resultados dos testes
    if 'test_output' in st.session_state:
        st.markdown("---")
        st.markdown('<div class="section-header">🧪 Resultados dos Testes Unitários</div>', unsafe_allow_html=True)
        
        st.code(st.session_state.test_output, language='text')
        
        if st.session_state.test_results.wasSuccessful():
            st.success("✅ Todos os testes passaram!")
        else:
            st.error("❌ Alguns testes falharam.")
            
def pagina_budget():
    st.markdown('<div class="main-header">BUDGET E ANÁLISE FINANCEIRA</div>', unsafe_allow_html=True)

    if 'metrics' not in st.session_state:
        st.info("ℹ️ Execute o algoritmo na página de 'IMPLEMENTAÇÃO DO ALGORITMO' para calcular o budget.")
        return

    # --- 1. Premissas de Custo (Inputs do Usuário) ---
    st.markdown('<div class="section-header">⚙️ 1. Premissas de Custo (Mensal)</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        custo_km_combustivel = st.number_input("Custo do Combustível por KM (R$)", 0.0, 10.0, 5.50, 0.10, help="Custo médio de diesel por KM rodado.")
    with col2:
        custo_km_manutencao = st.number_input("Custo de Manutenção por KM (R$)", 0.0, 5.0, 2.00, 0.10, help="Custo de pneus, óleo, desgaste, etc., por KM rodado.")

    st.markdown('<div class="section-header">📉 2. Premissas de Custo Fixo e Investimento</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        custo_fixo_mensal = st.number_input("Custos Fixos Mensais (R$)", 0, 10000000, 415000, 1000, help="Salários base, aluguel da garagem, seguros, depreciação, etc.")
    with col4:
        custo_investimento_total = st.number_input("Investimento Único no Projeto (R$)", 0, 500000, 200000, 1000, help="Custo para desenvolver/adquirir o software de otimização e treinar as equipes.")

    # --- 2. Dados dos Cenários (Puxado do st.session_state) ---
    st.markdown('<div class="section-header">📊 3. Análise Comparativa de Custos</div>', unsafe_allow_html=True)
    
    # Pegar dados do algoritmo
    km_otimizado = st.session_state.metrics['best_cost']
    km_atual_heuristica = st.session_state.greedy_cost
    
    # Permitir que o usuário insira um valor "atual" manual, usando a heurística como padrão
    km_atual = st.number_input(
        "Distância 'Cenário Atual' Manual (KM por Mês)", 
        0.0, 200000.0, 
        km_atual_heuristica * 30,  # Multiplicando por 30 para simular um mês
        100.0,
        help="Insira a quilometragem mensal atual. O padrão é (Resultado da Heurística Gulosa * 30 dias)."
    )
    st.caption(f"Valor diário da heurística gulosa: {km_atual_heuristica:,.2f} km. Valor Otimizado (B&B) diário: {km_otimizado:,.2f} km.")
    
    # Simular KM mensal otimizado
    km_otimizado_mensal = km_otimizado * 30

    # --- 3. Cálculos do Budget ---
    
    # Custos Variáveis
    custo_var_atual_comb = km_atual * custo_km_combustivel
    custo_var_atual_man = km_atual * custo_km_manutencao
    total_custo_var_atual = custo_var_atual_comb + custo_var_atual_man
    
    custo_var_otimizado_comb = km_otimizado_mensal * custo_km_combustivel
    custo_var_otimizado_man = km_otimizado_mensal * custo_km_manutencao
    total_custo_var_otimizado = custo_var_otimizado_comb + custo_var_otimizado_man
    
    # Custos Totais
    total_atual = custo_fixo_mensal + total_custo_var_atual
    total_otimizado = custo_fixo_mensal + total_custo_var_otimizado
    
    # Economia
    economia_mensal = total_atual - total_otimizado
    economia_percentual_total = (economia_mensal / total_atual) * 100 if total_atual > 0 else 0

    # --- 4. Exibição dos Resultados ---

    st.metric(label="Custo Total Otimizado (Mês)", value=f"R$ {total_otimizado:,.2f}", help=f"Valor anterior: R$ {total_atual:,.2f}")

    st.markdown("---")
    st.markdown("### 📋 Tabela de Budget Comparativo (Mensal)")
    
    # Criar DataFrame para a tabela
    budget_data = {
        'Categoria': [
            '**Custos Fixos**', 
            '   Salários, Aluguel, Depreciação, etc.',
            '**Custos Variáveis**', 
            '   Combustível (R$/km)', 
            '   Manutenção (R$/km)',
            '**TOTAL CUSTOS VARIÁVEIS**',
            '**CUSTO TOTAL MENSAL**'
        ],
        'Cenário Atual (R$)': [
            f"**{custo_fixo_mensal:,.2f}**", 
            f"{custo_fixo_mensal:,.2f}",
            f"**{total_custo_var_atual:,.2f}**", 
            f"{custo_var_atual_comb:,.2f}", 
            f"{custo_var_atual_man:,.2f}",
            f"**{total_custo_var_atual:,.2f}**",
            f"**{total_atual:,.2f}**"
        ]
    }
    budget_df = pd.DataFrame(budget_data)
    
    # Usar st.markdown para renderizar a tabela com negrito
    st.markdown(budget_df.to_markdown(index=False), unsafe_allow_html=True)


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
    
    # Análise de sensibilidade
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
                
                # Gráfico de sensibilidade
                fig = px.line(x=capacities, y=costs, 
                            title="Sensibilidade à Capacidade do Veículo",
                            labels={'x': 'Capacidade (kg)', 'y': 'Custo Total (km)'})
                fig.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white')
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Métricas de Performance")
        
        metrics_data = {
            'Metrica': ['Nós Expandidos', 'Profundidade Máxima', 'Tempo Execução', 
                        'Soluções Encontradas', 'Nós Poados', 'Custo Total'],
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
    
    # Visualização das rotas no mapa
    st.markdown("### 🗺️ Visualização das Rotas Otimizadas")
    
    if 'solution' in st.session_state:
        # Criar mapa
        mapa = folium.Map(location=[-25.4284, -49.2733], zoom_start=11)
        
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkred', 'lightred']
        
        # Adicionar rotas
        for i, route in enumerate(st.session_state.solution):
            color = colors[i % len(colors)]
            
            # Adicionar linha da rota
            route_coords = []
            for point_id in route:
                point_data = df[df['id_ponto'] == point_id].iloc[0]
                route_coords.append([point_data['latitude'], point_data['longitude']])
            
            if len(route_coords) > 1:
                folium.PolyLine(route_coords, color=color, weight=3, opacity=0.8,
                                popup=f'Rota Veículo {i+1}').add_to(mapa)
            
            # Adicionar marcadores
            for j, point_id in enumerate(route):
                point_data = df[df['id_ponto'] == point_id].iloc[0]
                
                if point_id == 0:  # Depósito
                    folium.Marker(
                        [point_data['latitude'], point_data['longitude']],
                        popup=f"🚛 DEPÓSITO (Veículo {i+1})",
                        tooltip="Depósito",
                        icon=folium.Icon(color='red', icon='home')
                    ).add_to(mapa)
                else:
                    folium.CircleMarker(
                        [point_data['latitude'], point_data['longitude']],
                        radius=6,
                        popup=f"Ponto {point_id} - Rota {i+1}",
                        tooltip=f"Ponto {point_id}",
                        color=color,
                        fill=True,
                        fillColor=color
                    ).add_to(mapa)
        
        folium_static(mapa, width=1200, height=500)
    
    # Visualização da Árvore de Busca
    st.markdown("---")
    st.markdown("### 🌳 Árvore de Busca Branch and Bound")
    
    if 'solver' in st.session_state and st.session_state.solver.tree_nodes:
        tab1, tab2 = st.tabs(["Visualização da Árvore", "Análise Detalhada"])
        
        with tab1:
            # Criar e mostrar visualização da árvore
            fig = create_tree_visualization(st.session_state.solver)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Árvore vazia ou não disponível para visualização.")
        
        with tab2:
            solver = st.session_state.solver
            
            # Informações detalhadas dos nós
            st.markdown("#### 📋 Informações Detalhadas")
            
            node_data = []
            for node in solver.tree_nodes[:50]:  # Limitar a 50 primeiros nós
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

# =============================================================================
# MENU PRINCIPAL
# =============================================================================

def main():
    st.sidebar.title("🗂️ Navegação")
    pagina_selecionada = st.sidebar.radio(
        "Selecione a página:",
        ["AQUISIÇÃO E PREPARO DOS DADOS", 
         "MODELAGEM MATEMÁTICA", 
         "IMPLEMENTAÇÃO DO ALGORITMO", 
         "RESULTADOS E ANÁLISE",
         "BUDGET E ANÁLISE FINANCEIRA"]  # <-- PÁGINA ADICIONADA AQUI
    )

    # Informações do dataset na sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Informações do Dataset")
    
    df_sidebar = load_real_data()
    if df_sidebar is not None:
        st.sidebar.write(f"**Total de registros:** {len(df_sidebar)}")
        st.sidebar.write(f"**Pontos de coleta:** {len(df_sidebar) - 1}")
        st.sidebar.write(f"**Depósito:** 1 (CIC)")
        st.sidebar.write(f"**Bairros atendidos:** {df_sidebar['bairro'].nunique() - 1}")
    else:
        st.sidebar.write("**Dados não carregados**")

    # Executar a página selecionada
    if pagina_selecionada == "AQUISIÇÃO E PREPARO DOS DADOS":
        pagina_aquisicao_preparo()
    elif pagina_selecionada == "MODELAGEM MATEMÁTICA":
        pagina_modelagem_matematica()
    elif pagina_selecionada == "BUDGET E ANÁLISE FINANCEIRA": # <-- CHAMADA ADICIONADA AQUI
        pagina_budget()
    elif pagina_selecionada == "IMPLEMENTAÇÃO DO ALGORITMO":
        pagina_implementacao_algoritmo()
    elif pagina_selecionada == "RESULTADOS E ANÁLISE":
        pagina_resultados_analise()

    # Rodapé
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Trabalho de Pesquisa Operacional - Otimização de Rotas de Coleta de Lixo<br>"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()