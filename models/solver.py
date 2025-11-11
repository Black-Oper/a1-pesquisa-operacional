import numpy as np
from math import radians, sin, cos, sqrt, atan2
import heapq
import time
from models.node import VRPTWNode

class VRPTWSolver:
    """Solver Branch and Bound para VRPTW"""
    
    def __init__(self, df, vehicle_capacity=5000, max_vehicles=5, time_limit=300):
        self.df = df
        self.n_points = len(df)
        self.vehicle_capacity = vehicle_capacity
        self.max_vehicles = max_vehicles
        self.time_limit = time_limit
        self.KM_PARA_MINUTOS_FATOR = 2.0
        
        self.dist_matrix = self._calculate_distance_matrix()
        
        self.nodes_expanded = 0
        self.max_depth = 0
        self.start_time = 0
        self.solutions_found = 0
        self.pruned_nodes = 0
        
        self.best_cost = float('inf')
        self.best_solution = None
        
        self.tree_nodes = []
        self.max_tree_nodes = 500
        
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
                    dist_matrix[i][j] = 6371 * c
                    
        return dist_matrix
    
    def _calculate_bound(self, node):
        """Calcula o limite inferior para um nó usando MST dos pontos não visitados"""
        bound = node.cost
        
        if len(node.visited) < self.n_points - 1:
            unvisited = set(range(1, self.n_points)) - node.visited
            
            if unvisited:
                last_point = node.get_last_point()
                min_exit_cost = min([self.dist_matrix[last_point][j] for j in unvisited], default=0)
                bound += min_exit_cost
                
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
                    bound += mst_cost * 0.5
                
                min_return_cost = min([self.dist_matrix[j][0] for j in unvisited], default=0)
                bound += min_return_cost
        
        return bound
    
    def _calculate_travel_time(self, point_a_idx, point_b_idx):
        return self.dist_matrix[point_a_idx][point_b_idx] * self.KM_PARA_MINUTOS_FATOR
    
    def _is_feasible(self, node, next_point):
        """Verifica se adicionar next_point à rota atual é viável"""
        if next_point in node.visited:
            return False
            
        demand = self.df.iloc[next_point]['demanda_kg']
        if node.vehicle_load + demand > self.vehicle_capacity:
            return False
            
        last_point = node.get_last_point()
        travel_time = self._calculate_travel_time(last_point, next_point)
        arrival_time = node.vehicle_time + travel_time
        
        time_window_start = self.df.iloc[next_point]['janela_inicio_min']
        time_window_end = self.df.iloc[next_point]['janela_fim_min']
        
        if arrival_time > time_window_end:
            return False
            
        return True
    
    def _can_start_new_route(self, node):
        return len(node.routes) < self.max_vehicles
    
    def _update_best_solution(self, node):
        """Atualiza a melhor solução encontrada"""
        complete_routes = []
        total_cost = 0
        
        for route in node.routes:
            if len(route) > 1:
                if route[-1] != 0:
                    complete_route = route + [0]
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
        current_route = [0]
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
        self.tree_nodes = []
        
        VRPTWNode.reset_counter()
        
        best_routes, best_cost = self._greedy_heuristic()
        self.best_solution = best_routes
        self.best_cost = best_cost
        
        print(f"Solução inicial (gulosa): custo = {best_cost:.2f} km")
        
        root = VRPTWNode()
        root.bound = self._calculate_bound(root)
        
        if len(self.tree_nodes) < self.max_tree_nodes:
            self.tree_nodes.append(root)
        
        queue = []
        if search_strategy == 'best-first':
            heapq.heappush(queue, root)
        else:
            queue = [root]
        
        while queue and (time.time() - self.start_time) < self.time_limit:
            if search_strategy == 'best-first':
                node = heapq.heappop(queue)
            else:
                node = queue.pop()
                
            self.nodes_expanded += 1
            self.max_depth = max(self.max_depth, node.level)
            
            if node.bound >= self.best_cost:
                self.pruned_nodes += 1
                node.is_pruned = True
                continue
            
            if len(node.visited) == self.n_points - 1:
                self.solutions_found += 1
                node.is_solution = True
                self._update_best_solution(node)
                continue
            
            expanded = False
            
            for next_point in range(1, self.n_points):
                if next_point not in node.visited and self._is_feasible(node, next_point):
                    expanded = True
                    
                    new_visited = node.visited.copy()
                    new_visited.add(next_point)
                    
                    new_routes = [route.copy() for route in node.routes]
                    new_routes[node.current_route_idx].append(next_point)
                    
                    last_point = node.get_last_point()
                    travel_time = self._calculate_travel_time(last_point, next_point)
                    arrival_time = node.vehicle_time + travel_time
                    
                    point_data = self.df.iloc[next_point]
                    time_window_start = point_data['janela_inicio_min']
                    service_time = point_data['tempo_servico_min']
                    
                    service_start_time = max(arrival_time, time_window_start)
                    new_time = service_start_time + service_time
                    new_cost = node.cost + self.dist_matrix[last_point][next_point]
                    new_load = node.vehicle_load + point_data['demanda_kg']
                    
                    new_node = VRPTWNode(
                        level=node.level + 1,
                        cost=new_cost,
                        visited=new_visited,
                        routes=new_routes,
                        current_route_idx=node.current_route_idx,
                        vehicle_load=new_load,
                        vehicle_time=new_time,
                        parent=node
                    )
                    
                    new_node.bound = self._calculate_bound(new_node)
                    
                    if new_node.bound < self.best_cost:
                        if search_strategy == 'best-first':
                            heapq.heappush(queue, new_node)
                        else:
                            queue.append(new_node)
                        
                        if len(self.tree_nodes) < self.max_tree_nodes:
                            self.tree_nodes.append(new_node)
                    else:
                        self.pruned_nodes += 1
                        new_node.is_pruned = True
                        if len(self.tree_nodes) < self.max_tree_nodes:
                            self.tree_nodes.append(new_node)
            
            if len(node.visited) < self.n_points - 1 and self._can_start_new_route(node):
                new_routes = [route.copy() for route in node.routes]
                
                if new_routes[node.current_route_idx][-1] != 0:
                    last_point = new_routes[node.current_route_idx][-1]
                    new_routes[node.current_route_idx].append(0)
                    route_return_cost = self.dist_matrix[last_point][0]
                else:
                    route_return_cost = 0
                
                new_routes.append([0])
                new_route_idx = len(new_routes) - 1
                
                new_node = VRPTWNode(
                    level=node.level + 1,
                    cost=node.cost + route_return_cost,
                    visited=node.visited.copy(),
                    routes=new_routes,
                    current_route_idx=new_route_idx,
                    vehicle_load=0,
                    vehicle_time=0,
                    parent=node
                )
                
                new_node.bound = self._calculate_bound(new_node)
                
                if new_node.bound < self.best_cost:
                    if search_strategy == 'best-first':
                        heapq.heappush(queue, new_node)
                    else:
                        queue.append(new_node)
                    
                    if len(self.tree_nodes) < self.max_tree_nodes:
                        self.tree_nodes.append(new_node)
                else:
                    self.pruned_nodes += 1
                    new_node.is_pruned = True
                    if len(self.tree_nodes) < self.max_tree_nodes:
                        self.tree_nodes.append(new_node)
            
            if not expanded and not self._can_start_new_route(node):
                self.pruned_nodes += 1
        
        execution_time = time.time() - self.start_time
        
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