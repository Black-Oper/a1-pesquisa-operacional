class VRPTWNode:
    """Nó da árvore de busca do Branch and Bound para VRPTW"""
    
    _node_counter = 0
    
    def __init__(self, level=0, cost=0, bound=0, visited=None, routes=None, 
                 current_route_idx=0, vehicle_load=0, vehicle_time=0, parent=None):
        self.level = level
        self.cost = cost
        self.bound = bound
        self.visited = visited or set()
        self.routes = routes or [[0]]
        self.current_route_idx = current_route_idx
        self.vehicle_load = vehicle_load
        self.vehicle_time = vehicle_time
        self.parent = parent
        
        VRPTWNode._node_counter += 1
        self.node_id = VRPTWNode._node_counter
        
        self.is_solution = False
        self.is_pruned = False
        
    def __lt__(self, other):
        return self.bound < other.bound
    
    def get_current_route(self):
        return self.routes[self.current_route_idx]
    
    def get_last_point(self):
        return self.get_current_route()[-1]
    
    @classmethod
    def reset_counter(cls):
        cls._node_counter = 0