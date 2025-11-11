import unittest
import pandas as pd
from models.solver import VRPTWSolver

class TestVRPTWSolver(unittest.TestCase):
    """Testes unitários para o solver VRPTW"""
    
    def setUp(self):
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
        dist_matrix = self.solver._calculate_distance_matrix()
        self.assertEqual(dist_matrix.shape, (3, 3))
        self.assertEqual(dist_matrix[0][0], 0)
        
    def test_bound_calculation(self):
        from models.node import VRPTWNode
        node = VRPTWNode()
        bound = self.solver._calculate_bound(node)
        self.assertGreaterEqual(bound, 0)
        
    def test_feasibility_check(self):
        from models.node import VRPTWNode
        node = VRPTWNode(visited=set(), vehicle_load=0, vehicle_time=0, routes=[[0]])
        feasible = self.solver._is_feasible(node, 1)
        self.assertTrue(feasible)