import plotly.graph_objects as go

def create_tree_visualization(solver):
    """Cria visualização da árvore de busca Branch and Bound usando Plotly"""
    
    if not solver.tree_nodes:
        return None
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    edge_x = []
    edge_y = []
    
    level_counts = {}
    level_positions = {}
    
    for node in solver.tree_nodes:
        level = node.level
        if level not in level_counts:
            level_counts[level] = 0
            level_positions[level] = 0
        level_counts[level] += 1
    
    node_positions = {}
    for node in solver.tree_nodes:
        level = node.level
        y = -level
        
        total_in_level = level_counts[level]
        position_in_level = level_positions[level]
        level_positions[level] += 1
        
        if total_in_level > 1:
            x = (position_in_level - (total_in_level - 1) / 2) * 2
        else:
            x = 0
        
        node_positions[node.node_id] = (x, y)
        
        node_x.append(x)
        node_y.append(y)
        
        route_str = '→'.join(map(str, node.get_current_route()))
        text = f"ID: {node.node_id}<br>"
        text += f"Nível: {node.level}<br>"
        text += f"Custo: {node.cost:.2f}<br>"
        text += f"Bound: {node.bound:.2f}<br>"
        text += f"Rota: {route_str}<br>"
        text += f"Visitados: {len(node.visited)}/{solver.n_points-1}"
        node_text.append(text)
        
        if node.is_solution:
            node_color.append('green')
        elif node.is_pruned:
            node_color.append('red')
        elif node.level == 0:
            node_color.append('blue')
        else:
            node_color.append('lightblue')
        
        if node.parent and node.parent.node_id in node_positions:
            parent_x, parent_y = node_positions[node.parent.node_id]
            edge_x.extend([parent_x, x, None])
            edge_y.extend([parent_y, y, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
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
    
    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text='Árvore de Busca Branch and Bound',
                font=dict(size=16)
            ),
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