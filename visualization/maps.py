import folium

def create_route_map(df, solution=None):
    """Cria mapa com visualização das rotas"""
    mapa = folium.Map(location=[-25.4284, -49.2733], zoom_start=11)
    
    if solution is None:
        # Apenas pontos
        for _, row in df.iterrows():
            if row['id_ponto'] == 0:
                folium.Marker(
                    [row['latitude'], row['longitude']],
                    popup=f"🚛 DEPÓSITO CENTRAL CIC\nBairro: {row['bairro']}",
                    tooltip="Depósito Central CIC",
                    icon=folium.Icon(color='red', icon='home', prefix='fa')
                ).add_to(mapa)
            else:
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
    else:
        # Com rotas
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkred', 'lightred']
        
        for i, route in enumerate(solution):
            color = colors[i % len(colors)]
            
            route_coords = []
            for point_id in route:
                point_data = df[df['id_ponto'] == point_id].iloc[0]
                route_coords.append([point_data['latitude'], point_data['longitude']])
            
            if len(route_coords) > 1:
                folium.PolyLine(route_coords, color=color, weight=3, opacity=0.8,
                                popup=f'Rota Veículo {i+1}').add_to(mapa)
            
            for j, point_id in enumerate(route):
                point_data = df[df['id_ponto'] == point_id].iloc[0]
                
                if point_id == 0:
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
    
    return mapa