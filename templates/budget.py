import streamlit as st
import pandas as pd

def pagina_budget():
    st.markdown('<div class="main-header">BUDGET E ANÁLISE FINANCEIRA</div>', unsafe_allow_html=True)

    if 'metrics' not in st.session_state:
        st.info("ℹ️ Execute o algoritmo na página de 'IMPLEMENTAÇÃO DO ALGORITMO' para calcular o budget.")
        return

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

    st.markdown('<div class="section-header">📊 3. Análise Comparativa de Custos</div>', unsafe_allow_html=True)
    
    km_otimizado = st.session_state.metrics['best_cost']
    km_atual_heuristica = st.session_state.greedy_cost
    
    km_atual = st.number_input(
        "Distância 'Cenário Atual' Manual (KM por Mês)", 
        0.0, 200000.0, 
        km_atual_heuristica * 30,
        100.0,
        help="Insira a quilometragem mensal atual. O padrão é (Resultado da Heurística Gulosa * 30 dias)."
    )
    st.caption(f"Valor diário da heurística gulosa: {km_atual_heuristica:,.2f} km. Valor Otimizado (B&B) diário: {km_otimizado:,.2f} km.")
    
    km_otimizado_mensal = km_otimizado * 30

    custo_var_atual_comb = km_atual * custo_km_combustivel
    custo_var_atual_man = km_atual * custo_km_manutencao
    total_custo_var_atual = custo_var_atual_comb + custo_var_atual_man
    
    custo_var_otimizado_comb = km_otimizado_mensal * custo_km_combustivel
    custo_var_otimizado_man = km_otimizado_mensal * custo_km_manutencao
    total_custo_var_otimizado = custo_var_otimizado_comb + custo_var_otimizado_man
    
    total_atual = custo_fixo_mensal + total_custo_var_atual
    total_otimizado = custo_fixo_mensal + total_custo_var_otimizado
    
    economia_mensal = total_atual - total_otimizado
    economia_percentual_total = (economia_mensal / total_atual) * 100 if total_atual > 0 else 0

    st.metric(label="Custo Total Otimizado (Mês)", value=f"R$ {total_otimizado:,.2f}", help=f"Valor anterior: R$ {total_atual:,.2f}")

    st.markdown("---")
    st.markdown("### 📋 Tabela de Budget Comparativo (Mensal)")
    
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
    
    st.markdown(budget_df.to_markdown(index=False), unsafe_allow_html=True)