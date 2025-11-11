import streamlit as st

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