# 🗑️ Otimização de Rotas de Coleta de Lixo (VRPTW) com Branch and Bound

Este projeto é um dashboard interativo em **Streamlit** que analisa, modela e resolve um Problema de Roteamento de Veículos com Janelas de Tempo (VRPTW) aplicado a dados reais de coleta de lixo na cidade de Curitiba.

A aplicação implementa um algoritmo **Branch and Bound (B&B)** do zero para encontrar rotas otimizadas, visando minimizar a distância total percorrida pela frota de veículos.

## 🎯 Problema Solucionado

O projeto aborda o **Problema de Roteamento de Veículos com Janelas de Tempo (VRPTW)**, um problema clássico de otimização combinatória NP-difícil. O objetivo é determinar um conjunto de rotas de custo mínimo (distância) para uma frota de veículos, de forma que:

1.  Cada rota comece e termine no depósito central (Ponto 0, no CIC).
2.  Todos os 200 pontos de coleta sejam visitados exatamente uma vez.
3.  A demanda total de resíduos de uma rota não exceda a capacidade do caminhão.
4.  O serviço em cada ponto seja realizado dentro da sua janela de tempo (ex: das 06:00 às 14:00).

## ✨ Funcionalidades do Dashboard

O dashboard é dividido em várias seções para cobrir todo o ciclo do projeto de Pesquisa Operacional:

* **📊 Aquisição e Preparo dos Dados:**
    * Carrega o dataset real de Curitiba.
    * Exibe estatísticas descritivas (total de pontos, demanda total, bairros atendidos).
    * Apresenta visualizações interativas:
        * Mapa (Folium) com a localização de todos os pontos e do depósito.
        * Gráficos (Plotly) de distribuição por bairro, prioridade e demanda.
    * Disponibiliza uma tabela de dados filtrável.

* **📐 Modelagem Matemática:**
    * Uma página estática que descreve formalmente o modelo VRPTW.
    * Define os conjuntos, parâmetros, variáveis de decisão, função objetivo e restrições.
    * Explica os conceitos de relaxação, critérios de poda e estratégias de busca (DFS vs. Best-First) usados no Branch and Bound.

* **⚙️ Implementação do Algoritmo:**
    * Permite ao usuário **executar o solver B&B** interativamente.
    * **Configuração de Parâmetros:** O usuário pode definir a capacidade dos veículos, número máximo de veículos, tempo limite de execução e estratégia de busca.
    * **Upload de Dados:** Permite usar a base padrão ou fazer upload de um CSV personalizado (com template disponível).
    * **Visualização da Árvore B&B:** Exibe um gráfico dinâmico (Plotly) da árvore de busca gerada, mostrando nós explorados, podados e soluções encontradas em tempo real.

* **🗺️ Resultados e Análise:**
    * Exibe as métricas de performance do solver (tempo, nós expandidos, custo da solução).
    * Compara o resultado otimizado com uma heurística gulosa simples, calculando o ganho percentual.
    * Mostra as rotas otimizadas em um mapa Folium, com cores diferentes para cada veículo.

* **💰 Budget e Análise Financeira:**
    * Calcula o impacto financeiro da otimização.
    * O usuário insere premissas de custo (custo/km, custos fixos mensais).
    * A aplicação gera uma tabela comparativa de "Custo Atual" vs. "Custo Otimizado", projetando a economia mensal.

## 📦 Dataset

* **Fonte:** [Kaggle - Rota Coleta Curitiba](https://www.kaggle.com/datasets/jeonjungkookbts/rota-coleta-curitiba)
* **Descrição:** O dataset contém 201 registros (1 depósito + 200 pontos de coleta) e 9 colunas, incluindo `id_ponto`, `bairro`, `latitude`, `longitude`, `demanda_kg`, `tempo_servico_min`, `janela_inicio` e `janela_fim`.
* **Licença:** Os dados são públicos. Ao utilizá-los, cite a fonte original no Kaggle.
>
> `df = pd.read_csv('rota_coleta_curitiba.csv')`

## 🧠 Lógica da Implementação (Rastreabilidade)

A lógica central do projeto está nas classes `VRPTWSolver` e `VRPTWNode`, que implementam o algoritmo Branch and Bound.

### 1. Pré-processamento de Dados (`load_real_data`)

1.  **Cache:** A função usa `@st.cache_data` para carregar o CSV apenas uma vez e armazená-lo em cache, melhorando a performance do dashboard.
2.  **Conversão de Tempo:** A decisão de implementação mais crítica é a conversão das janelas de tempo. Colunas como `janela_inicio` ("06:00") são strings, o que impossibilita cálculos. Elas são convertidas para minutos desde a meia-noite (ex: "06:00" -> `360`). Isso permite aritmética simples para verificar as janelas de tempo.

### 2. Cálculo de Distâncias (`_calculate_distance_matrix`)

* Antes de iniciar o solver, é calculada uma matriz de distâncias (N x N) entre todos os 201 pontos.
* **Lógica:** A **fórmula de Haversine** é usada para calcular a distância geodésica (em km) entre dois pares de latitude/longitude. Isso é feito uma única vez e armazenado na matriz `self.dist_matrix` para consultas rápidas.

### 3. Algoritmo Branch and Bound (`VRPTWSolver.solve`)

O B&B é um algoritmo de busca exata. Ele explora inteligentemente o espaço de soluções (árvore de busca) para encontrar a solução ótima, podando ramos que comprovadamente não levarão a um resultado melhor.

#### A. Heurística Inicial (Definindo o *Upper Bound* - UB)

* **Propósito:** O B&B precisa de um "benchmark" inicial. Se não tivermos uma solução, nosso `best_cost` (Upper Bound ou Limite Superior) é infinito, e não podemos podar nada.
* **Lógica (`_greedy_heuristic`):** Antes de iniciar o B&B, uma heurística gulosa (estilo "vizinho mais próximo") é executada. Ela constrói rotas rapidamente, sempre escolhendo o próximo ponto viável mais próximo.
* **Resultado:** Isso nos dá uma solução completa, mas provavelmente não-ótima (ex: `best_cost = 1500 km`). Este valor é o nosso **Upper Bound (UB)** inicial.

#### B. O Nó (`VRPTWNode`)

* Cada nó na árvore representa uma **solução parcial**. Ele armazena:
    * `cost`: O custo (distância) acumulado até este ponto.
    * `visited`: O conjunto de pontos já visitados.
    * `routes`: A lista de rotas atuais (ex: `[[0, 5, 12], [0, 8]]`).
    * `bound`: O **Limite Inferior (LB)**. Esta é a "mágica" do B&B.

#### C. O Limite Inferior (Bound - `_calculate_bound`)

* **Propósito:** Esta é a "parte Bound" (limitar). Para qualquer nó (solução parcial), precisamos de uma *estimativa otimista* de qual será o custo *mínimo* para completar a solução a partir dali.
* **Lógica:** O bound é calculado como:
    `bound = (Custo Atual) + (Estimativa Mínima para Terminar)`
* A estimativa é uma relaxação: (custo mínimo para sair do ponto atual) + (uma estimativa de Árvore Geradora Mínima - MST - para conectar todos os pontos restantes) + (custo mínimo para voltar ao depósito de algum ponto não visitado).
* Esta é a métrica mais importante do nó.

#### D. O Loop de Busca (`solve`)

1.  **Inicialização:** O `root` (nó raiz, no depósito, custo 0) é criado e seu `bound` é calculado. Ele é adicionado a uma fila de prioridade (`heapq`).
2.  **Loop:** O algoritmo entra em um loop `while queue is not empty`:
3.  **Seleção:** Pega o nó com o **menor `bound`** da fila (estratégia Best-First). Este é o nó *mais promissor*.
4.  **PODA (Pruning):** O algoritmo faz a pergunta-chave:
    `if node.bound >= self.best_cost:`
    * Se a *estimativa mais otimista* (`bound`) deste nó já é *pior* que a *melhor solução completa* que já encontramos (`best_cost`), não há sentido em explorar este ramo. O nó é descartado (**podado**).
5.  **SOLUÇÃO:** Se o nó não foi podado, verificamos: `len(node.visited) == self.n_points - 1`?
    * Se sim, encontramos uma solução completa.
    * Verificamos se seu `cost < self.best_cost`. Se for, ótimo! Encontramos uma solução melhor. Atualizamos `self.best_cost` para este novo valor (ex: `best_cost` agora é `1300 km`).
6.  **RAMIFICAÇÃO (Branching):** Se não foi podado e não é uma solução, precisamos "ramificar" (criar filhos). O solver tenta duas ações:
    * **Opção 1 (Adicionar Ponto):** Para cada ponto `P` ainda não visitado, o solver verifica se é viável adicioná-lo à rota atual (função `_is_feasible`).
    * **Opção 2 (Nova Rota):** O solver tenta "fechar" a rota atual (voltar ao depósito) e começar uma nova, se o `max_vehicles` permitir.

#### E. Verificação de Viabilidade (`_is_feasible`)

* Esta função garante que as regras do VRPTW sejam seguidas. Ao tentar adicionar um `next_point`, ela verifica duas coisas:
    1.  **Capacidade:** `node.vehicle_load + demand[next_point] <= self.vehicle_capacity`
    2.  **Janela de Tempo:** O tempo de chegada (`arrival_time`) ao ponto deve ser **menor ou igual** ao `janela_fim_min` do ponto. (Se o caminhão chegar *antes* da `janela_inicio_min`, o código assume que ele pode esperar).

Este processo de "Ramificar" e "Podar" continua até que o tempo se esgote ou a fila fique vazia, garantindo que a `best_cost` encontrada seja a solução ótima (dado tempo suficiente).

## 🚀 Como Executar o Projeto Localmente

### 1. Pré-requisitos

* Python 3.8 ou superior
* Git

### 2. Instalação

1.  Clone o repositório:
    ```bash
    git clone <url-do-seu-repositorio>
    cd <nome-do-repositorio>
    ```

2.  Crie e ative um ambiente virtual (recomendado):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  Crie um arquivo chamado `requirements.txt` com o conteúdo abaixo e instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

4.  Certifique-se de que o arquivo de dados `rota_coleta_curitiba (1).csv` (ou `rota_coleta_curitiba.csv`) esteja na mesma pasta que o `main.py`.

### 3. Execução

1.  Para iniciar o dashboard Streamlit, execute o seguinte comando no seu terminal:
    ```bash
    streamlit run main.py
    ```

2.  O Streamlit abrirá automaticamente o projeto no seu navegador.