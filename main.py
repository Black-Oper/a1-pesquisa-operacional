import streamlit as st
from config.styles import CSS_STYLES, PAGE_CONFIG
from data.loader import load_real_data
from templates import (
    pagina_aquisicao_preparo,
    pagina_modelagem_matematica,
    pagina_implementacao_algoritmo,
    pagina_resultados_analise,
    pagina_budget
)

# Configuração da página
st.set_page_config(**PAGE_CONFIG)

# Aplicar CSS personalizado
st.markdown(CSS_STYLES, unsafe_allow_html=True)

def main():
    st.sidebar.title("🗂️ Navegação")
    pagina_selecionada = st.sidebar.radio(
        "Selecione a página:",
        ["AQUISIÇÃO E PREPARO DOS DADOS", 
         "MODELAGEM MATEMÁTICA", 
         "IMPLEMENTAÇÃO DO ALGORITMO", 
         "RESULTADOS E ANÁLISE",
         "BUDGET E ANÁLISE FINANCEIRA"]
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
    elif pagina_selecionada == "BUDGET E ANÁLISE FINANCEIRA":
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