import streamlit as st
import pandas as pd

@st.cache_data
def load_real_data():
    """Carrega os dados reais do arquivo CSV"""
    try:
        df = pd.read_csv('rota_coleta_curitiba (1).csv')
        
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

def time_to_minutes(time_str):
    """Converte string de tempo para minutos desde meia-noite"""
    try:
        if ':' in str(time_str):
            hours, minutes = map(int, str(time_str).split(':'))
            return hours * 60 + minutes
        else:
            return 0
    except:
        return 0