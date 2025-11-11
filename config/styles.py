CSS_STYLES = """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #2b2b2b;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #2d2d2d 0%, #404040 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin: 0.8rem 0;
        color: #ffffff;
        border: 1px solid #404040;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4fc3f7;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #b0b0b0;
        margin-bottom: 0.5rem;
    }
    .stDataFrame {
        background-color: #1e1e1e;
        border-radius: 0.5rem;
    }
    .stMarkdown table {
        width: 100%;
        background-color: #2d2d2d;
        color: #ffffff;
        border-radius: 0.5rem;
    }
    .stMarkdown th {
        background-color: #3a3a3a;
        color: #4fc3f7;
        padding: 0.5rem;
    }
    .stMarkdown td {
        padding: 0.5rem;
        border-bottom: 1px solid #444;
    }
    .math-formula {
        background-color: #1a1a1a;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        color: #ffffff;
    }
    .algorithm-box {
        background-color: #2d2d2d;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #444;
        margin: 1rem 0;
        color: #ffffff;
    }
    .success-box {
        background-color: #1b5e20;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #ffffff;
    }
    .warning-box {
        background-color: #ff6f00;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #ffffff;
    }
    .node-box {
        background-color: #333333;
        padding: 0.5rem;
        margin: 0.2rem;
        border-radius: 0.3rem;
        border: 1px solid #555;
        font-size: 0.8rem;
    }
</style>
"""

PAGE_CONFIG = {
    "page_title": "Otimização de Rotas - Coleta de Lixo Curitiba",
    "page_icon": "🗑️",
    "layout": "wide"
}