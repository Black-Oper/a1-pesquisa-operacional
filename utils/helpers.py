def format_currency(value):
    """Formata valores monetários"""
    return f"R$ {value:,.2f}"

def format_distance(value):
    """Formata distâncias"""
    return f"{value:,.2f} km"

def format_time(value):
    """Formata tempo em minutos"""
    return f"{value:,.0f} min"

def format_percentage(value):
    """Formata porcentagens"""
    return f"{value:.1f}%"