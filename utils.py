import pandas as pd
import streamlit as st
import io
import base64
from typing import Optional

def format_bytes(size_in_bytes: int) -> str:
    """Formats bytes to a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} TB"

def set_page_style():
    """Injects custom CSS to style the Streamlit application as a premium, glassmorphic dark-themed dashboard."""
    custom_css = """
    <style>
    /* Import modern Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global page layout overrides */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Fade-in Animation for elements */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .metric-card-container, [data-testid="stDataFrame"], [data-testid="stPlotlyChart"], .chat-bubble, .custom-info-box, .custom-success-box {
        animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    /* Title font configuration with gradients */
    h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        background: linear-gradient(135deg, #ffffff 40%, #a5b4fc 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        padding-bottom: 8px;
        margin-bottom: 20px !important;
    }
    
    h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
    }

    /* Sidebar container styling */
    [data-testid="stSidebar"] {
        background-color: #080b11 !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Sidebar Radio Navigation Item Custom Design */
    div[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 6px;
        padding-top: 10px;
    }
    
    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: rgba(30, 41, 59, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.02) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin: 4px 0px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(99, 102, 241, 0.1) !important;
        border-color: rgba(99, 102, 241, 0.25) !important;
        transform: translateX(4px);
    }
    
    div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(79, 70, 229, 0.35) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.7) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Hide the native radio indicator dots completely */
    div[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarker"] {
        display: none !important;
    }
    
    /* Style navigation label text */
    div[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stWidgetLabel"] p {
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        margin: 0 !important;
        padding: 0 !important;
        transition: color 0.2s ease;
    }
    
    div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] div[data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
    }

    /* Hide standard top header bar and footer */
    header[data-testid="stHeader"] {
        background-color: rgba(11, 15, 25, 0.8) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid #1e293b;
    }

    /* Custom main glassmorphic container cards */
    .metric-card-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.35) 0%, rgba(15, 23, 42, 0.55) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 22px;
        box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.25);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 16px;
    }
    
    .metric-card-container:hover {
        transform: translateY(-4px) scale(1.01);
        border-color: rgba(99, 102, 241, 0.45);
        box-shadow: 0 15px 35px 0 rgba(99, 102, 241, 0.15);
    }

    /* Custom Metric Value styling */
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 700;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        font-family: 'Outfit', sans-serif;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 6px;
        display: flex;
        align-items: center;
    }
    
    .metric-delta.positive {
        color: #10b981;
    }
    
    .metric-delta.negative {
        color: #ef4444;
    }

    /* File Uploader styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(99, 102, 241, 0.35);
        border-radius: 14px;
        padding: 24px;
        background-color: rgba(15, 23, 42, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1;
        background-color: rgba(99, 102, 241, 0.05);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.12);
    }
    
    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
        padding: 0 !important;
    }

    /* Beautiful custom buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.35) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.5) !important;
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    }
    
    .stButton>button:active {
        transform: translateY(0) !important;
    }

    /* Secondary styled buttons */
    div[data-testid="stFormSubmitButton"] button, 
    .download-btn-container button {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%) !important;
        box-shadow: 0 4px 14px 0 rgba(6, 182, 212, 0.35) !important;
    }
    
    div[data-testid="stFormSubmitButton"] button:hover,
    .download-btn-container button:hover {
        background: linear-gradient(135deg, #0891b2 0%, #0369a1 100%) !important;
        box-shadow: 0 6px 20px 0 rgba(6, 182, 212, 0.5) !important;
    }

    /* Form and inputs styling */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stTextArea>div>div>textarea {
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }

    /* Table styling overrides */
    div[data-testid="stTable"] table {
        border-collapse: collapse;
        background-color: #0f172a;
        color: #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #1e293b !important;
    }

    div[data-testid="stTable"] th {
        background-color: #1e293b !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
        border-bottom: 2px solid #334155 !important;
    }

    div[data-testid="stTable"] td {
        padding: 12px 16px !important;
        border-bottom: 1px solid #1e293b !important;
    }

    /* Tab styles */
    button[data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        color: #94a3b8;
        border-bottom-width: 2px !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #6366f1 !important;
        border-bottom-color: #6366f1 !important;
    }

    /* Chat Styling */
    .chat-bubble {
        padding: 16px 20px;
        border-radius: 16px;
        margin-bottom: 16px;
        max-width: 85%;
        line-height: 1.6;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    
    .chat-bubble.user {
        background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        margin-left: auto;
        color: #f1f5f9;
        border-bottom-right-radius: 2px;
    }
    
    .chat-bubble.assistant {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        margin-right: auto;
        color: #f8fafc;
        border-bottom-left-radius: 2px;
    }

    /* Dashboard Footer styling */
    .dashboard-footer {
        text-align: center;
        padding: 24px 0;
        margin-top: 48px;
        border-top: 1px solid #1e293b;
        color: #64748b;
        font-size: 0.85rem;
    }

    /* custom info boxes */
    .custom-info-box {
        background-color: rgba(30, 41, 59, 0.4);
        border-left: 4px solid #6366f1;
        padding: 18px;
        border-radius: 6px 12px 12px 6px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .custom-success-box {
        background-color: rgba(16, 185, 129, 0.08);
        border-left: 4px solid #10b981;
        padding: 18px;
        border-radius: 6px 12px 12px 6px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .custom-warning-box {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        padding: 18px;
        border-radius: 6px 12px 12px 6px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .custom-error-box {
        background-color: rgba(239, 68, 68, 0.08);
        border-left: 4px solid #ef4444;
        padding: 18px;
        border-radius: 6px 12px 12px 6px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_metric_card(label: str, value: str, delta: Optional[str] = None, is_positive: bool = True, icon: Optional[str] = None, accent_color: str = "#6366f1"):
    """Renders a custom glassmorphic metric card using HTML and CSS with colored left border accents."""
    delta_class = "positive" if is_positive else "negative"
    delta_symbol = "▲" if is_positive else "▼"
    
    delta_html = ""
    if delta:
        delta_html = f'<div class="metric-delta {delta_class}">{delta_symbol} {delta}</div>'
        
    icon_html = f'<span style="font-size: 1.6rem; margin-right: 10px;">{icon}</span>' if icon else ""
    
    card_html = f"""
    <div class="metric-card-container" style="border-left: 5px solid {accent_color};">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            {icon_html}
            <div class="metric-label">{label}</div>
        </div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def to_excel(df: pd.DataFrame) -> bytes:
    """Converts a pandas DataFrame to bytes for Excel download."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def render_footer():
    """Renders the standard footer on pages."""
    footer_html = """
    <div class="dashboard-footer">
        <p>⚡ AI Data Analysis Assistant • Built with Streamlit & Groq Llama-3.3 • © 2026</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
