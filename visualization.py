import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from typing import Tuple, Optional, List

# Set modern Matplotlib settings for high quality PNG export
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = '#0f172a'
plt.rcParams['axes.facecolor'] = '#0f172a'
plt.rcParams['grid.color'] = '#1e293b'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5

# Theme color scheme matching electric blue/indigo/purple styling
COLOR_PALETTE = ['#6366f1', '#06b6d4', '#ec4899', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444']

def get_chart_dir() -> str:
    """Gets or creates the charts output directory."""
    chart_dir = os.path.join(os.getcwd(), 'charts')
    if not os.path.exists(chart_dir):
        os.makedirs(chart_dir)
    return chart_dir

def save_matplotlib_figure(fig, filename: str) -> str:
    """Saves a matplotlib figure as PNG and returns the absolute path."""
    filepath = os.path.join(get_chart_dir(), filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    return filepath

def generate_plotly_chart(
    df: pd.DataFrame, 
    chart_type: str, 
    x_col: str, 
    y_col: Optional[str] = None, 
    color_col: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    show_grid: bool = True
) -> go.Figure:
    """
    Generates a beautifully styled interactive Plotly chart.
    """
    fig = None
    
    # 1. Bar Chart
    if chart_type == "Bar Chart":
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=title,
                     color_discrete_sequence=COLOR_PALETTE)
    
    # 2. Line Chart
    elif chart_type == "Line Chart":
        fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title,
                      color_discrete_sequence=COLOR_PALETTE)
        fig.update_traces(line=dict(width=3))
        
    # 3. Scatter Plot
    elif chart_type == "Scatter Plot":
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title,
                         color_discrete_sequence=COLOR_PALETTE)
        fig.update_traces(marker=dict(size=10, opacity=0.8))
        
    # 4. Histogram
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_col, title=title,
                           color_discrete_sequence=COLOR_PALETTE,
                           nbins=30)
        
    # 5. Pie Chart
    elif chart_type == "Pie Chart":
        # For pie chart, if y_col is provided, we sum it, else we count occurrences
        if y_col:
            grouped = df.groupby(x_col)[y_col].sum().reset_index()
            fig = px.pie(grouped, names=x_col, values=y_col, title=title,
                         color_discrete_sequence=COLOR_PALETTE)
        else:
            grouped = df[x_col].value_counts().reset_index()
            grouped.columns = [x_col, 'count']
            fig = px.pie(grouped, names=x_col, values='count', title=title,
                         color_discrete_sequence=COLOR_PALETTE)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
    # 6. Box Plot
    elif chart_type == "Box Plot":
        fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title,
                     color_discrete_sequence=COLOR_PALETTE)
        
    # 7. Heatmap (Correlation)
    elif chart_type == "Heatmap":
        # Calculate correlation for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr = numeric_df.corr().round(2)
            fig = px.imshow(corr, text_auto=True, 
                            color_continuous_scale='RdBu_r', 
                            aspect="auto",
                            title=title)
        else:
            # Empty fallback
            fig = go.Figure()
            fig.update_layout(title="No numeric columns available for heatmap")
            
    # Apply professional styling to Plotly figures
    if fig:
        fig.update_layout(
            font_family="Plus Jakarta Sans, sans-serif",
            title_font_family="Outfit, sans-serif",
            title_font_size=20,
            title_font_color="#ffffff",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            font_color="#94a3b8",
            legend_title_font_color="#ffffff",
            legend_font_color="#cbd5e1",
            margin=dict(l=60, r=40, t=80, b=60),
            hovermode="closest",
        )
        
        # Style axes (except for Heatmap/Pie)
        if chart_type != "Pie Chart" and chart_type != "Heatmap":
            fig.update_xaxes(
                title_text=x_label or x_col,
                title_font_color="#ffffff",
                gridcolor="#1e293b" if show_grid else "rgba(0,0,0,0)",
                zerolinecolor="#334155",
                linecolor="#334155"
            )
            fig.update_yaxes(
                title_text=y_label or y_col,
                title_font_color="#ffffff",
                gridcolor="#1e293b" if show_grid else "rgba(0,0,0,0)",
                zerolinecolor="#334155",
                linecolor="#334155"
            )
            
    return fig

def generate_matplotlib_chart(
    df: pd.DataFrame, 
    chart_type: str, 
    x_col: str, 
    y_col: Optional[str] = None, 
    color_col: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    show_grid: bool = True
) -> Tuple[plt.Figure, bytes]:
    """
    Generates a beautifully styled static Matplotlib/Seaborn figure for downloading as PNG.
    Returns: (Figure, bytes)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pre-configure labels
    x_label = x_label or x_col
    y_label = y_label or (y_col if y_col else "Count")
    
    # 1. Bar Chart
    if chart_type == "Bar Chart":
        if color_col:
            sns.barplot(data=df, x=x_col, y=y_col, hue=color_col, ax=ax, palette=COLOR_PALETTE)
        else:
            sns.barplot(data=df, x=x_col, y=y_col, ax=ax, color=COLOR_PALETTE[0])
            
    # 2. Line Chart
    elif chart_type == "Line Chart":
        if color_col:
            sns.lineplot(data=df, x=x_col, y=y_col, hue=color_col, ax=ax, palette=COLOR_PALETTE, linewidth=2.5)
        else:
            sns.lineplot(data=df, x=x_col, y=y_col, ax=ax, color=COLOR_PALETTE[0], linewidth=2.5)
            
    # 3. Scatter Plot
    elif chart_type == "Scatter Plot":
        if color_col:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=color_col, ax=ax, palette=COLOR_PALETTE, s=80, alpha=0.8)
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, color=COLOR_PALETTE[0], s=80, alpha=0.8)
            
    # 4. Histogram
    elif chart_type == "Histogram":
        sns.histplot(data=df, x=x_col, ax=ax, bins=30, color=COLOR_PALETTE[1], kde=True)
        y_label = "Frequency"
        
    # 5. Pie Chart
    elif chart_type == "Pie Chart":
        if y_col:
            grouped = df.groupby(x_col)[y_col].sum().reset_index()
            labels = grouped[x_col].tolist()
            sizes = grouped[y_col].tolist()
        else:
            grouped = df[x_col].value_counts().reset_index()
            labels = grouped.iloc[:, 0].tolist()
            sizes = grouped.iloc[:, 1].tolist()
            
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=COLOR_PALETTE[:len(labels)],
               startangle=90, textprops={'color': 'white', 'fontsize': 10},
               wedgeprops={'edgecolor': '#0f172a', 'linewidth': 1.5})
        ax.axis('equal')
        
    # 6. Box Plot
    elif chart_type == "Box Plot":
        if color_col:
            sns.boxplot(data=df, x=x_col, y=y_col, hue=color_col, ax=ax, palette=COLOR_PALETTE)
        else:
            sns.boxplot(data=df, x=x_col, y=y_col, ax=ax, palette=COLOR_PALETTE)
            
    # 7. Heatmap
    elif chart_type == "Heatmap":
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr = numeric_df.corr().round(2)
            sns.heatmap(corr, annot=True, cmap='RdBu_r', ax=ax, fmt=".2f",
                        vmin=-1, vmax=1, center=0, cbar=True,
                        annot_kws={'size': 10, 'weight': 'bold'})
            # Adjust label rotations
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            ax.text(0.5, 0.5, "No numeric columns\nfor correlation matrix", 
                    ha='center', va='center', color='white', fontsize=14)
            
    # Professional general styles
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=20, fontfamily='sans-serif')
    
    if chart_type != "Pie Chart" and chart_type != "Heatmap":
        ax.set_xlabel(x_label, fontsize=12, color='#94a3b8', labelpad=10)
        ax.set_ylabel(y_label, fontsize=12, color='#94a3b8', labelpad=10)
        ax.tick_params(colors='#94a3b8', labelsize=10)
        
        # Grid settings
        if show_grid:
            ax.grid(True, color='#1e293b', linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
        else:
            ax.grid(False)
            
        # Customize spine colors
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#334155')
            ax.spines[spine].set_linewidth(1.0)
            
        # Legend styling
        if ax.get_legend():
            legend = ax.legend(facecolor='#0f172a', edgecolor='#334155')
            for text in legend.get_texts():
                text.set_color('#cbd5e1')
                
    elif chart_type == "Heatmap":
        ax.tick_params(colors='#94a3b8', labelsize=10)
        
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, facecolor='#0f172a', bbox_inches='tight')
    buf.seek(0)
    img_bytes = buf.getvalue()
    
    # Also save to charts directory for local workspace archival
    clean_title = "".join(x for x in title if x.isalnum() or x in " -_").strip().replace(" ", "_")
    filename = f"{chart_type.lower().replace(' ', '_')}_{clean_title or 'chart'}.png"
    save_matplotlib_figure(fig, filename)
    
    return fig, img_bytes
