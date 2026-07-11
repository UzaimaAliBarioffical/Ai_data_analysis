import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import io
from dotenv import load_dotenv

# Import custom modules
import utils
import analysis
import visualization
import ai

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State variables
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "df" not in st.session_state:
    st.session_state.df = None
if "stats" not in st.session_state:
    st.session_state.stats = None
if "cleaning_report" not in st.session_state:
    st.session_state.cleaning_report = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "auto_insights" not in st.session_state:
    st.session_state.auto_insights = None
if "ai_report" not in st.session_state:
    st.session_state.ai_report = None
if "groq_api_key" not in st.session_state:
    # Load from environment initially if present
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")

# Inject premium custom CSS styles
utils.set_page_style()

# Application-level polish for a cleaner, consistent workspace.
st.markdown("""
<style>
    .stApp { background: #f8fafc; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #172554 100%); }
    [data-testid="stSidebar"] * { color: #e2e8f0; }
    [data-testid="stSidebar"] .stRadio label { border-radius: 10px; padding: 8px 10px; transition: background-color .2s ease; }
    [data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,.10); }
    .stButton > button { border: 0; border-radius: 9px; background: #2563eb; color: white; font-weight: 600; padding: .55rem 1rem; }
    .stButton > button:hover { background: #1d4ed8; color: white; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; box-shadow: 0 2px 8px rgba(15,23,42,.04); }
    .stDataFrame, [data-testid="stPlotlyChart"] { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation Panel
st.sidebar.markdown(
    """
    <div style='text-align: center; padding-bottom: 20px;'>
        <h2 style='color: #ffffff; margin-bottom: 5px; font-size: 1.8rem;'>📊 Data Analysis</h2>
        <p style='color: #bfdbfe; font-size: 0.85rem; letter-spacing: .03em;'>DATA INSIGHTS WORKSPACE</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation radio list
page = st.sidebar.radio(
    "NAVIGATION",
    options=[
        "🏠 Home",
        "📂 Upload Dataset",
        "📊 Analysis",
        "📈 Visualization",
        "🤖 AI Assistant",
        "📝 AI Report",
        "⚙ Settings"
    ],
    index=0
)

# Sidebar System Badges
st.sidebar.markdown("<br><hr style='border-color: #1e293b;'><br>", unsafe_allow_html=True)

# 1. API Status Badge
has_api_key = bool(st.session_state.groq_api_key or os.getenv("GROQ_API_KEY"))
if has_api_key:
    st.sidebar.markdown(
        """
        <div style='padding: 10px; background-color: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; text-align: center;'>
            <span style='color: #10b981; font-weight: 600; font-size: 0.85rem;'>Groq API: Connected 🟢</span>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        """
        <div style='padding: 10px; background-color: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; text-align: center;'>
            <span style='color: #ef4444; font-weight: 600; font-size: 0.85rem;'>Groq API: Config Required 🔴</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.sidebar.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

# 2. Dataset Status Badge
if st.session_state.df is not None:
    rows, cols = st.session_state.df.shape
    st.sidebar.markdown(
        f"""
        <div style='padding: 10px; background-color: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 8px;'>
            <div style='color: #6366f1; font-weight: 600; font-size: 0.85rem; text-align: center; margin-bottom: 5px;'>Dataset Loaded 🟢</div>
            <div style='color: #cbd5e1; font-size: 0.75rem; text-align: center;'>Rows: {rows} | Cols: {cols}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        """
        <div style='padding: 10px; background-color: rgba(148, 163, 184, 0.1); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 8px; text-align: center;'>
            <span style='color: #94a3b8; font-weight: 600; font-size: 0.85rem;'>No Dataset Active 🔴</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------- PAGE 1: HOME -----------------
if page == "🏠 Home":
    st.markdown("<h1 style='color: #ffffff;'>🏠 Dashboard Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Welcome to the next-generation AI-powered Data Analysis Assistant. This application automates data cleaning, builds responsive visualizations, computes deep profiles, and answers complex business questions using the Groq Llama-3.3 LLM.</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="metric-card-container" style="height: 250px;">
                <h3 style="color: #6366f1; margin-bottom: 12px;">📂 Automated Loading & Cleaning</h3>
                <p style="color: #cbd5e1; font-size: 0.9rem;">Upload CSV tables and automatically run duplicate checks, handle missing values via custom imputation strategies (mean, median, mode), and repair column types with simple toggles.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            """
            <div class="metric-card-container" style="height: 250px;">
                <h3 style="color: #06b6d4; margin-bottom: 12px;">📈 Interactive Visualizer</h3>
                <p style="color: #cbd5e1; font-size: 0.9rem;">Generate responsive Bar charts, Line charts, Scatter plots, Histograms, Pie charts, Box plots, and Correlation Heatmaps. Tweak headers, legends, and grids, and export in 300-DPI publication PNG formats.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            """
            <div class="metric-card-container" style="height: 250px;">
                <h3 style="color: #ec4899; margin-bottom: 12px;">🤖 Advanced Groq Chatbot</h3>
                <p style="color: #cbd5e1; font-size: 0.9rem;">Query your dataset in natural language. Our contextual engine parses your dataset, prepares mathematical summaries, and feeds them into Groq LLM to answer complex analytical queries instantly.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Demo dataset section
    st.markdown("### ⚡ Quick Start Demonstration")
    st.markdown(
        """
        <div class="custom-info-box">
            <span style="font-weight: bold; color: #6366f1;">Tip:</span> Don't have a dataset on hand? Click the button below to instantly load a pre-packaged <strong>Mock Sales & Customer Dataset</strong>. You can use it to test all the analysis, visualization, and AI features immediately!
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("🚀 Load Mock Dataset"):
        mock_path = os.path.join(os.getcwd(), 'assets', 'mock_sales_data.csv')
        if os.path.exists(mock_path):
            try:
                df = pd.read_csv(mock_path)
                st.session_state.raw_df = df.copy()
                st.session_state.df = df.copy()
                st.session_state.stats = analysis.calculate_statistics(df)
                st.session_state.cleaning_report = None
                st.session_state.chat_history = []
                st.session_state.auto_insights = None
                st.session_state.ai_report = None
                st.success("Mock dataset loaded successfully! Head over to the 'Analysis' or 'Visualization' tabs to explore.")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading mock dataset: {str(e)}")
        else:
            st.error(f"Mock dataset file not found at path: {mock_path}")
            
    utils.render_footer()

# ----------------- PAGE 2: UPLOAD DATASET -----------------
elif page == "📂 Upload Dataset":
    st.markdown("<h1 style='color: #ffffff;'>📂 Upload & Clean Dataset</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Upload your tabular data, review schema metrics, configure custom imputation parameters, and clean the data.</p>", unsafe_allow_html=True)
    
    # 1. File Uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        is_valid, msg, loaded_df = analysis.validate_csv(uploaded_file)
        
        if not is_valid:
            st.markdown(f'<div class="custom-error-box">⚠️ {msg}</div>', unsafe_allow_html=True)
        else:
            # Check if this is a newly uploaded file to clear session states
            if st.session_state.raw_df is None or not st.session_state.raw_df.equals(loaded_df):
                st.session_state.raw_df = loaded_df.copy()
                st.session_state.df = loaded_df.copy()
                st.session_state.stats = analysis.calculate_statistics(loaded_df)
                st.session_state.cleaning_report = None
                st.session_state.chat_history = []
                st.session_state.auto_insights = None
                st.session_state.ai_report = None
                st.markdown(f'<div class="custom-success-box">🎉 {msg}</div>', unsafe_allow_html=True)
                st.rerun()

    # If dataset is loaded, show cleaning options and preview
    if st.session_state.df is not None:
        st.markdown("<br><hr style='border-color: #1e293b;'><br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ✨ Data Cleaning Options")
            st.markdown("Configure the cleaning routines below and hit the button to apply changes.")
            
            # Cleaning configuration options
            remove_dups = st.checkbox("Remove Duplicate Rows", value=True)
            convert_dt = st.checkbox("Auto Convert Date/Time columns", value=True)
            
            num_strategy = st.selectbox(
                "Imputation Strategy for Numeric columns",
                options=["median", "mean", "zero", "drop", "none"],
                index=0,
                format_func=lambda x: {
                    "median": "Fill with Median",
                    "mean": "Fill with Mean",
                    "zero": "Fill with 0",
                    "drop": "Drop rows with missing numericals",
                    "none": "Keep missing values"
                }[x]
            )
            
            cat_strategy = st.selectbox(
                "Imputation Strategy for Categorical columns",
                options=["mode", "placeholder", "drop", "none"],
                index=0,
                format_func=lambda x: {
                    "mode": "Fill with Mode",
                    "placeholder": "Fill with 'Unknown'",
                    "drop": "Drop rows with missing categories",
                    "none": "Keep missing values"
                }[x]
            )
            
            # Clean Button
            if st.button("✨ Clean Dataset", use_container_width=True):
                with st.spinner("Executing cleaning routines..."):
                    cleaned, report = analysis.clean_dataset(
                        st.session_state.raw_df, 
                        remove_duplicates=remove_dups, 
                        fill_numeric=num_strategy, 
                        fill_categorical=cat_strategy,
                        convert_dates=convert_dt
                    )
                    st.session_state.df = cleaned
                    st.session_state.stats = analysis.calculate_statistics(cleaned)
                    st.session_state.cleaning_report = report
                    # Clear insights because dataset updated
                    st.session_state.auto_insights = None
                    st.session_state.ai_report = None
                    st.success("Dataset cleaned successfully!")
                    st.rerun()
            
            # Reset Button
            if st.button("🔄 Reset to Raw Dataset", use_container_width=True):
                st.session_state.df = st.session_state.raw_df.copy()
                st.session_state.stats = analysis.calculate_statistics(st.session_state.raw_df)
                st.session_state.cleaning_report = None
                st.session_state.auto_insights = None
                st.session_state.ai_report = None
                st.success("Reverted to raw, original CSV data.")
                st.rerun()

        with col2:
            st.markdown("### 📊 Dataset Profile Summary")
            
            # Display cleaning report if executed
            if st.session_state.cleaning_report:
                rep = st.session_state.cleaning_report
                st.markdown(
                    f"""
                    <div style="background-color: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.2); padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                        <h4 style="color: #6366f1; margin: 0 0 10px 0;">Cleaning Execution Log</h4>
                        <ul style="color: #cbd5e1; font-size: 0.88rem; margin: 0; padding-left: 20px;">
                            <li>Original Shape: {rep['original_shape'][0]} rows, {rep['original_shape'][1]} cols</li>
                            <li>Final Cleaned Shape: {rep['final_shape'][0]} rows, {rep['final_shape'][1]} cols</li>
                            <li>Duplicate rows removed: <strong>{rep['duplicates_removed']}</strong></li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Missing values handled summary
                if rep['missing_filled']:
                    st.markdown("**Imputed Missing Values:**")
                    for col, action in rep['missing_filled'].items():
                        st.markdown(f"- Column `{col}`: {action}")
                
                # Conversions summary
                if rep['type_conversions']:
                    st.markdown("**Type Conversions:**")
                    for conversion in rep['type_conversions']:
                        st.markdown(f"- {conversion}")
            else:
                # Default profile general facts
                stats = st.session_state.stats
                rows, cols = st.session_state.df.shape
                mem = utils.format_bytes(stats["memory_usage_bytes"])
                
                st.markdown(
                    f"""
                    * **Total Records:** {rows} rows
                    * **Total Dimensions:** {cols} columns
                    * **Duplicates Detected:** {stats['duplicate_rows']}
                    * **Dataset Size on disk/RAM:** {mem}
                    """
                )
                
                # Missing values indicator
                null_cols = {k: v for k, v in stats['missing_values'].items() if v > 0}
                if null_cols:
                    st.markdown("**Columns containing missing values:**")
                    for col, count in null_cols.items():
                        pct = stats['missing_percentage'][col]
                        st.markdown(f"- `{col}`: {count} missing rows ({pct}%)")
                else:
                    st.markdown("✨ No missing values detected in the raw dataset.")

        # Large Data Preview Section at the bottom
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔍 Dataset Preview")
        
        tab_head, tab_tail, tab_types = st.tabs(["First 10 Rows", "Last 10 Rows", "Schema & Data Types"])
        
        with tab_head:
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
            
        with tab_tail:
            st.dataframe(st.session_state.df.tail(10), use_container_width=True)
            
        with tab_types:
            stats = st.session_state.stats
            types_df = pd.DataFrame({
                "Column Name": stats["columns"],
                "Data Type": [stats["dtypes"][col] for col in stats["columns"]],
                "Null Values Count": [stats["missing_values"][col] for col in stats["columns"]],
                "Null Percentage (%)": [stats["missing_percentage"][col] for col in stats["columns"]]
            })
            st.dataframe(types_df, use_container_width=True, hide_index=True)

    else:
        st.info("Please upload a CSV dataset or load the mock dataset on the Home page to start.")
        
    utils.render_footer()

# ----------------- PAGE 3: ANALYSIS -----------------
elif page == "📊 Analysis":
    st.markdown("<h1 style='color: #ffffff;'>📊 Statistical Analysis Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Compute high-fidelity descriptive metrics, correlation values, and value distribution profiles for the active dataset.</p>", unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        stats = st.session_state.stats
        
        # 1. Dashboard KPI Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        
        rows, cols = df.shape
        dups = stats["duplicate_rows"]
        mem = utils.format_bytes(stats["memory_usage_bytes"])
        
        # Calculate overall missing value %
        total_elements = rows * cols
        total_missing = sum(stats["missing_values"].values())
        missing_pct_overall = (total_missing / total_elements * 100) if total_elements > 0 else 0
        
        with col1:
            utils.render_metric_card("Total Rows", f"{rows:,}", icon="📊", accent_color="#6366f1")
        with col2:
            utils.render_metric_card("Total Columns", f"{cols}", icon="📐", accent_color="#06b6d4")
        with col3:
            utils.render_metric_card("Duplicate Rows", f"{dups}", icon="📋", is_positive=(dups == 0), accent_color="#ec4899")
        with col4:
            utils.render_metric_card("Missing Values", f"{missing_pct_overall:.2f}%", icon="⚠️", is_positive=(missing_pct_overall < 5.0), accent_color="#f59e0b")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabs for analytical groupings
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔢 Numerical Summary", 
            "🔤 Categorical Distribution", 
            "🔗 Correlation Matrix", 
            "📉 Data Quality Profiler"
        ])
        
        # TAB 1: Numerical Summary
        with tab1:
            if stats["numeric_summary"]:
                num_df = pd.DataFrame(stats["numeric_summary"]).T
                # Reorder columns for clean visualization
                cols_order = ["count", "unique_values", "mean", "median", "mode", "min", "max", "variance", "std_dev", "null_percentage"]
                num_df = num_df[cols_order]
                num_df.columns = ["Count", "Unique Values", "Mean", "Median", "Mode", "Minimum", "Maximum", "Variance", "Std Dev", "Null %"]
                st.dataframe(num_df.style.format("{:.2f}", na_rep="-"), use_container_width=True)
            else:
                st.info("No numerical columns found in this dataset.")
                
        # TAB 2: Categorical Distribution
        with tab2:
            if stats["categorical_summary"]:
                cat_cols = list(stats["categorical_summary"].keys())
                selected_cat = st.selectbox("Select Categorical Column to explore", options=cat_cols)
                
                if selected_cat:
                    summary = stats["categorical_summary"][selected_cat]
                    
                    col_det1, col_det2 = st.columns([1, 2])
                    
                    with col_det1:
                        st.markdown(
                            f"""
                            <div class="metric-card-container">
                                <h4 style="color: #6366f1; margin-top:0;">Column: '{selected_cat}'</h4>
                                <hr style="border-color: #1e293b; margin: 10px 0;">
                                <p><strong>Unique values count:</strong> {summary['unique_values']}</p>
                                <p><strong>Top Category:</strong> '{summary['top_category']}' ({summary['top_category_count']} occurrences)</p>
                                <p><strong>Bottom Category:</strong> '{summary['bottom_category']}' ({summary['bottom_category_count']} occurrences)</p>
                                <p><strong>Null elements:</strong> {summary['null_count']} ({summary['null_percentage']:.2f}%)</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                    with col_det2:
                        st.markdown(f"**Top 10 Value Distribution for '{selected_cat}':**")
                        dist_df = pd.DataFrame({
                            "Category": list(summary["distribution"].keys()),
                            "Count": list(summary["distribution"].values())
                        })
                        
                        # Plotly pie chart of this category
                        fig = px.pie(dist_df, names="Category", values="Count", 
                                     color_discrete_sequence=visualization.COLOR_PALETTE)
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#cbd5e1",
                            margin=dict(l=0, r=0, t=30, b=0),
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns found in this dataset.")
                
        # TAB 3: Correlation Matrix
        with tab3:
            if stats["correlation_matrix"]:
                corr_df = pd.DataFrame(stats["correlation_matrix"])
                st.markdown("**Pearson's Correlation Coefficient Table:**")
                st.dataframe(corr_df.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.4f}"), use_container_width=True)
                
                # Render heatmap
                st.markdown("<br>**Correlation Matrix Heatmap Visualizer:**", unsafe_allow_html=True)
                fig = visualization.generate_plotly_chart(df, "Heatmap", x_col="", title="Correlation Coefficient Heatmap")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numerical columns to calculate correlation matrix.")
                
        # TAB 4: Data Quality Profiler
        with tab4:
            st.markdown("**Null Values Count and Percentage Distribution per Column:**")
            
            null_data = pd.DataFrame({
                "Column": stats["columns"],
                "Missing Rows": [stats["missing_values"][col] for col in stats["columns"]],
                "Null Percentage (%)": [stats["missing_percentage"][col] for col in stats["columns"]]
            })
            
            col_q1, col_q2 = st.columns([1, 1])
            
            with col_q1:
                st.dataframe(null_data.style.bar(subset=["Null Percentage (%)"], color="#ef4444").format({"Null Percentage (%)": "{:.2f}%"}), use_container_width=True, hide_index=True)
                
            with col_q2:
                # Plotly bar chart for missing percentage
                null_data_filtered = null_data[null_data["Missing Rows"] > 0]
                if not null_data_filtered.empty:
                    fig = px.bar(null_data_filtered, x="Column", y="Null Percentage (%)", 
                                 title="Columns with Missing Values (%)",
                                 color_discrete_sequence=['#ef4444'])
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#cbd5e1",
                        margin=dict(l=20, r=20, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("Clean dataset! 100% complete records. No null cells found.")
    else:
        st.info("Please upload a CSV dataset or load the mock dataset on the Home page to start.")
        
    utils.render_footer()

# ----------------- PAGE 4: VISUALIZATION -----------------
elif page == "📈 Visualization":
    st.markdown("<h1 style='color: #ffffff;'>📈 Data Visualization Workshop</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Construct bespoke interactive charts and compile detailed explanations. Export assets in high-definition formats.</p>", unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        stats = st.session_state.stats
        
        col_cfg, col_plot = st.columns([1, 2])
        
        with col_cfg:
            st.markdown("### ⚙️ Chart Configuration")
            
            chart_type = st.selectbox(
                "Select Chart Type",
                options=["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Pie Chart", "Box Plot", "Heatmap"]
            )
            
            # Select column dropdowns
            cols = stats["columns"]
            num_cols = stats["numeric_cols"]
            
            x_col = st.selectbox("Select X-Axis Column", options=cols, index=0)
            
            # Logic for Y-axis (Pie and Histogram do not strictly need Y-axis)
            y_col = None
            if chart_type not in ["Histogram", "Heatmap"]:
                # Pie chart is optional
                if chart_type == "Pie Chart":
                    y_opts = ["None"] + num_cols
                    selected_y = st.selectbox("Select Slice Value Column (Optional)", options=y_opts)
                    y_col = None if selected_y == "None" else selected_y
                else:
                    y_col = st.selectbox("Select Y-Axis Column", options=num_cols, index=min(1, len(num_cols)-1))
                    
            # Color grouping selection
            color_col = None
            if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot"]:
                color_opts = ["None"] + cols
                selected_color = st.selectbox("Select Color Grouping Column (Optional)", options=color_opts)
                color_col = None if selected_color == "None" else selected_color
                
            st.markdown("<br>**Chart Labels & Styling**", unsafe_allow_html=True)
            chart_title = st.text_input("Chart Title", value=f"{chart_type} of {x_col}" + (f" vs {y_col}" if y_col else ""))
            x_lbl = st.text_input("X-Axis Custom Label", value=x_col)
            y_lbl = ""
            if chart_type not in ["Histogram", "Heatmap"]:
                y_lbl = st.text_input("Y-Axis Custom Label", value=y_col if y_col else "Count")
                
            show_grid = st.checkbox("Show Grid Lines", value=True)
            
        with col_plot:
            st.markdown("### 📊 Interactive Visualization Render")
            
            # Generate and render Interactive Plotly Chart
            plotly_fig = visualization.generate_plotly_chart(
                df=df,
                chart_type=chart_type,
                x_col=x_col,
                y_col=y_col,
                color_col=color_col,
                title=chart_title,
                x_label=x_lbl,
                y_label=y_lbl,
                show_grid=show_grid
            )
            
            st.plotly_chart(plotly_fig, use_container_width=True)
            
            # Render Static Chart for Download (using Matplotlib backend)
            fig, img_bytes = visualization.generate_matplotlib_chart(
                df=df,
                chart_type=chart_type,
                x_col=x_col,
                y_col=y_col,
                color_col=color_col,
                title=chart_title,
                x_label=x_lbl,
                y_label=y_lbl,
                show_grid=show_grid
            )
            
            # Download Button Container
            st.markdown('<div class="download-btn-container">', unsafe_allow_html=True)
            st.download_button(
                label="📥 Download Chart as PNG (300 DPI)",
                data=img_bytes,
                file_name=f"{chart_type.lower().replace(' ', '_')}_{x_col.lower()}.png",
                mime="image/png",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # AI Chart Explanation Section (PDF requirement step 5)
            st.markdown("<br><hr style='border-color: #1e293b;'><br>", unsafe_allow_html=True)
            st.markdown("### 🤖 AI Chart Interpretation")
            
            if not has_api_key:
                st.warning("Please configure your Groq API Key in settings or .env file to generate AI chart explanations.")
            else:
                if st.button("🪄 Ask AI to Interpret Chart"):
                    with st.spinner("Analyzing chart details..."):
                        # Build specific question for chart explanation
                        chart_desc = f"I have generated a {chart_type} where X-axis is '{x_col}'"
                        if y_col:
                            chart_desc += f" and Y-axis is '{y_col}'"
                        if color_col:
                            chart_desc += f", grouped by the '{color_col}' column."
                        chart_desc += f" The title is '{chart_title}'. "
                        
                        question = (
                            f"{chart_desc} Provide a clear, short, and highly understandable business explanation of what this chart represents. "
                            f"Focus on telling the user the core takeaway, percentages, or trends shown in the data."
                        )
                        
                        explanation = ai.ask_ai_assistant(df, stats, question)
                        st.markdown(
                            f"""
                            <div class="custom-info-box" style="margin-top: 15px;">
                                <h4 style="color: #6366f1; margin: 0 0 10px 0; display:flex; align-items:center;">
                                    <span style="margin-right: 8px;">🤖</span> AI Explanation
                                </h4>
                                <div style="color: #f1f5f9; font-size: 0.95rem; line-height:1.6;">
                                    {explanation}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
    else:
        st.info("Please upload a CSV dataset or load the mock dataset on the Home page to start.")
        
    utils.render_footer()

# ----------------- PAGE 5: AI ASSISTANT -----------------
elif page == "🤖 AI Assistant":
    st.markdown("<h1 style='color: #ffffff;'>🤖 Interactive AI Assistant Chat</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Query dataset statistics, ask complex mathematical questions, and retrieve localized counts in natural language. Powered by Groq Llama-3.3.</p>", unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        stats = st.session_state.stats
        
        # Check API key configuration
        if not has_api_key:
            st.error("Groq API Key is not configured. Please configure it in the 'Settings' tab or inside a local '.env' file to communicate with the model.")
        else:
            # 1. Quick suggestion chips/buttons
            st.markdown("**💡 Quick suggestion prompts:**")
            quick_cols = st.columns(3)
            
            # Choose queries dynamically based on columns
            num_cols = stats["numeric_cols"]
            cat_cols = stats["categorical_cols"]
            
            q1 = "What is the general overview of this dataset?"
            q2 = f"Calculate the average values for numeric columns."
            q3 = "Which category appears most frequently?"
            
            if len(num_cols) > 0:
                q2 = f"What is the average, minimum, and maximum value for '{num_cols[0]}'?"
            if len(cat_cols) > 0:
                q3 = f"Which value appears most frequently in '{cat_cols[0]}', and what is its count?"
                
            with quick_cols[0]:
                if st.button(f"📋 {q1}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": q1})
                    with st.spinner("Groq is typing..."):
                        ans = ai.ask_ai_assistant(df, stats, q1)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})
                    st.rerun()
                    
            with quick_cols[1]:
                if st.button(f"📈 {q2}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": q2})
                    with st.spinner("Groq is typing..."):
                        ans = ai.ask_ai_assistant(df, stats, q2)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})
                    st.rerun()
                    
            with quick_cols[2]:
                if st.button(f"🏷️ {q3}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": q3})
                    with st.spinner("Groq is typing..."):
                        ans = ai.ask_ai_assistant(df, stats, q3)
                        st.session_state.chat_history.append({"role": "assistant", "content": ans})
                    st.rerun()
            
            st.markdown("<br><hr style='border-color: #1e293b;'><br>", unsafe_allow_html=True)
            
            # Display Chat History
            chat_container = st.container()
            with chat_container:
                for msg_item in st.session_state.chat_history:
                    role = msg_item["role"]
                    content = msg_item["content"]
                    
                    if role == "user":
                        st.markdown(
                            f"""
                            <div class="chat-bubble user">
                                <strong>You:</strong><br>{content}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="chat-bubble assistant">
                                <strong>🤖 Assistant:</strong><br>{content}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            
            # Chat input
            user_input = st.chat_input("Ask a question about the dataset (e.g. 'Which city has the maximum orders?')")
            
            if user_input:
                # Add user input
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Fetch response
                with st.spinner("Groq is thinking..."):
                    response = ai.ask_ai_assistant(df, stats, user_input)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
                
            # Clear chat button
            if st.session_state.chat_history:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🧹 Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
    else:
        st.info("Please upload a CSV dataset or load the mock dataset on the Home page to start.")
        
    utils.render_footer()

# ----------------- PAGE 6: AI REPORT -----------------
elif page == "📝 AI Report":
    st.markdown("<h1 style='color: #ffffff;'>📝 AI Automated Reports</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Compile auto-generated diagnostic profiles, business intelligence findings, and operational executive reports based on dataset statistics.</p>", unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        stats = st.session_state.stats
        
        if not has_api_key:
            st.error("Groq API Key is not configured. Please configure it in Settings or .env file to generate reports.")
        else:
            rep_tab1, rep_tab2 = st.tabs(["✨ Automated Insights", "📄 Executive Summary Report"])
            
            # TAB 1: Automated Insights
            with rep_tab1:
                st.markdown("### 🔮 Data Outliers & Pattern Insights")
                st.markdown("Click the button below to trigger real-time AI modeling. This will parse statistical properties, evaluate variance behaviors, locate outliers, and formulate business suggestions.")
                
                if st.button("🪄 Generate Automated Insights", use_container_width=True):
                    with st.spinner("Compiling automated insights from Groq..."):
                        insights = ai.generate_auto_insights(df, stats)
                        st.session_state.auto_insights = insights
                
                if st.session_state.auto_insights:
                    st.markdown("<br><hr style='border-color: #1e293b;'><br>", unsafe_allow_html=True)
                    st.markdown(st.session_state.auto_insights)
                    
                    # Download option
                    st.markdown('<div class="download-btn-container">', unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download Insights as TXT",
                        data=st.session_state.auto_insights,
                        file_name="dataset_ai_insights.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            # TAB 2: Executive Report
            with rep_tab2:
                st.markdown("### 📄 Production Executive Summary Report")
                st.markdown("Generate a structured, printable report detailing general quality assessments, key discovery points, statistical summaries, operational recommendations, and summaries.")
                
                if st.button("📄 Compile Executive Report", use_container_width=True):
                    with st.spinner("Compiling comprehensive analytics report..."):
                        report_text = ai.generate_full_report(df, stats)
                        st.session_state.ai_report = report_text
                        
                if st.session_state.ai_report:
                    st.markdown("<br><hr style='border-color: #1e293b;'><br>", unsafe_allow_html=True)
                    st.markdown(st.session_state.ai_report)
                    
                    # Download option
                    st.markdown('<div class="download-btn-container">', unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download Executive Report as TXT",
                        data=st.session_state.ai_report,
                        file_name="executive_data_analysis_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Please upload a CSV dataset or load the mock dataset on the Home page to start.")
        
    utils.render_footer()

# ----------------- PAGE 7: SETTINGS -----------------
elif page == "⚙ Settings":
    st.markdown("<h1 style='color: #ffffff;'>⚙ Settings & Configuration</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Adjust variables, verify Groq API configurations, and purge cache values.</p>", unsafe_allow_html=True)
    
    st.markdown("### 🔑 Groq SDK Configuration")
    
    # Text input for Groq API Key
    st.session_state.groq_api_key = st.text_input(
        "Groq API Key", 
        value=st.session_state.groq_api_key, 
        type="password",
        help="Input your Groq API key (starts with 'gsk_'). This is kept inside the session memory."
    )
    
    col_set1, col_set2 = st.columns([1, 1])
    
    with col_set1:
        if st.button("🔌 Test API Connectivity", use_container_width=True):
            if not st.session_state.groq_api_key:
                st.error("Please enter an API Key first.")
            else:
                with st.spinner("Verifying connection to Groq API..."):
                    try:
                        from groq import Groq
                        test_client = Groq(api_key=st.session_state.groq_api_key)
                        test_comp = test_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": "Return the word 'Success'."}],
                            max_tokens=10
                        )
                        result = test_comp.choices[0].message.content.strip()
                        if "Success" in result or result:
                            st.success("API Connectivity Test Passed! Connected to llama-3.3-70b-versatile model.")
                        else:
                            st.error(f"Unexpected response from API: {result}")
                    except Exception as e:
                        st.error(f"API Connectivity Test Failed! Details: {str(e)}")
                        
    with col_set2:
        if st.button("🧹 Clear All App State", use_container_width=True):
            st.session_state.raw_df = None
            st.session_state.df = None
            st.session_state.stats = None
            st.session_state.cleaning_report = None
            st.session_state.chat_history = []
            st.session_state.auto_insights = None
            st.session_state.ai_report = None
            st.success("Session state cleared completely. You can now load a new dataset.")
            st.rerun()
            
    utils.render_footer()
