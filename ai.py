import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()

def get_groq_client() -> Optional[Groq]:
    """
    Initializes and returns the Groq client.
    First checks environment variables, then falls back to session state.
    """
    # Try fetching from environment first
    api_key = os.getenv("GROQ_API_KEY")
    
    # Fallback to streamlit session state if available
    if not api_key:
        try:
            import streamlit as st
            if "groq_api_key" in st.session_state and st.session_state.groq_api_key:
                api_key = st.session_state.groq_api_key
        except Exception:
            pass
            
    if not api_key:
        return None
        
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None

def build_dataset_context(df: pd.DataFrame, stats: Dict[str, Any]) -> str:
    """
    Creates a comprehensive textual summary of the dataset to be sent in the prompt.
    Includes shape, columns, types, missing values, descriptive statistics, and first 20 rows.
    """
    # 1. Dataset overview
    shape_info = f"Shape: {stats['row_count']} rows, {stats['col_count']} columns\n"
    duplicate_info = f"Duplicate Rows: {stats['duplicate_rows']}\n"
    memory_info = f"Memory Usage: {stats['memory_usage_bytes']} bytes\n"
    
    # 2. Columns & Data Types & Nulls
    col_details = "Columns details:\n"
    for col in stats['columns']:
        dtype = stats['dtypes'][col]
        null_count = stats['missing_values'][col]
        null_pct = stats['missing_percentage'][col]
        col_details += f"  - Name: '{col}', Type: {dtype}, Nulls: {null_count} ({null_pct}%)\n"
        
    # 3. Numeric Summary
    num_summary = "Numerical Columns Statistics:\n"
    if stats['numeric_summary']:
        for col, summary in stats['numeric_summary'].items():
            num_summary += (
                f"  - '{col}': Mean: {summary['mean']:.2f}, Median: {summary['median']:.2f}, "
                f"Min: {summary['min']:.2f}, Max: {summary['max']:.2f}, Std Dev: {summary['std_dev']:.2f}, "
                f"Unique Values: {summary['unique_values']}\n"
            )
    else:
        num_summary += "  - No numeric columns found.\n"
        
    # 4. Categorical Summary
    cat_summary = "Categorical Columns Statistics:\n"
    if stats['categorical_summary']:
        for col, summary in stats['categorical_summary'].items():
            cat_summary += (
                f"  - '{col}': Unique Values: {summary['unique_values']}, "
                f"Top: '{summary['top_category']}' (Count: {summary['top_category_count']}), "
                f"Bottom: '{summary['bottom_category']}' (Count: {summary['bottom_category_count']})\n"
                f"    Top distribution: {summary['distribution']}\n"
            )
    else:
        cat_summary += "  - No categorical columns found.\n"
        
    # 5. First 20 Rows
    # Convert first 20 rows to markdown or CSV string
    first_20_df = df.head(20)
    first_20_str = first_20_df.to_csv(index=False)
    
    context = (
        "=======================================\n"
        "DATASET METADATA & CONTEXT\n"
        "=======================================\n"
        f"{shape_info}"
        f"{duplicate_info}"
        f"{memory_info}\n"
        f"{col_details}\n"
        f"{num_summary}\n"
        f"{cat_summary}\n"
        "=======================================\n"
        "FIRST 20 ROWS OF DATASET (CSV format):\n"
        "=======================================\n"
        f"{first_20_str}\n"
        "=======================================\n"
    )
    return context

def ask_ai_assistant(df: pd.DataFrame, stats: Dict[str, Any], query: str) -> str:
    """
    Answers a natural language query based ONLY on the provided dataset context.
    """
    client = get_groq_client()
    if not client:
        return "Error: Groq API client could not be initialized. Please check your GROQ_API_KEY in settings or .env file."
        
    context = build_dataset_context(df, stats)
    
    system_prompt = (
        "You are an expert AI Data Analysis Assistant.\n"
        "You are given a context describing a dataset (its shape, column names, data types, missing values, descriptive statistics, and the first 20 rows).\n"
        "Your task is to answer the user's question about the dataset based STRICTLY on the context provided.\n\n"
        "CRITICAL RULES:\n"
        "1. Answer the question using ONLY the provided dataset details, statistics, and sample rows.\n"
        "2. Do NOT make up any information or make assumptions that cannot be directly supported by the context.\n"
        "3. If the question cannot be answered using the provided dataset details, statistics, or sample rows, you MUST reply: "
        "'The uploaded dataset does not contain enough information.'\n"
        "4. Keep your answer professional, concise, clear, and formatted in clean Markdown.\n"
        "5. If mathematical calculations are needed, explain your steps briefly based on the statistics."
    )
    
    user_prompt = f"Dataset Context:\n{context}\n\nUser Question:\n{query}"
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for factual analysis
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"API Failure: An error occurred while communicating with Groq API. Details: {str(e)}"

def generate_auto_insights(df: pd.DataFrame, stats: Dict[str, Any]) -> str:
    """
    Automatically generates dataset overview, interesting trends, highest value, lowest value,
    outliers, patterns, business insights, and recommendations.
    """
    client = get_groq_client()
    if not client:
        return "Error: Groq API client could not be initialized. Please configure your GROQ_API_KEY."
        
    context = build_dataset_context(df, stats)
    
    system_prompt = (
        "You are an expert Data Scientist and Business Intelligence Analyst.\n"
        "Analyze the provided dataset profile and generate comprehensive AI Insights.\n"
        "Your insights must be fully based on the provided dataset statistics and first 20 rows.\n\n"
        "You MUST structure your response into the following sections using clear Markdown headers:\n"
        "1. ### Dataset Overview: Briefly describe what the dataset represents, its size, and the columns.\n"
        "2. ### Key Trends & Patterns: Identify interesting correlations, distribution behaviors, or temporal patterns.\n"
        "3. ### Extremes & Outliers: Detail the highest and lowest values in crucial columns, and discuss any notable extreme data points.\n"
        "4. ### Strategic Business Insights: Convert the statistical findings into business-relevant interpretations.\n"
        "5. ### Actionable Recommendations: Provide practical recommendations that a company or manager should take based on this analysis."
    )
    
    user_prompt = f"Dataset Context for Analysis:\n{context}"
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"API Failure: An error occurred while communicating with Groq API. Details: {str(e)}"

def generate_full_report(df: pd.DataFrame, stats: Dict[str, Any]) -> str:
    """
    Generates a professional, publication-grade analytical report in text/markdown.
    """
    client = get_groq_client()
    if not client:
        return "Error: Groq API client could not be initialized. Please configure your GROQ_API_KEY."
        
    context = build_dataset_context(df, stats)
    
    system_prompt = (
        "You are a Senior Principal Data Consultant.\n"
        "Your task is to compile a formal, high-quality, professional Data Analysis Report.\n"
        "The report must contain all sections specified below, formatted beautifully in Markdown.\n\n"
        "REPORT STRUCTURE TO FOLLOW:\n"
        "# PROFESSIONAL DATA ANALYSIS EXECUTIVE REPORT\n\n"
        "## 1. Executive Summary\n"
        "Provide a high-level summary of the analysis goals, the dataset scope, and the main takeaways.\n\n"
        "## 2. Dataset Profile & Quality Assessment\n"
        "Analyze the structure, size, column data types, missing records, duplicates, and general data quality.\n\n"
        "## 3. Important Statistical Diagnostics\n"
        "Document key statistics (means, ranges, standard deviations, distributions, and correlation behaviors).\n\n"
        "## 4. Key Findings & Analytics Discoveries\n"
        "Present the core findings found within the data. Use bullet points for readability.\n\n"
        "## 5. Strategic Business Insights\n"
        "Translate statistics into strategic insights for decision-makers. Explain the 'why' behind the numbers.\n\n"
        "## 6. Recommendations & Action Plan\n"
        "Deliver clear, prioritized, actionable business or operational recommendations based on the findings.\n\n"
        "## 7. Conclusion\n"
        "Provide a final concluding statement on the dataset's utility and next steps.\n\n"
        "Make sure to display exact figures and percentages where appropriate, based strictly on the provided context."
    )
    
    user_prompt = f"Dataset Context for compiling Report:\n{context}"
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=2500
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"API Failure: An error occurred while communicating with Groq API. Details: {str(e)}"
