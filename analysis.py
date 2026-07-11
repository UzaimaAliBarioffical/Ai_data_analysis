import pandas as pd
import numpy as np
import io
from typing import Dict, Any, Tuple, List, Optional

def validate_csv(uploaded_file) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validates the uploaded CSV file.
    Returns:
        (is_valid, message, dataframe)
    """
    if uploaded_file is None:
        return False, "No file uploaded.", None
    
    if not uploaded_file.name.endswith('.csv'):
        return False, "Unsupported file format. Please upload a CSV file.", None
    
    try:
        # Seek back to 0 just in case
        uploaded_file.seek(0)
        # Read a small chunk to check if empty
        df = pd.read_csv(uploaded_file)
        if df.empty:
            return False, "The uploaded CSV file is empty.", None
        
        return True, "CSV file is valid and loaded successfully.", df
    except pd.errors.EmptyDataError:
        return False, "The uploaded CSV file contains no data.", None
    except pd.errors.ParserError as e:
        return False, f"CSV parsing error: Check file format. Details: {str(e)}", None
    except Exception as e:
        return False, f"An unexpected error occurred while reading the file: {str(e)}", None

def clean_dataset(
    df: pd.DataFrame, 
    remove_duplicates: bool = True, 
    fill_numeric: str = "median",       # "mean", "median", "drop", "zero", "none"
    fill_categorical: str = "mode",     # "mode", "placeholder", "drop", "none"
    convert_dates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans the dataset and returns the cleaned DataFrame along with a detailed cleaning report.
    """
    cleaned_df = df.copy()
    report = {
        "original_shape": df.shape,
        "duplicates_removed": 0,
        "missing_filled": {},
        "type_conversions": [],
        "errors": []
    }
    
    # 1. Remove duplicates
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        report["duplicates_removed"] = removed

    # 2. Convert date-like columns automatically
    if convert_dates:
        for col in cleaned_df.columns:
            # Check if column name contains 'date' or 'time' (case-insensitive)
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                    report["type_conversions"].append(f"Converted '{col}' to DateTime")
                except Exception as e:
                    report["errors"].append(f"Could not convert date column '{col}': {str(e)}")

    # 3. Numeric Column conversions and missing values
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    # Also check if object columns contain numbers and convert them if mostly numeric
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        # Avoid checking if it's already a date
        if pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
            continue
        try:
            # If at least 70% can be numeric, let's coerce it
            non_null_vals = cleaned_df[col].dropna()
            if len(non_null_vals) > 0:
                converted = pd.to_numeric(non_null_vals, errors='coerce')
                valid_num_pct = converted.notnull().sum() / len(non_null_vals)
                if valid_num_pct > 0.7:
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                    numeric_cols.append(col)
                    report["type_conversions"].append(f"Coerced '{col}' to Numeric")
        except Exception:
            pass

    # 4. Handle missing values
    for col in cleaned_df.columns:
        null_count = cleaned_df[col].isnull().sum()
        if null_count > 0:
            # Handle numeric column
            if col in numeric_cols or pd.api.types.is_numeric_dtype(cleaned_df[col]):
                if fill_numeric == "mean":
                    val = cleaned_df[col].mean()
                    cleaned_df[col] = cleaned_df[col].fillna(val)
                    report["missing_filled"][col] = f"Filled {null_count} missing values with mean ({val:.2f})"
                elif fill_numeric == "median":
                    val = cleaned_df[col].median()
                    cleaned_df[col] = cleaned_df[col].fillna(val)
                    report["missing_filled"][col] = f"Filled {null_count} missing values with median ({val:.2f})"
                elif fill_numeric == "zero":
                    cleaned_df[col] = cleaned_df[col].fillna(0)
                    report["missing_filled"][col] = f"Filled {null_count} missing values with 0"
                elif fill_numeric == "drop":
                    cleaned_df = cleaned_df.dropna(subset=[col])
                    report["missing_filled"][col] = f"Dropped {null_count} rows with missing values"
            # Handle date/time column
            elif pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
                if fill_categorical == "drop" or fill_numeric == "drop":
                    cleaned_df = cleaned_df.dropna(subset=[col])
                    report["missing_filled"][col] = f"Dropped {null_count} rows with missing dates"
                else:
                    # Fill with forward fill or placeholder date
                    cleaned_df[col] = cleaned_df[col].ffill().bfill()
                    report["missing_filled"][col] = f"Filled {null_count} missing dates using forward/backward fill"
            # Handle categorical / object columns
            else:
                if fill_categorical == "mode":
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        val = mode_val[0]
                        cleaned_df[col] = cleaned_df[col].fillna(val)
                        report["missing_filled"][col] = f"Filled {null_count} missing values with mode ('{val}')"
                elif fill_categorical == "placeholder":
                    cleaned_df[col] = cleaned_df[col].fillna("Unknown")
                    report["missing_filled"][col] = f"Filled {null_count} missing values with 'Unknown'"
                elif fill_categorical == "drop":
                    cleaned_df = cleaned_df.dropna(subset=[col])
                    report["missing_filled"][col] = f"Dropped {null_count} rows with missing categories"

    report["final_shape"] = cleaned_df.shape
    return cleaned_df, report

def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes all standard statistics for numerical and categorical columns.
    """
    stats = {}
    
    # 1. General Profile
    stats["row_count"] = len(df)
    stats["col_count"] = len(df.columns)
    stats["columns"] = df.columns.tolist()
    stats["dtypes"] = df.dtypes.astype(str).to_dict()
    stats["memory_usage_bytes"] = int(df.memory_usage(deep=True).sum())
    stats["duplicate_rows"] = int(df.duplicated().sum())
    stats["missing_values"] = df.isnull().sum().to_dict()
    stats["missing_percentage"] = (df.isnull().mean() * 100).round(2).to_dict()
    
    # 2. Numeric Statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats["numeric_cols"] = numeric_cols
    numeric_summary = {}
    
    for col in numeric_cols:
        col_series = df[col].dropna()
        if col_series.empty:
            continue
        
        # Calculate mode safely (might be empty or have multiple)
        mode_val = col_series.mode()
        mode_val = mode_val[0] if not mode_val.empty else np.nan
        
        numeric_summary[col] = {
            "count": len(col_series),
            "mean": float(col_series.mean()),
            "median": float(col_series.median()),
            "mode": float(mode_val) if not pd.isna(mode_val) else None,
            "min": float(col_series.min()),
            "max": float(col_series.max()),
            "variance": float(col_series.var()) if len(col_series) > 1 else 0.0,
            "std_dev": float(col_series.std()) if len(col_series) > 1 else 0.0,
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": float((df[col].isnull().mean() * 100)),
            "unique_values": int(df[col].nunique())
        }
    stats["numeric_summary"] = numeric_summary
    
    # 3. Categorical Statistics
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    stats["categorical_cols"] = categorical_cols
    categorical_summary = {}
    
    for col in categorical_cols:
        col_series = df[col].dropna()
        if col_series.empty:
            continue
            
        value_counts = col_series.value_counts()
        top_cats = value_counts.head(10).to_dict()
        
        categorical_summary[col] = {
            "count": len(col_series),
            "unique_values": int(df[col].nunique()),
            "top_category": str(value_counts.idxmax()) if not value_counts.empty else None,
            "top_category_count": int(value_counts.max()) if not value_counts.empty else 0,
            "bottom_category": str(value_counts.idxmin()) if not value_counts.empty else None,
            "bottom_category_count": int(value_counts.min()) if not value_counts.empty else 0,
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": float((df[col].isnull().mean() * 100)),
            "distribution": top_cats
        }
    stats["categorical_summary"] = categorical_summary
    
    # 4. Correlation Matrix
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        stats["correlation_matrix"] = corr.round(4).to_dict()
    else:
        stats["correlation_matrix"] = {}
        
    return stats
