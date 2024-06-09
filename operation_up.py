import copy
import time
import pandas as pd
import streamlit as st
from operation import save_to_history

def split_columns(df, column, data_type, delimiter, date_part, operator, value, new_column_name):
    print("Data type:", data_type)
    print("Delimiter:", delimiter)
    try:
        save_to_history(df)
        if data_type == "object" and delimiter is not None:
            # Split column based on delimiter
            if new_column_name is None:
                new_column_name = f"{column}_split"
            df[new_column_name] = df[column].str.split(delimiter).str.get(0)
        elif data_type == "datetime64[ns]" and date_part is not None:
            # Split datetime column based on date_part
            if new_column_name is None:
                new_column_name = f"{column}_{date_part}"
            if date_part == "day":
                df[new_column_name] = df[column].dt.day
            elif date_part == "month":
                df[new_column_name] = df[column].dt.month
            elif date_part == "year":
                df[new_column_name] = df[column].dt.year
        elif data_type != "object" and operator is not None:
            # Split numerical column based on operator and value
            if new_column_name is None:
                new_column_name = f"{column}_split"
            if operator == "add":
                df[new_column_name] = df[column] + value
            elif operator == "subtract":
                df[new_column_name] = df[column] - value
            elif operator == "multiply":
                df[new_column_name] = df[column] * value
            elif operator == "divide":
                df[new_column_name] = df[column] / value
        else:
            raise ValueError("Invalid input")
        
        st.success(f"Column '{column}' has been split successfully.")
        time.sleep(1)
        st.experimental_rerun()
        
    except Exception as e:
        st.error(f"Error splitting column '{column}': {e}")
    
def combine_columns(df, columns, operation=None, delimiter=None, new_column_name=None):
    
    try:
        save_to_history(df)
        if all(df[col].dtype in ['int64', 'float64'] for col in columns):
            if operation == "add":
                df[new_column_name] = df[columns].sum(axis=1)
            elif operation == "subtract":
                df[new_column_name] = df[columns].diff(axis=1).iloc[:, -1]
            elif operation == "multiply":
                df[new_column_name] = df[columns].prod(axis=1)
            elif operation == "divide":
                df[new_column_name] = df[columns].iloc[:, 0] / df[columns].iloc[:, 1:].prod(axis=1)
        else:
            df[new_column_name] = df[columns].astype(str).agg(delimiter.join, axis=1)
            
        st.success(f"Columns combined successfully into '{new_column_name}'.")
        time.sleep(1)
        st.experimental_rerun()
        
    except Exception as e:

        st.error(f"Error combining columns: {e}")
        
def prepare_mapping(df, col_column4, data_type):
    mapping = {}
    
    if data_type == "int64" or data_type == "float64":
        unique_values = df[col_column4].unique()
        st.write("Provide the text for each unique number value:")
        mapping_df = pd.DataFrame({
            'Number': unique_values,
            'Text': [''] * len(unique_values)
        })
        with st.expander("Mapping", expanded=True):
            with st.container():
                edited_df = st.data_editor(mapping_df, height=300, width=600, key="mapping_table")
        mapping = dict(zip(edited_df['Number'], edited_df['Text']))
        format_type = "number_to_text"  # Updated this line
        
    else:
        unique_values = df[col_column4].unique()
        st.write("Auto numbering for unique text values:")
        mapping_df = pd.DataFrame({
            'Text': unique_values,
            'Number': range(len(unique_values))
        })
        with st.expander("Mapping", expanded=True):
            with st.container():
                edited_df = st.data_editor(mapping_df, height=300, width=600, key="mapping_table")
        mapping = dict(zip(edited_df['Text'], edited_df['Number']))
        format_type = "text_to_number"  # Updated this line
    
    return mapping, format_type

def format_columns(df, column, format_type, mapping=None, new_column_name=None):
    df_copy = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    
    try:
        save_to_history(df)
        if format_type == "text_to_number":
            df_copy[new_column_name] = df_copy[column].map(mapping)
        elif format_type == "number_to_text":
            df_copy[new_column_name] = df_copy[column].map(mapping)
        st.session_state['df'] = df_copy
        st.success(f"Column '{column}' has been formatted successfully. New column '{new_column_name}' created.")
        time.sleep(1)
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error formatting column '{column}': {e}")
    return df_copy