import copy
import time
import pandas as pd
import streamlit as st
from operation import save_to_history

def split_columns(df, column, data_type, delimiter, date_part, operator, value, new_column_name, direction, include_delimiter, add_char, add_position, add_target):
    try:
        save_to_history(df)
        if data_type == "object":
            # Split column based on delimiter
            if new_column_name is None:
                new_column_name = f"{column}_split"
            
            if delimiter is not None:
                if direction=="Left to right":
                    if include_delimiter:
                        df[new_column_name] = df[column].apply(lambda x: x.split(delimiter, 1)[0] + delimiter if delimiter in x else x)
                    else:
                        df[new_column_name] = df[column].apply(lambda x: x.split(delimiter, 1)[0] if delimiter in x else x)
                else:
                    if include_delimiter:
                        df[new_column_name] = df[column].apply(lambda x: delimiter + x.split(delimiter)[-1] if delimiter in x else x)
                    else:
                        df[new_column_name] = df[column].apply(lambda x: x.split(delimiter)[-1] if delimiter in x else x)
            else:
                if add_position == 'start':
                    df[new_column_name] = df[column].apply(lambda x: add_char + x)
                elif add_position == 'end':
                    df[new_column_name] = df[column].apply(lambda x: x + add_char)
                elif add_position == 'before':
                    df[new_column_name] = df[column].apply(lambda x: x.replace(add_target, add_char + add_target) if add_target in x else x)
                elif add_position == 'after':
                    df[new_column_name] = df[column].apply(lambda x: x.replace(add_target, add_target + add_char) if add_target in x else x)

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

def add_row(df, new_row_df, dataframe_placeholder):
    try:
        save_to_history(df)
        new_row_series = pd.Series(new_row_df.iloc[0])
        df = pd.concat([df, pd.DataFrame([new_row_series])], ignore_index=True)
        st.session_state['df'] = df
        st.success("Row added successfully!")
        dataframe_placeholder.dataframe(df.tail(10))
    except Exception as e:
        st.error("Failed to add row")

def delete_row(df, row_index, dataframe_placeholder):
    try:
        save_to_history(df)
        df = df.drop(row_index).reset_index(drop=True)
        st.session_state['df'] = df
        st.success("Row deleted successfully!")
        start_row = max(row_index - 5, 0)
        end_row = min(row_index + 5, len(df))
        dataframe_placeholder.dataframe(df.iloc[start_row:end_row])
    except Exception as e:
        st.error("Failed to delete row")

def filter_rows(df, filter_column, filter_operator, filter_value, dataframe_placeholder):
    try:
        if df[filter_column].dtype in ["int64", "float64"]:
            filter_value = float(filter_value)
        else:
            filter_value = str(filter_value)
        
        # Apply filter
        if filter_operator == ">":
            filtered_df = df[df[filter_column] > filter_value]
        elif filter_operator == "<":
            filtered_df = df[df[filter_column] < filter_value]
        else:
            raise ValueError("Unsupported filter operator")
        
        st.write(f"Filtered Dataframe based on {filter_column} {filter_operator} {filter_value}:")
        dataframe_placeholder.dataframe(filtered_df)
    except Exception as e:
        st.error(f"Failed to filter rows: {e}")

def update_row(df, row_index, edited_row, dataframe_placeholder):
    try:
        save_to_history(df)
        df.iloc[row_index] = edited_row.iloc[0]
        st.session_state['df'] = df
        st.success("Row updated successfully!")
        start_row = max(row_index - 5, 0)
        end_row = min(row_index + 5, len(df))
        dataframe_placeholder.dataframe(df.iloc[start_row:end_row])
    except Exception as e:
        st.error("Failed to update row")
        
def merge_dataframes(df1, df2, left_on, right_on):
    merged_df = pd.merge(df1, df2, how='inner', left_on=left_on, right_on=right_on)
    return merged_df