import copy
import time
import pandas as pd
import streamlit as st
from operation import save_to_history

def change_type(df, column_name, new_type):
    try:
        save_to_history(df)
        df[column_name] = df[column_name].astype(new_type)
        st.session_state['df'] = df  # Update session state
        st.success(f"Column '{column_name}' has been converted to {new_type}.")
        time.sleep(1)
        st.experimental_rerun()  # Reload the tab
    except Exception as e:
        st.error(f"Error converting column '{column_name}' to {new_type}: {e}")
        
def del_null(df, column_name):
    try:
        save_to_history(df)
        df = df.dropna(subset=[column_name])
        st.session_state['df'] = df  # Update session state
        st.success(f"Null values in column '{column_name}' have been deleted.")
        time.sleep(1)
        st.experimental_rerun()  # Reload the tab
    except Exception as e:
        st.error(f"Error deleting null values in column '{column_name}': {e}")
        
        
def change_null(df, column_name, strategy, new_value=None, selected_values=None):
    try:
        save_to_history(df)
        if strategy == "Most Value":
            most_value = df[column_name].mode()[0]
            df[column_name].fillna(most_value, inplace=True)
        elif strategy == "Least Value":
            least_value = df[column_name].mode().iloc[-1]
            df[column_name].fillna(least_value, inplace=True)
        elif strategy == "Max Value":
            max_value = df[column_name].max()
            df[column_name].fillna(max_value, inplace=True)
        elif strategy == "Mean Value":
            mean_value = df[column_name].mean()
            df[column_name].fillna(mean_value, inplace=True)
        elif strategy == "Min Value":
            min_value = df[column_name].min()
            df[column_name].fillna(min_value, inplace=True)
        elif strategy == "New Value":
            if df[column_name].dtype == "datetime64[ns]":
                new_value = pd.to_datetime(new_value)
            df[column_name].fillna(new_value, inplace=True)
        elif strategy == "Replace Values":
            df[column_name].replace(selected_values, new_value, inplace=True)
            
        st.session_state['df'] = df  # Update session state
        st.success(f"Values in column '{column_name}' have been changed to '{new_value if strategy == 'Replace Values' else strategy}'.")
        time.sleep(1)
        st.experimental_rerun()  # Reload the tab
    except Exception as e:
        st.error(f"Error changing values in column '{column_name}': {e}")
        
def unique_column(df, column_name):
    unique_values = df[column_name].unique()
    unique_df = pd.DataFrame({
        column_name: unique_values,
        "Count": [df[df[column_name] == value].shape[0] for value in unique_values]})
    st.write(unique_df)

def drop_column(df, column_name):
    try:
        save_to_history(df)
        df.drop(columns=column_name, inplace=True)
        st.session_state['df'] = df  # Update session state
        st.success(f"Column '{column_name}' dropped successfully.")
        time.sleep(1)
        st.experimental_rerun()  # Reload the tab
    except Exception as e:
        st.error(f"Error dropping column '{column_name}': {e}")
        
def rename_column(df, old_name, new_name):
    try:
        save_to_history(df)
        df.rename(columns={old_name: new_name}, inplace=True)
        st.session_state['df'] = df  # Update session state
        st.success(f"Column '{old_name}' has been renamed to '{new_name}'.")
        time.sleep(1)
        st.experimental_rerun()  # Reload the tab
    except Exception as e:
        st.error(f"Error renaming column '{old_name}': {e}")