import copy
import time
from matplotlib import pyplot as plt
import pandas as pd
import streamlit as st
from operation import save_to_history
import seaborn as sb

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
        
def plot_chart(data, selected_column, chart_type, y_column=None):
    st.write(f"## {chart_type}")

    if chart_type == 'Bar Chart':
        fig, ax = plt.subplots()
        if y_column and selected_column != y_column:
            data_grouped = data.groupby([selected_column, y_column]).size().unstack(fill_value=0)
            data_grouped = data_grouped.loc[:, data_grouped.sum().nlargest(10).index]  # Lấy 10 giá trị cao nhất
            data_grouped.plot(kind='bar', ax=ax)
        else:
            bar_chart_data = data[selected_column].value_counts().nlargest(10)  # Lấy 10 giá trị cao nhất
            bar_chart_data.plot(kind='bar', ax=ax)
        st.pyplot(fig)

    elif chart_type == 'Line Chart':
        fig, ax = plt.subplots()
        if y_column and selected_column != y_column:
            data.plot(x=selected_column, y=y_column, ax=ax)
        else:
            ax.plot(data[selected_column])
        st.pyplot(fig)

    elif chart_type == 'Scatter Plot':
        if y_column and selected_column != y_column:
            fig, ax = plt.subplots()
            sb.scatterplot(data=data, x=selected_column, y=y_column, ax=ax)
            st.pyplot(fig)
        else:
            st.write("Please select a different column for Y-axis.")

    elif chart_type == 'Pie Chart':
        fig, ax = plt.subplots()
        pie_chart_data = data[selected_column].value_counts().nlargest(10)  # Lấy 10 giá trị cao nhất
        ax.pie(pie_chart_data, labels=pie_chart_data.index, autopct='%1.1f%%')
        st.pyplot(fig)

    elif chart_type == 'Histogram':
        fig, ax = plt.subplots()
        if y_column and selected_column != y_column:
            sb.histplot(data=data, x=selected_column, hue=y_column, multiple="stack", ax=ax)
        else:
            sb.histplot(data=data, x=selected_column, ax=ax)
        st.pyplot(fig)