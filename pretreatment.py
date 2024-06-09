import time
import pandas as pd
import streamlit as st
from operation import undo, clean_data, save_data, search_dataframe
from operation_info import change_null, change_type, del_null, drop_column, rename_column, unique_column
from operation_up import prepare_mapping, combine_columns, format_columns, split_columns

def information_page(df):
    submenu = st.sidebar.selectbox("Select a section", ["Overview", "Details", "Statistics"])
    
    if submenu == "Overview":
        dataframe_view, dataframe_info, column_detail, decription_info = st.tabs(["View Dataframe", "Dataframe Info", "Column Detail", "Decription Info"])
        
        with dataframe_view:
            st.write("View Dataframe")
            col1, col2 = st.columns([5 , 1])
            
            with col2:
                display_search = st.checkbox("Search")
                if display_search:
                    search_term = st.text_input("Search Term")
                    search_column = st.selectbox("Search Column", ["All Columns"] + df.columns.tolist())
                display_type = st.selectbox("Display Type", ["Head", "Tail", "Sample"])
                display_count = st.number_input("Number of Rows", min_value=1, value=10)
                
            with col1:
                if display_search and search_term:
                    if search_column == "All Columns":
                        result_df = search_dataframe(df, search_term)
                    else:
                        result_df = search_dataframe(df, search_term, column=search_column)
                    st.write(f"Search Results for '{search_term}' in '{search_column}':")
                else:
                    result_df = df
                if display_type == "Head":
                    st.write(result_df.head(display_count))
                elif display_type == "Tail":
                    st.write(result_df.tail(display_count))
                elif display_type == "Sample":
                    st.write(result_df.sample(display_count))
            
        with dataframe_info:
            st.write("Dataframe Information")
            col3, col4 = st.columns([5 , 1])
            
            with col4:
                col_column1 = st.selectbox("Choose Column", df.columns, key="column1")
                new_type = st.selectbox("New Type", ["int64", "float64", "object", "datetime64[ns]"])
                new_type_button = st.button("Change Type")
                del_null_button = st.button("Delete Null")
                data_type = df[col_column1].dtype
                if data_type == "object" or data_type == "datetime64[ns]":
                    get_null = st.selectbox("Change Value", ["Most Value", "Least Value", "New Value"])
                else:
                    get_null = st.selectbox("Change Value", ["Max Value", "Mean Value", "Min Value", "New Value"])
                if get_null == "New Value":
                    change_null_text = st.text_input("New Value")
                change_null_button = st.button("Change Null")

            with col3:       
                info_df = pd.DataFrame({
                    "Data Type": df.dtypes,
                    "Total Values": df.count(),
                    "NaN Count": df.isna().sum(),
                    "Unique Values": df.nunique(),
                })
                st.dataframe(info_df)
                
        with column_detail:
            st.write("Column Detail")
            col5, col6, col7 = st.columns([2 ,3 , 1])
            
            with col7:
                col_column2 = st.selectbox("chose Column", df.columns, key="column2")
                drop_col_button = st.button("Drop")
                new_column_name = st.text_input("New Name")
                rename_col_button = st.button("Rename")
                unique_col_button = st.button("Unique")
            
            with col5:
                st.write(df.columns)
                
            with col6:    
                try:
                    if unique_col_button:
                        unique_column(df, col_column2)
                except NameError:
                    pass
        
        with decription_info:        
            st.write("Description Information")
            st.write(df.describe())
        
        # Undo, Clean Data, Save buttons
        col8, col9, col10, col11 = st.columns([3 ,1 , 1, 1])
        with col8:
            st.write("")
        with col9:
            undo_button = st.button("Undo")
        with col10:
            clean_button = st.button("Clean Data")
        with col11:
            if st.button("Save Data"):
                save_data(df, "data_clean.csv")
                
        if new_type_button:
            change_type(df, col_column1, new_type)
        if del_null_button:
            del_null(df, col_column1)
        if change_null_button:
            change_null(df, col_column1, get_null, change_null_text)
        if drop_col_button:
            drop_column(df, col_column2)
        if rename_col_button and new_column_name:
           rename_column(df, col_column2, new_column_name)
        if undo_button:
            undo()
        if clean_button:
            clean_data(df)
                
    elif submenu == "Details":
        dataframe_view, split_column, combine_column, format_column = st.tabs(["View Dataframe", "Split Column", "Combine Column", "Format Column"])
        
        with dataframe_view:
            st.write("View Dataframe")
            
            col1, col2 = st.columns([5 , 1])
            with col2:
                display_search = st.checkbox("Search")
                if display_search:
                    search_term = st.text_input("Search Term")
                    search_column = st.selectbox("Search Column", ["All Columns"] + df.columns.tolist())
                display_type = st.selectbox("Display Type", ["Head", "Tail", "Sample"])
                display_count = st.number_input("Number of Rows", min_value=1, value=10)
            with col1:
                if display_search and search_term:
                    if search_column == "All Columns":
                        result_df = search_dataframe(df, search_term)
                    else:
                        result_df = search_dataframe(df, search_term, column=search_column)
                    st.write(f"Search Results for '{search_term}' in '{search_column}':")
                else:
                    result_df = df
                if display_type == "Head":
                    st.write(result_df.head(display_count))
                elif display_type == "Tail":
                    st.write(result_df.tail(display_count))
                elif display_type == "Sample":
                    st.write(result_df.sample(display_count))
                    
        with split_column:
            st.write("Split Column")
            
            col_column3 = st.selectbox("Choose Column", df.columns, key="column3")
            new_column_name3 = st.text_input("New Column Name")
            data_type3 = df[col_column3].dtype
            
            delimiter = None
            date_part = None
            operator = None
            value = None
            
            if data_type3 == "object":
                delimiter = st.text_input("Delimiter")
            elif data_type3 == "datetime64[ns]":
                date_part = st.selectbox("Date Part", ["day", "month", "year"])
            else:
                operator = st.selectbox("Operator", ["add", "subtract", "multiply", "divide"])
                value = st.number_input("Value", value=0)
                
            separate_button = st.button("Separate Column")
                
                    
        with combine_column:
            st.write("Combine Columns")
            
            col_column4 = st.multiselect("Choose Columns to Combine", df.columns, key="column4")
            new_column_name4 = st.text_input("New Column Name", key="combine_name")           
            
            if col_column4:
                if all(df[col].dtype in ['int64', 'float64'] for col in col_column4):
                    operation = st.selectbox("Operation", ["add", "subtract", "multiply", "divide"])
                    delimiter1 = None
                else:
                    operation = None
                    delimiter1 = st.text_input("Delimiter")
                    
            combine_button = st.button("Combine Columns")
                
                
                    
        with format_column:
            st.write("Format Column")
            
            col_column5 = st.selectbox("Choose Column", df.columns, key="column5")
            new_column_name5 = st.text_input("New Column Name", key="format_name")
            data_type5 = df[col_column5].dtype
            
            mapping, format_type = prepare_mapping(df, col_column5, data_type5)
            
            format_button = st.button("Format Column")
            
            

        
        # Undo, Clean Data, Save buttons

        col8, col9, col10, col11 = st.columns([3 ,1 , 1, 1])
        with col8:
            st.write("")
        with col9:
            undo_button = st.button("Undo")
        with col10:
            clean_button = st.button("Clean Data")
        with col11:
            if st.button("Save Data"):
                save_data(df, "data_clean.csv")
                
        if separate_button:
            split_columns(df, col_column3, data_type3, delimiter, date_part, operator, value, new_column_name3)
        if combine_button:
            combine_columns(df, col_column4, operation, delimiter1, new_column_name4)
        if format_button:
            format_columns(df, col_column5, format_type, mapping, new_column_name5)
        if undo_button:
            undo()
        if clean_button:
            clean_data(df)
