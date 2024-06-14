import time
from matplotlib import pyplot as plt
import pandas as pd
import streamlit as st
from operation import undo, clean_data, save_data, search_dataframe
from operation_info import change_null, change_type, plot_chart, del_null, drop_column, rename_column, unique_column
from operation_up import add_row, delete_row, filter_rows, prepare_mapping, combine_columns, format_columns, split_columns, update_row, merge_dataframes

def information_page(df):
    submenu = st.sidebar.selectbox("Select a section", ["Overview", "Details", "Visualizations"])
    
    if submenu == "Overview":
        dataframe_view, dataframe_info, column_detail, decription_info, recomment = st.tabs(["View Dataframe", "Dataframe Info", "Column Detail", "Decription Info", "Recomment"])
        
        with dataframe_view:
            st.write("View Dataframe")
            col1, col2 = st.columns([5 , 1])
            
            with col2:
                display_search = st.checkbox("Search")
                if display_search:
                    search_term = st.text_input("Search Term")
                    search_column = st.selectbox("Search Column", ["All Columns"] + df.columns.tolist())
                display_type = st.selectbox("Display Type", ["Head", "Tail", "Sample"])
                
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
                    st.write(result_df.head(len(result_df)))
                elif display_type == "Tail":
                    st.write(result_df.tail(len(result_df)))
                elif display_type == "Sample":
                    st.write(result_df.sample(len(result_df)))
            
        with dataframe_info:
            st.write("Dataframe Information")
            col3, col4 = st.columns([5 , 1])
            
            with col4:
                col_column1 = st.selectbox("Choose Column", df.columns, key="column1")
                new_type = st.selectbox("New Type", ["int64", "float64", "object", "datetime64[ns]"])
                new_type_button = st.button("Change Type")
                del_null_button = st.button("Delete Null")
                change_null_text = None

            with col3:       
                info_df = pd.DataFrame({
                    "Data Type": df.dtypes,
                    "Total Values": df.count(),
                    "NaN Count": df.isna().sum(),
                    "Unique Values": df.nunique(),
                })
                st.dataframe(info_df)
                
            not_null_checkbox = st.checkbox("Not Null")
            selected_values = []
            if not_null_checkbox:
                unique_values = df[col_column1].dropna().unique()
                selected_values = st.multiselect("Select Values to Replace", unique_values)
                change_null_text = st.text_input("New Value for Selected")
            else:
                data_type = df[col_column1].dtype
                if data_type == "object" or data_type == "datetime64[ns]":
                    get_null = st.selectbox("Change Value", ["Most Value", "Least Value", "New Value"])
                else:
                    get_null = st.selectbox("Change Value", ["Max Value", "Mean Value", "Min Value", "New Value"])
                if get_null == "New Value":
                    change_null_text = st.text_input("New Value")
                
            change_null_button = st.button("Change Value")
                
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
            
        with recomment:

            # Selectbox for object (categorical) variable
            object_cols = df.select_dtypes(include='object').columns.tolist()
            object_var = st.selectbox('Select a categorical variable', object_cols)

            # Selectbox for numerical variable
            num_cols = df.select_dtypes(include='number').columns.tolist()
            num_var = st.selectbox('Select a numerical variable', num_cols)

            # Grouping data
            grouped_data = df.groupby(object_var)[num_var].mean().reset_index()
            grouped_data['count'] = df.groupby(object_var)[num_var].count().values

            # Sorting the grouped data and selecting the top 10 by count
            top_grouped_data = grouped_data.sort_values(by='count', ascending=False).head(10)

            # Display the top grouped data
            st.write("Top 10 Grouped Data:")
            st.dataframe(top_grouped_data)

            # Plotting
            fig, ax = plt.subplots()
            top_grouped_data.plot(kind='bar', x=object_var, y='count', ax=ax)
            plt.title(f'Top 10 Counts of {num_var} by {object_var}')
            plt.xlabel(object_var)
            plt.ylabel('Count')

            # Display the plot
            st.pyplot(fig)
        
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
            if not_null_checkbox:
                change_null(df, col_column1, "Replace Values", change_null_text, selected_values)
            else:
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
        dataframe_view, split_column, combine_column, format_column, row_operations, merge_file = st.tabs(["View Dataframe", "Split Column", "Combine Column", "Format Column", "Row Operations", "Merge File"])
        
        with dataframe_view:
            st.write("View Dataframe")
            
            col1, col2 = st.columns([5 , 1])
            with col2:
                display_search = st.checkbox("Search")
                if display_search:
                    search_term = st.text_input("Search Term")
                    search_column = st.selectbox("Search Column", ["All Columns"] + df.columns.tolist())
                display_type = st.selectbox("Display Type", ["Head", "Tail", "Sample"])
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
                    st.write(result_df.head(len(result_df)))
                elif display_type == "Tail":
                    st.write(result_df.tail(len(result_df)))
                elif display_type == "Sample":
                    st.write(result_df.sample(len(result_df)))
                    
        with split_column:
            st.write("Split Column")
            
            col_column3 = st.selectbox("Choose Column", df.columns, key="column3")
            new_column_name3 = st.text_input("New Column Name")
            data_type3 = df[col_column3].dtype
            
            delimiter = None
            date_part = None
            operator = None
            value = None
            include_delimiter = False
            direction = "Left to right"
            add_char = None
            add_position = None
            add_target = None

            
            if data_type3 == "object":
                action = st.radio("Choose action:", ("split", "add"))
                if action == "split":
                    delimiter = st.text_input("Delimiter")
                    direction = st.radio("Split direction:", ("Left to right", "Right to left"))
                    include_delimiter = st.checkbox("Include delimiter in new column")
                else:
                    add_char = st.text_input("Character to add")
                    add_position = st.radio("Position to add character:", ("start", "end", "before", "after"))
                    if add_position in ["before", "after"]:
                        add_target = st.text_input("Target substring")    
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
            
        with row_operations:
            st.write("Row Operations")
            
            # Initialize button variables to False
            add_row_button = delete_row_button = filter_button = update_row_button = False
            st.write("Dataframe Preview:")
            dataframe_placeholder = st.empty()
            dataframe_placeholder.dataframe(result_df)
            
            operation = st.selectbox("Choose Operation", ["Add Row", "Delete Row", "Filter Rows", "Update Row"])
                
            if operation == "Add Row":
                st.write("Enter new row data:")
                new_row_data = {column: "" for column in df.columns}
                new_row_df = pd.DataFrame(new_row_data, index=[0])
                new_row_df = st.data_editor(new_row_df, key="new_row_editor")
                add_row_button = st.button("Add Row")
                    
            elif operation == "Delete Row":
                row_index = st.number_input("Row Index", min_value=0, max_value=len(df)-1, key="delete_row_index")
                if row_index is not None:
                    st.dataframe(df.iloc[[row_index]])
                delete_row_button = st.button("Delete Row")
            
            elif operation == "Filter Rows":
                filter_column = st.selectbox("Filter Column", df.columns, key="filter_column")
                filter_operator = st.selectbox("Operator", [">", "<"], key="filter_operator")
                filter_value = st.text_input("Filter Value", key="filter_value")
                filter_button = st.button("Filter Rows")
                
            elif operation == "Update Row":
                row_index = st.number_input("Row Index", min_value=0, max_value=len(df)-1, key="update_row_index")
                if row_index is not None:
                    row_data = df.iloc[[row_index]].copy()
                    st.write("Editing Row:")
                    edited_row = st.data_editor(row_data, key="edit_row")
                update_row_button = st.button("Update Row")

            if add_row_button:
                add_row(df, new_row_df, dataframe_placeholder)

            if delete_row_button:
                delete_row(df, row_index, dataframe_placeholder)

            if filter_button:
                filter_rows(df, filter_column, filter_operator, filter_value, dataframe_placeholder)

            if update_row_button:
                update_row(df, row_index, edited_row, dataframe_placeholder)

        with merge_file:
            st.title("Merge Files")

            st.write("Upload the first file:")
            uploaded_file1 = st.file_uploader("Choose a CSV file", type=["csv"], key="file_uploader1")

            st.write("Upload the second file:")
            uploaded_file2 = st.file_uploader("Choose a CSV file", type=["csv"], key="file_uploader2")

            if uploaded_file1 and uploaded_file2:
                df1 = pd.read_csv(uploaded_file1)  # Read uploaded file into a DataFrame
                df2 = pd.read_csv(uploaded_file2)  # Read uploaded file into a DataFrame
                
                st.write(df1.head())
                st.write(df2.head())

                left_on = st.selectbox("Choose column from Dataframe 1", df1.columns)
                right_on = st.selectbox("Choose column from Dataframe 2", df2.columns)

                merge_button = st.button("Merge Dataframes")

                if merge_button:
                    merged_df = merge_dataframes(df1, df2, left_on, right_on)
                    st.subheader("Merged Dataframe")
                    st.write(merged_df)


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
            split_columns(df, col_column3, data_type3, delimiter, date_part, operator, value, new_column_name3, direction, include_delimiter, add_char, add_position, add_target)
        if combine_button:
            combine_columns(df, col_column4, operation, delimiter1, new_column_name4)
        if format_button:
            format_columns(df, col_column5, format_type, mapping, new_column_name5)
        if undo_button:
            undo()
        if clean_button:
            clean_data(df)
    
    elif submenu == "Visualizations":
        st.write("Visualizations")
        
        # Filter columns where not all unique values are equal to the total number of rows
        valid_columns = [col for col in df.columns if df[col].nunique() < len(df)]

        selected_column = st.sidebar.selectbox('Select a column', valid_columns, index=0)
        
        # Sidebar: Select chart type
        chart_type = st.sidebar.selectbox('Select chart type', [
            'Bar Chart', 
            'Line Chart', 
            'Scatter Plot', 
            'Pie Chart', 
            'Histogram'
        ])

        y_column = None
        if chart_type in ['Bar Chart', 'Line Chart', 'Scatter Plot', 'Histogram']:
            y_column = st.sidebar.selectbox('Select Y-axis column (optional)', ['None'] + valid_columns)
            if y_column == 'None':
                y_column = None

        # Plot chart based on user selection
        plot_chart(df, selected_column, chart_type, y_column)
