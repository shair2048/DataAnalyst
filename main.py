import copy
import time
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

def save_to_history(df):
    if 'df_history' not in st.session_state:
        st.session_state['df_history'] = []
    st.session_state['df_history'].append(copy.deepcopy(df))

def uploadfile():
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            # Đọc dữ liệu từ tập tin CSV hoặc Excel
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error('Only CSV and Excel files are supported.')
                return None
            save_to_history(df)
            
            # Loại bỏ cột "Unnamed" nếu có
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            st.session_state['df'] = df
        except Exception as e:
            st.error(f'Error: {e}')
    else:
        if 'df' in st.session_state:
            return st.session_state['df']
        else:
            return None
    st.experimental_rerun()

def undo():
    if 'df_history' in st.session_state and st.session_state['df_history']:
        st.session_state['df'] = st.session_state['df_history'].pop()
    else:
        st.warning("No previous version available to undo.")

def clean_data(df):
    save_to_history(df)
    for column in df.columns:
        null_count = df[column].isnull().sum()
        total_count = len(df[column])
        
        if null_count > total_count / 3:
            # Nếu số lượng giá trị null lớn hơn 1/3 tổng số lượng giá trị, xóa cột đó
            del df[column]
        elif null_count > 0:
            if df[column].dtype == 'object' or df[column].dtype == 'datetime64[ns]':
                most_common_value = df[column].mode()[0]
                df[column].fillna(most_common_value, inplace=True)
            else:
                mean_value = df[column].mean()
                df[column].fillna(mean_value, inplace=True)
    
    st.session_state['df'] = df
    st.success("Data cleaned successfully.")
    
def save_data(df, file_name):
    try:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download",
            data=csv_data,
            file_name=file_name,
            mime='text/csv',
        )
    except Exception as e:
        st.error(f"Error saving data: {e}")

def search_dataframe(df, search_term, column=None):
    if column:
        # Tìm kiếm theo một cột cụ thể
        return df[df[column].astype(str).str.contains(search_term, case=False)]
    else:
        # Tìm kiếm toàn bộ DataFrame
        return df[df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]
    
def split_column(df, column, data_type, delimiter=None, date_part=None, operator=None, value=None, new_column_name=None):
    if data_type == "object" and delimiter is not None:
        # Tách cột dựa trên delimiter
        if new_column_name is None:
            new_column_name = f"{column}_split"
        df[new_column_name] = df[column].str.split(delimiter).str.get(0)  # Thực hiện tách cột
    elif data_type == "datetime64[ns]" and date_part is not None:
        # Tách cột datetime theo date_part
        if new_column_name is None:
            new_column_name = f"{column}_{date_part}"
        if date_part == "day":
            df[new_column_name] = df[column].dt.day
        elif date_part == "month":
            df[new_column_name] = df[column].dt.month
        elif date_part == "year":
            df[new_column_name] = df[column].dt.year
    elif data_type != "object" and operator is not None:
        # Tách cột số dựa trên operator và value
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
    
def combine_columns(df, columns, operation=None, delimiter=None, new_column_name=None):
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
    return df

def format_columns(df, column, format_type, mapping=None, new_column_name=None):
    df_copy = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    
    if format_type == "text_to_number":
        df_copy[new_column_name] = df_copy[column].map(mapping)
    elif format_type == "number_to_text":
        df_copy[new_column_name] = df_copy[column].map(mapping)
    return df_copy

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
                del_null = st.button("Delete Null")
                data_type = df[col_column1].dtype
                if data_type == "object" or data_type == "datetime64[ns]":
                    get_null = st.selectbox("Change Value", ["Most Value", "Least Value", "New Value"])
                else:
                    get_null = st.selectbox("Change Value", ["Max Value", "Mean Value", "Min Value", "New Value"])
                if get_null == "New Value":
                    change_null = st.text_input("New Value")
                change_null_button = st.button("Change Null")

            with col3:
                if new_type_button:
                    try:
                        save_to_history(df)
                        df[col_column1] = df[col_column1].astype(new_type)
                        st.session_state['df'] = df  # Update session state
                        st.success(f"Column '{col_column1}' has been converted to {new_type}.")
                        time.sleep(1)
                        st.experimental_rerun()  # Reload the tab
                    except Exception as e:
                        st.error(f"Error converting column '{col_column1}' to {new_type}: {e}")
                if del_null:
                    save_to_history(df)
                    df = df.dropna(subset=[col_column1])
                    st.session_state['df'] = df  # Update session state
                    st.success(f"Null values in column '{col_column1}' have been deleted.")
                    time.sleep(1)
                    st.experimental_rerun()  # Reload the tab
                if change_null_button:
                    try:
                        save_to_history(df)
                        if get_null == "Most Value":
                            most_value = df[col_column1].mode()[0]
                            df[col_column1].fillna(most_value, inplace=True)
                        elif get_null == "Least Value":
                            least_value = df[col_column1].mode().iloc[-1]  
                            df[col_column1].fillna(least_value, inplace=True)
                        elif get_null == "Max Value":
                            max_value = df[col_column1].max()
                            df[col_column1].fillna(max_value, inplace=True)
                        elif get_null == "Mean Value":
                            mean_value = df[col_column1].mean()
                            df[col_column1].fillna(mean_value, inplace=True)
                        elif get_null == "Min Value":
                            min_value = df[col_column1].min()
                            df[col_column1].fillna(min_value, inplace=True)
                        elif get_null == "New Value":
                            if data_type == "datetime64[ns]":
                                change_null = pd.to_datetime(change_null)
                            df[col_column1].fillna(change_null, inplace=True)
                            
                        st.session_state['df'] = df  # Update session state
                        st.success(f"Null values in column '{col_column1}' have been changed to '{get_null if get_null != 'New Value' else change_null}'.")
                        time.sleep(1)
                        st.experimental_rerun()  # Reload the tab
                    except Exception as e:
                        st.error(f"Error changing null values in column '{col_column1}': {e}")
                        
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
                drop_col = st.button("Drop")
                if drop_col:
                    save_to_history(df)
                    df = df.drop(columns=col_column2)
                    st.session_state['df'] = df  # Update session state
                    time.sleep(1)
                    st.experimental_rerun()  # Reload the tab
                new_column_name = st.text_input("New Name")
                rename_col = st.button("Rename")
                if rename_col:
                    if new_column_name:  # Kiểm tra xem đã nhập tên cột mới chưa
                        save_to_history(df)
                        df.rename(columns={col_column2: new_column_name}, inplace=True)
                        st.session_state['df'] = df  # Update session state
                        time.sleep(1)
                        st.experimental_rerun()  # Reload the tab
                unique_col = st.button("Unique")
                if unique_col:
                    unique_values = df[col_column2].unique()
                    unique_df = pd.DataFrame({
                        col_column2: unique_values,
                        "Count": [df[df[col_column2] == value].shape[0] for value in unique_values]})
                    
            with col5:
                st.write(df.columns)
            with col6:    
                try:
                    st.write(unique_df)
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
            if st.button("Undo"):
                undo()
                time.sleep(1)
                st.experimental_rerun()  # Reload the tab
        with col10:
            if st.button("Clean Data"):
                clean_data(df)
                time.sleep(1)
                st.experimental_rerun()  # Reload the tab
        with col11:
            if st.button("Save Data"):
                save_data(st.session_state['df'], "data_clean.csv")
                
    elif submenu == "Details":
        dataframe_view, separate_column, combine_column, format_column = st.tabs(["View Dataframe", "Separate Column", "Combine Column", "Format Column"])
        
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
        with separate_column:
            st.write("Separate Column")
        
            col_column3 = st.selectbox("Choose Column", df.columns, key="column3")
            new_column_name = st.text_input("New Column Name")
            data_type = df[col_column3].dtype
            if data_type == "object":
                delimiter = st.text_input("Delimiter")
            elif data_type == "datetime64[ns]":
                date_part = st.selectbox("Date Part", ["day", "month", "year"])
            else:
                operator = st.selectbox("Operator", ["add", "subtract", "multiply", "divide"])
                value = st.number_input("Value", value=0)
            split_button = st.button("Split Column")
            if split_button:
                save_to_history(df)
                try:
                    if data_type == "object":
                        split_column(df, col_column3, data_type, delimiter=delimiter, new_column_name=new_column_name)
                    elif data_type == "datetime64[ns]":
                        split_column(df, col_column3, data_type, date_part=date_part, new_column_name=new_column_name)
                    else:
                        split_column(df, col_column3, data_type, operator=operator, value=value, new_column_name=new_column_name)
                    st.success(f"Column '{col_column3}' has been split successfully.")
                    time.sleep(1)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error splitting column '{col_column3}': {e}")
                    
        with combine_column:
            st.write("Combine Columns")
            
            columns_to_combine = st.multiselect("Choose Columns to Combine", df.columns)
            new_column_name = st.text_input("New Column Name", key="combine_name")
            
            if columns_to_combine:
                if all(df[col].dtype in ['int64', 'float64'] for col in columns_to_combine):
                    operation = st.selectbox("Operation", ["add", "subtract", "multiply", "divide"])
                    delimiter = None
                else:
                    operation = None
                    delimiter = st.text_input("Delimiter")
                
                combine_button = st.button("Combine Columns")
                
                if combine_button:
                    save_to_history(df)
                    try:
                        df = combine_columns(df, columns_to_combine, operation=operation, delimiter=delimiter, new_column_name=new_column_name)
                        st.session_state['df'] = df
                        st.success(f"Columns combined successfully into '{new_column_name}'.")
                        time.sleep(1)
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error combining columns: {e}")
        with format_column:
            st.write("Format Column")
            
            col_column4 = st.selectbox("Choose Column", df.columns, key="column4")
            new_column_name = st.text_input("New Column Name", key="format_name")
            data_type = df[col_column4].dtype
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
            
            format_button = st.button("Format Column")
            if format_button:
                save_to_history(df)
                try:
                    df = format_columns(df, col_column4, format_type, mapping, new_column_name=new_column_name)
                    st.session_state['df'] = df
                    st.success(f"Column '{col_column4}' has been formatted successfully. New column '{new_column_name}' created.")
                    time.sleep(1)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error formatting column '{col_column4}': {e}")
        
        # Undo, Clean Data, Save buttons
        col8, col9, col10, col11 = st.columns([3 ,1 , 1, 1])
        with col8:
            st.write("")
        with col9:
            if st.button("Undo"):
                undo()
                time.sleep(1)
                st.experimental_rerun()  # Reload the tab
        with col10:
            if st.button("Clean Data"):
                clean_data(df)
                time.sleep(1)
                st.experimental_rerun()  # Reload the tab
        with col11:
            if st.button("Save Data"):
                save_data(st.session_state['df'], "data_clean.csv")

                

                
                    

def diagram_page():
    st.title("Settings")
    st.write("Welcome to the settings page!")
    
    submenu = st.selectbox("Select a setting", ["Profile", "Preferences", "Security"])
    if submenu == "Profile":
        st.write("This is the Profile section.")
    elif submenu == "Preferences":
        st.write("This is the Preferences section.")
    elif submenu == "Security":
        st.write("This is the Security section.")

def main():
    df = st.session_state.get('df')
    if df is None:
        uploadfile()
    else:
        st.sidebar.write(f"File uploaded successfully.")
        if st.sidebar.button("Reload File"):
            st.session_state.pop('df', None)
            st.experimental_rerun()

    with st.sidebar:
        selected = option_menu("Main Menu", ["CSV Information", 'Diagram'], 
            icons=['house', 'gear'], menu_icon="cast", default_index=0)
        
    if selected == 'CSV Information' and df is not None:
        information_page(df)
    elif selected == 'Diagram':
        diagram_page()
    else:
        st.write('Unknown option')

if __name__ == "__main__":
    main()
