import copy
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

def information_page(df):
    submenu = st.sidebar.selectbox("Select a section", ["Overview", "Details", "Statistics"])
    
    if submenu == "Overview":
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
        
        st.write("View Data Information")
        
        col3, col4 = st.columns([5 , 1])
        with col4:
            col_column1 = st.selectbox("Choose Column", df.columns, key="column1")
            new_type = st.selectbox("New Type", ["int32", "int64", "float64", "object", "datetime64[ns]"])
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
                except Exception as e:
                    st.error(f"Error converting column '{col_column1}' to {new_type}: {e}")
            if del_null:
                save_to_history(df)
                df = df.dropna(subset=[col_column1])
                st.session_state['df'] = df  # Update session state
                st.success(f"Null values in column '{col_column1}' have been deleted.")
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
                except Exception as e:
                    st.error(f"Error changing null values in column '{col_column1}': {e}")
                    
            info_df = pd.DataFrame({
                "Data Type": df.dtypes,
                "Total Values": df.count(),
                "NaN Count": df.isna().sum(),
                "Unique Values": df.nunique(),
            })
            st.dataframe(info_df)
          
        st.write("View Decription Information")
        
        col5, col6, col7 = st.columns([2 ,3 , 1])
        
        with col7:
            #chọn cột muốn xóa và xóa
            col_column2 = st.selectbox("chose Column", df.columns, key="column2")
            drop_col = st.button("Drop")
            if drop_col:
                save_to_history(df)
                df = df.drop(columns=col_column2)
                st.session_state['df'] = df  # Update session state
            new_column_name = st.text_input("New Name")
            rename_col = st.button("Rename")
            if rename_col:
                if new_column_name:  # Kiểm tra xem đã nhập tên cột mới chưa
                    save_to_history(df)
                    df.rename(columns={col_column2: new_column_name}, inplace=True)
                    st.session_state['df'] = df  # Update session state
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
            
        #print description
        st.write("Description of Data")
        st.write(df.describe())
        
        # Undo, Clean Data, Save buttons
        col8, col9, col10, col11 = st.columns([3 ,1 , 1, 1])
        with col8:
            st.write("")
        with col9:
            if st.button("Undo"):
                undo()
        with col10:
            if st.button("Clean Data"):
                clean_data(df)
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
