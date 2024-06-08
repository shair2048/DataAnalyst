import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
# import plotly.graph_objects as go

import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

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
            
            # Loại bỏ cột "Unnamed" nếu có
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            st.session_state['df'] = df
            return df
        except Exception as e:
            st.error(f'Error: {e}')
            return None
    else:
        return None


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
            new_type = st.selectbox("New Type", ["int32", "int64", "float64", "object", "bool", "datetime64[ns]", "category"])
            new_type_button = st.button("Change Type")
            del_null = st.button("Delete Null")
            change_null = st.text_input("Change Null Value")
            change_null_button = st.button("Change Null")

        with col3:

            if new_type_button:
                try:
                    df[col_column1] = df[col_column1].astype(new_type)
                    st.session_state['df'] = df  # Update session state
                    st.success(f"Column '{col_column1}' has been converted to {new_type}.")
                except Exception as e:
                    st.error(f"Error converting column '{col_column1}' to {new_type}: {e}")
            if del_null:
                df = df.dropna(subset=[col_column1])
                st.session_state['df'] = df  # Update session state
                st.success(f"Null values in column '{col_column1}' have been deleted.")
            if change_null_button:
                try:
                    df[col_column1].fillna(change_null, inplace=True)
                    st.session_state['df'] = df  # Update session state
                    st.success(f"Null values in column '{col_column1}' have been changed to '{change_null}'.")
                except Exception as e:
                    st.error(f"Error changing null values in column '{col_column1}' to '{change_null}': {e}")
                    
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
            drop_col = st.button("  Drop  ")
            if drop_col:
                df = df.drop(columns=col_column2)
                st.session_state['df'] = df  # Update session state
            new_column_name = st.text_input("New Name")
            rename_col = st.button(" Rename ")
            if rename_col:
                if new_column_name:  # Kiểm tra xem đã nhập tên cột mới chưa
                    df.rename(columns={col_column2: new_column_name}, inplace=True)  
                    st.session_state['df'] = df  # Update session state
            unique_col = st.button("Unique")
            if unique_col:
                unique_values = df[col_column2].unique()
                unique_df = pd.DataFrame({col_column2: unique_values})
                
        with col5:
            st.write(df.columns)
        with col6:    
            try:
                st.write(unique_df)
            except NameError:
                pass



        
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
    if 'df' not in st.session_state:
        df = uploadfile()  # Assign the result of uploadfile() to df
    else:
        df = st.session_state['df']

    with st.sidebar:
        selected = option_menu("Main Menu", ["CSV Information", 'Diagram'], 
            icons=['house', 'gear'], menu_icon="cast", default_index=1)
        selected
        
    if selected == 'CSV Information' and df is not None:  # Check if df is not None
        information_page(df)  # Pass df as a parameter
    elif selected == 'Diagram':
        diagram_page()
    else:
        st.write('Unknown option')

if __name__ == "__main__":
    main()