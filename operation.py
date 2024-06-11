import copy
import time
import pandas as pd
import streamlit as st

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
        st.success("Undo successful.")
        time.sleep(1)
        st.experimental_rerun()
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
    
    # Gán lại df đã được xử lý vào st.session_state['df']
    st.session_state['df'] = df
    st.success("Data cleaned successfully.")
    time.sleep(1)
    st.experimental_rerun()
    
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
    

