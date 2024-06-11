import base64
import streamlit as st
from streamlit_option_menu import option_menu
from operation import uploadfile
from pretreatment import information_page
from models import choose_model

def main():
    global columns
    df = st.session_state.get('df')
    if df is None:
        uploadfile()
    else:
        st.sidebar.write(f"File uploaded successfully.")
        if st.sidebar.button("Reload File"):
            st.session_state.pop('df', None)
            st.experimental_rerun()
            
        # def optimize_memory_usage(df):
        #     for col in df.select_dtypes(include=['float64']).columns:
        #         df[col] = df[col].astype('float32')
        #     for col in df.select_dtypes(include=['int64']).columns:
        #         df[col] = df[col].astype('int32')
        #     for col in df.select_dtypes(include=['object']).columns:
        #         num_unique_values = len(df[col].unique())
        #         num_total_values = len(df[col])
        #         if num_unique_values / num_total_values < 0.5:
        #             df[col] = df[col].astype('category')
        #     return df

        # df = optimize_memory_usage(df)
        
    with st.sidebar:
        selected = option_menu("Data Analyst", ["File Information", 'Models'], 
                               icons=['file-earmark', 'bar-chart'], menu_icon="activity", default_index=0)

        
    if selected == 'File Information' and df is not None:
        information_page(df)
    elif selected == 'Models' and df is not None:
        choose_model(df)

if __name__ == "__main__":
    main()
