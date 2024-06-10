# main.py
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

    with st.sidebar:
        selected = option_menu("Data Analyst", ["File Information", 'Models'], 
                               icons=['file-earmark', 'bar-chart'], menu_icon="activity", default_index=0)

        
    if selected == 'File Information' and df is not None:
        information_page(df)
    elif selected == 'Models' and df is not None:
        choose_model(df)
    else:
        st.write('Unknown option')

if __name__ == "__main__":
    main()
