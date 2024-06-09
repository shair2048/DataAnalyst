# main.py
import streamlit as st
from streamlit_option_menu import option_menu
from operation import uploadfile
from pretreatment import information_page
from diagram import diagram_page

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
    elif selected == 'Diagram' and df is not None:
        diagram_page()
    else:
        st.write('Unknown option')

if __name__ == "__main__":
    main()
