import base64
import streamlit as st
from streamlit_option_menu import option_menu
from operation import uploadfile
from pretreatment import information_page
from models import choose_model
from test import execute_model

image_path = "ima.png"
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode()
    return base64_image
base64_image = get_base64_image(image_path)
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{base64_image}") no-repeat top right fixed;
        background-size: 6%;
        background-attachment: local;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 50px; /* Adjust the top margin here */
    }}
    .css-1v3fvcr {{
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

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
        selected = option_menu("Data Analyst", ["File Information", 'Models', 'Model'], 
                               icons=['file-earmark', 'bar-chart'], menu_icon="activity", default_index=0)

        
    if selected == 'File Information' and df is not None:
        information_page(df)
    elif selected == 'Models' and df is not None:
        choose_model(df)
    elif selected == 'Model' and df is not None:
        stored_df = st.session_state['df']
        execute_model(stored_df)

if __name__ == "__main__":
    main()
