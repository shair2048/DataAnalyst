import streamlit as st

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
