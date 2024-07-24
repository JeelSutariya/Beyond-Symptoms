import streamlit as st
from streamlit_option_menu import option_menu
from pages import home_page, diabetes_page, heart_disease_page, parkinsons_page
import os

# Set the path to the images directory
IMAGES_DIR = r"D:\Software Development\Beyond-Symptoms\static\images"

st.set_page_config(
    page_title="Beyond Symptoms: Multi-Disease Prediction",
    page_icon="🏥",
    layout="wide"
)

class MultiPage:
    def __init__(self):
        self.pages = {
            "Home": home_page,
            "Diabetes Prediction": diabetes_page,
            "Heart Disease Prediction": heart_disease_page,
            "Parkinson's Prediction": parkinsons_page
        }
    
    def run(self):
        with st.sidebar:
            logo_path = os.path.join(IMAGES_DIR, "logo.jpg")
            if os.path.exists(logo_path):
                st.image(logo_path, width=200)
            else:
                st.warning("Logo image not found.")
            
            page = option_menu(
                menu_title='Navigation',
                options=list(self.pages.keys()),
                icons=['house', 'activity', 'heart', 'person'],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "5!important", "background-color": "#fafafa"},
                    "icon": {"color": "orange", "font-size": "25px"}, 
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "#02ab21"},
                }
            )

        self.pages[page]()

if __name__ == "__main__":
    app = MultiPage()
    app.run()