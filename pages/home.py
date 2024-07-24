import streamlit as st
import os

# Set the path to the images directory
IMAGES_DIR = r"D:\Software Development\Beyond-Symptoms\static\images"

def home_page():
    st.title("Beyond Symptoms: Multi-Disease Prediction System")
    
    image_path = os.path.join(IMAGES_DIR, "logo.png")
    if os.path.exists(image_path):
        st.image(image_path, width=200)
    else:
        st.warning("Heart disease image not found.")

    st.write("""
    Welcome to Beyond Symptoms, an advanced multi-disease prediction system powered by machine learning.
    Our platform currently offers predictions for three critical health conditions:
    
    1. **Diabetes**
    2. **Heart Disease**
    3. **Parkinson's Disease**
    
    Navigate through the sidebar to access each prediction module. Input your health metrics, 
    and our AI models will provide you with instant predictions to support early detection and intervention.
    
    Remember, while our system uses state-of-the-art machine learning models, it's designed to complement, 
    not replace, professional medical advice. Always consult with healthcare professionals for accurate 
    diagnosis and treatment plans.
    
    Made by Jeel Sutariya! You can find the source code at: 
    [https://github.com/JeelSutariya/Beyond-Symptoms](https://github.com/JeelSutariya/Beyond-Symptoms)
    
    This project is based on research published in IEEE. Read the paper here:
    [Predictive Disease Modeling for Proactive Healthcare](https://ieeexplore.ieee.org/document/10581019)
    """)
    
    st.subheader("How it works")
    st.image("https://mermaid.ink/img/pako:eNqVll1P2zAUhv-KlYtNmmBSKZ-92FT6EZu1tKJs05QiZBK3teo6UeywMcp_n-OThHi0BbiwsM95j-33ie0-emEcMa_lzVOaLNB1dyqR-WsHE01TfYP297-g82AQ0wi1k0TwkGoeyxvIOrfhTkAk15wK_pehiU4ZXQmui5SOTelChQm9ZxEamgmFKuJdG193Ob1jmikIrlG3AYqoGL9d5eOfk6VwdZiZVaIuV4wqVokPQLzIg7cRBLdVGNN0yaWK5cfnyZugT8rQ1ulhVyEVLFVGdwi6T7fKDuX5aMYFqzbbQB_M6vKmmTeHtkovMOtPBH1AOF4xNKZzVuT3bLz_-F2xFE2YYKFWaJyyiIc5BXT9kLAnSIW277q5Rn4wSphElb01cW2e_gYz1wiD1rV4Z4Gal2tEQF73d7MYWt-W8BuVGUQmmUZ9zkSk0CxOq00UMr8BioPA2tOT2kB43mmXalpmHkBmE4zsmK94WS3ma2Gg34Q9XMbGtYY79staeRgYSZLGIVOqWF19EoDpHwX2e9iQcAQJx8GQLh0jM8XlHLlHoBQdg-iksqWmu2IqE-VB808g8zSYLOLfUASNWWqcW1EZmsPBdMrDyr1TSD-rCreje27SqIxy2KGgfMVSBxG2ErwDkfOtFFoMnLDLyf2qajZhgIV3wsI1WLjhjllYeDcsDLDwVlgYYOFtsDbcO6USiOHXiWEght9IDAMx_B5ixErIDmK141koCfAiLq_6Ma75RIAW2UmL1GiRhjtmaZHdtAjQIltpEaBFttF6cceXOmBFXmdFgBV5IysCrMh7WPln5knAeUPOrPoCPP1Jpbn12zLWC9N7XmDp7kXNyL4zlPv9LejJyJlHZXfw1k-9l9v4wVWWP-T2lZ96oBjAs3bNlOP6MOhQEWaCaoY6sZwZr41tQ2qM-FOkXAZjEett0VGtwNWogzpZel_eGmNQ_j88sDsbFguwnct6ZFRUtp0xdJiMHJ-BzcA5VQOH28Db8wyeFeWR-WX0mMemnrF_xaZey_wbsRk130buz5NJpZmOJw8y9Fo6zdiel8bZfOG1ZlQo08uSyGzQ3O7G9RWkPP0Ds-7zPw",
             caption="Detailed application flowchart", use_column_width=True)

    st.subheader("Model Accuracies")
    col1, col2, col3 = st.columns(3)
    col1.metric("Diabetes Model", "80%", "")
    col2.metric("Heart Disease Model", "88%", "")
    col3.metric("Parkinson's Model", "95%", "")

    st.markdown("---")
    st.subheader("Research and Development")
    st.write("""
    This project is based on extensive research in the field of machine learning for healthcare. 
    The methodology and findings have been published in an IEEE paper. For a detailed understanding 
    of the algorithms and techniques used, please refer to our research paper:
    
    [Predictive Disease Modeling for Proactive Healthcare](https://ieeexplore.ieee.org/document/10581019)
    
    This paper provides insights into the development process, the selection of machine learning models, 
    and the evaluation of their performance in predicting early-stage diseases.
    """)

    st.markdown("---")
    st.subheader("Connect with the Developer")
    st.markdown("""
    - GitHub: [JeelSutariya](https://github.com/JeelSutariya)
    - LinkedIn: [Jeel Sutariya](https://in.linkedin.com/in/jeel-sutariya)
    - Research: [IEEE Xplore](https://ieeexplore.ieee.org/document/10581019)
    """)

    st.markdown("---")
    st.write("""
    This project is for educational purposes only and should not be used as a substitute for professional medical advice.
    """)