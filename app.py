import streamlit as st
import pytesseract
import cv2
import numpy as np
import re
import pandas as pd
from PIL import Image
import io

# =============== CONFIGURATION STREAMLIT ===============
st.set_page_config(page_title="Facture SOMELEC Reader", page_icon="⚡", layout="centered")

st.markdown("""
    <style>
    body {background-color: #f9fafc;}
    h1 {color: #004b8d; text-align:center;}
    .stButton>button {
        background-color:#004b8d;
        color:white;
        border-radius:10px;
        height:3em;
        width:100%;
    }
    .stTable {border-radius:12px;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Lecture automatique de facture SOMELEC")
st.write("Dépose une **photo nette** ou un **scan** de ta facture SOMELEC. L’application identifiera automatiquement les informations du client.")

# =============== UPLOAD IMAGE ===============
uploaded_file = st.file_uploader("📤 Importer une facture (JPG ou PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    # Lire et afficher l’image originale
    image = Image.open(uploaded_file)
    st.image(image, caption="Facture importée", use_column_width=True)
    img_cv = np.array(image.convert("RGB"))

    # =============== PRÉTRAITEMENT IMAGE ===============
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Améliorer contraste et binariser
    result = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41, 15
    )

    # Option de visualisation
    with st.expander("🧠 Aperçu du texte détecté (noir sur blanc)"):
        st.image(result, caption="Zones de texte noir détectées", use_column_width=True)

    # =============== OCR (Tesseract) ===============
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    texte = pytesseract.image_to_string(result, lang="fra", config=custom_config)

    with st.expander("📜 Texte OCR brut détecté"):
        st.text(texte)

    # =============== EXTRACTION PAR REGEX ===============
    champs = {
        "Référence": r"([A-Z0-9]{8,})",
        "Nom Client": r"([A-ZÉÈÂÔÛÎ ]{6,})",
        "Ancien Index": r"Ancien\s*Index[:\s]*([0-9]+)",
        "Nouvel Index": r"Nouvel\s*Index[:\s]*([0-9]+)",
        "Consommation": r"Consommation[:\s]*([0-9]+)",
        "Total à payer": r"Total\s*(?:à|a)\s*Payer[:\s]*([\d\s,\.]+)",
        "Date": r"(\d{1,2}/\d{1,2}/\d{4})"
    }

    results = {}
    for k, pattern in champs.items():
        match = re.search(pattern, texte)
        results[k] = match.group(1).strip() if match else "Non trouvé"

    # =============== AFFICHAGE TABLEAU ===============
    st.markdown("---")
    st.subheader("📊 Informations extraites")
    df = pd.DataFrame(list(results.items()), columns=["Champ", "Valeur"])
    st.table(df)

    # =============== EXPORTATION CSV ===============
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Télécharger les résultats (CSV)", data=csv,
                       file_name="facture_somelec.csv", mime="text/csv")

    st.success("✅ Extraction terminée avec succès !")
else:
    st.info("➡️ Dépose une facture pour commencer l’analyse.")
