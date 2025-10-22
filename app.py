import streamlit as st
import pytesseract
import cv2
import numpy as np
import re
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Lecture Facture SOMELEC", page_icon="⚡", layout="centered")

st.title("⚡ Lecture automatique de facture SOMELEC")

uploaded_file = st.file_uploader("Dépose ta facture ici (format JPG ou PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    # Lire l'image
    image = Image.open(uploaded_file)
    img_cv = np.array(image)
    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # OCR : lecture du texte complet
    texte = pytesseract.image_to_string(gray, lang="fra")

    # Afficher le texte brut (debug)
    with st.expander("🧾 Texte détecté par OCR"):
        st.text(texte)

    # Extraction par regex
    data = {}
    data["Référence"] = re.search(r"([A-Z0-9]{8,})", texte)
    data["Nom Client"] = re.search(r"([A-Z ]{5,})", texte)
    data["Ancien Index"] = re.search(r"Ancien\s*Index[:\s]*([0-9]+)", texte)
    data["Nouvel Index"] = re.search(r"Nouvel\s*Index[:\s]*([0-9]+)", texte)
    data["Consommation"] = re.search(r"Consommation[:\s]*([0-9]+)", texte)
    data["Total à payer"] = re.search(r"Total\s*(?:à|a)\s*Payer[:\s]*([\d\s,\.]+)", texte)
    data["Date"] = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", texte)

    # Nettoyage des résultats
    results = {k: (v.group(1) if v else "Non trouvé") for k, v in data.items()}

    # Afficher dans un tableau
    df = pd.DataFrame(list(results.items()), columns=["Champ", "Valeur"])
    st.table(df)

    # Option d’export
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Télécharger les résultats (CSV)", data=csv, file_name="facture_somelec.csv", mime="text/csv")
