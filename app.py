import streamlit as st
import pytesseract
import cv2
import numpy as np
import re
import pandas as pd
from PIL import Image

# ---------------- CONFIGURATION STREAMLIT ----------------
st.set_page_config(page_title="Facture SOMELEC Reader", page_icon="⚡", layout="centered")

st.title("⚡ Lecture automatique de facture SOMELEC")
st.markdown("Dépose une **photo nette** de la facture SOMELEC pour extraire automatiquement les informations du client.")

# ---------------- UPLOAD IMAGE ----------------
uploaded_file = st.file_uploader("📤 Importer une facture (JPG ou PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    # Lire et afficher l’image originale
    image = Image.open(uploaded_file)
    st.image(image, caption="Facture importée", use_column_width=True)

    # Conversion en tableau OpenCV
    img_cv = np.array(image.convert("RGB"))

    # ---------------- FILTRAGE DES TEXTES NOIRS ----------------
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Inversion (texte noir -> blanc)
    result = cv2.bitwise_not(mask_black)

    # Affichage debug
    with st.expander("🧠 Zones détectées (texte noir uniquement)"):
        st.image(result, caption="Filtrage noir", use_column_width=True)

    # ---------------- OCR ----------------
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,./:-'
    texte = pytesseract.image_to_string(result, lang="fra", config=custom_config)

    with st.expander("📜 Texte détecté par OCR"):
        st.text(texte)

    # ---------------- EXTRACTION AVEC REGEX ----------------
    data = {}

    data["Référence"] = re.search(r'([A-Z0-9]{8,})', texte)
    data["Nom Client"] = re.search(r'([A-Z ]{6,})', texte)
    data["Ancien Index"] = re.search(r'Ancien\s*Index[:\s]*([0-9]+)', texte)
    data["Nouvel Index"] = re.search(r'Nouvel\s*Index[:\s]*([0-9]+)', texte)
    data["Consommation"] = re.search(r'Consommation[:\s]*([0-9]+)', texte)
    data["Total à payer"] = re.search(r'Total\s*(?:à|a)\s*Payer[:\s]*([\d\s,\.]+)', texte)
    data["Date"] = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', texte)

    # Nettoyage
    results = {k: (v.group(1).strip() if v else "Non trouvé") for k, v in data.items()}

    # ---------------- AFFICHAGE ----------------
    st.subheader("📊 Informations extraites")
    df = pd.DataFrame(list(results.items()), columns=["Champ", "Valeur"])
    st.table(df)

    # ---------------- EXPORT CSV ----------------
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Télécharger les résultats (CSV)", data=csv, file_name="facture_somelec.csv", mime="text/csv")

    st.success("✅ Extraction terminée avec succès ! Vérifie les valeurs ci-dessus.")
else:
    st.info("➡️ Charge une facture pour commencer l’analyse.")
