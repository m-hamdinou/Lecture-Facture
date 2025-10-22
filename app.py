import streamlit as st
import pytesseract
import cv2
import numpy as np
import re
import pandas as pd
from PIL import Image

# ====================== CONFIGURATION STREAMLIT ======================
st.set_page_config(page_title="Facture SOMELEC Reader", page_icon="⚡", layout="centered")

st.markdown("""
<style>
body {background-color: #f9fafc;}
h1 {color: #004b8d; text-align:center;}
h2, h3 {color:#004b8d;}
.stButton>button {
    background-color:#004b8d;
    color:white;
    border-radius:10px;
    height:3em;
    width:100%;
}
table {
    border-radius:10px;
    border:1px solid #004b8d;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ Lecture automatique de factures SOMELEC")
st.write("Déposez une **photo claire** ou un **scan** de votre facture SOMELEC. "
         "L’application détectera automatiquement les informations importantes (référence, index, total, etc.).")

# ====================== UPLOAD IMAGE ======================
uploaded_file = st.file_uploader("📤 Importer une facture (JPG ou PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    # Lecture de l’image
    image = Image.open(uploaded_file)
    st.image(image, caption="🧾 Facture importée", use_column_width=True)

    img_cv = np.array(image.convert("RGB"))

    # ====================== PRÉTRAITEMENT DOUX ======================
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # Étape 1 : égalisation adaptative du contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Étape 2 : léger flou pour réduire le bruit
    blur = cv2.GaussianBlur(gray_eq, (3, 3), 0)

    # Étape 3 : seuillage doux automatique (Otsu)
    _, result = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    with st.expander("🧠 Aperçu du prétraitement"):
        st.image(result, caption="Texte noir sur fond clair (optimisé)", use_column_width=True)

    # ====================== OCR (TESSERACT) ======================
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    texte = pytesseract.image_to_string(result, lang="fra", config=custom_config)

    with st.expander("📜 Texte détecté par OCR"):
        st.text(texte)

    # ====================== EXTRACTION PAR REGEX ======================
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
    for champ, pattern in champs.items():
        match = re.search(pattern, texte)
        results[champ] = match.group(1).strip() if match else "Non trouvé"

    # ====================== AFFICHAGE DES RÉSULTATS ======================
    st.markdown("---")
    st.subheader("📊 Informations extraites")
    df = pd.DataFrame(list(results.items()), columns=["Champ", "Valeur"])
    st.table(df)

    # ====================== EXPORT CSV ======================
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Télécharger les résultats (CSV)",
        data=csv,
        file_name="facture_somelec.csv",
        mime="text/csv"
    )

    st.success("✅ Extraction terminée avec succès ! Vérifiez les valeurs ci-dessus.")
else:
    st.info("➡️ Déposez une facture pour commencer l’analyse.")
