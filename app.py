import streamlit as st
import pytesseract
import cv2
import numpy as np
import re
import pandas as pd
from PIL import Image
import io
import base64

# ========================== CONFIGURATION ==========================
st.set_page_config(page_title="Lecture Facture SOMELEC", page_icon="⚡", layout="centered")

# --- Style CSS pro (bleu SOMELEC) ---
st.markdown("""
    <style>
    body {background-color: #f9fafc;}
    h1 {color: #004b8d; text-align:center; margin-bottom: 10px;}
    h2, h3 {color:#004b8d;}
    .stButton>button {
        background-color:#004b8d;
        color:white;
        border-radius:10px;
        height:3em;
        width:100%;
        font-weight:bold;
    }
    table {
        border-radius:10px;
        border:1px solid #004b8d;
    }
    .footer {
        text-align:center;
        color:gray;
        font-size:13px;
        margin-top:40px;
    }
    </style>
""", unsafe_allow_html=True)

# ========================== ENTÊTE ==========================
st.title("⚡ Lecture automatique de factures SOMELEC")
st.write("Téléversez une **photo claire** ou un **scan net** d’une facture SOMELEC. "
         "L’application détecte et extrait automatiquement les informations clés du document.")

# ========================== UPLOAD IMAGE ==========================
uploaded_file = st.file_uploader("📤 Importer une facture (JPG ou PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🧾 Facture importée", use_container_width=True)
    img_cv = np.array(image.convert("RGB"))

    # ========================== PRÉTRAITEMENT OPTIMISÉ ==========================
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # Étape 1 : améliorer contraste local
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Étape 2 : lisser légèrement pour réduire le bruit
    blur = cv2.medianBlur(gray_eq, 3)

    # Étape 3 : seuillage Otsu automatique
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Étape 4 : nettoyer les petits points parasites
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    with st.expander("🧠 Aperçu du prétraitement (équilibré)"):
        st.image(clean, caption="Texte net, fond clair", use_container_width=True)

    # ========================== OCR ==========================
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,./:- --dpi 300'
    texte = pytesseract.image_to_string(clean, lang="fra", config=custom_config)

    with st.expander("📜 Texte détecté par OCR"):
        st.text(texte)

    # ========================== EXTRACTION AVEC REGEX ==========================
    champs = {
        "Référence": r"([A-Z]{2}\s*[0-9]{2}\s*[A-Z0-9 ]{6,})",
        "Nom Client": r"([A-ZÉÈÂÔÛÎ\s]{5,}NOUADHIBOU)",
        "Ancien Index": r"Ancien\s*Index\s*[:\s]*([0-9]{4,6})",
        "Nouvel Index": r"Nouvel\s*Index\s*[:\s]*([0-9]{4,6})",
        "Consommation": r"Consommation\s*[:\s]*([0-9]{2,5})",
        "Total à payer": r"TOTAL\s*(?:A|À)\s*PAYER\s*[:\s]*([\d\s,\.]+)",
        "Date": r"(\d{1,2}/\d{1,2}/\d{4})"
    }

    results = {}
    for champ, pattern in champs.items():
        match = re.search(pattern, texte)
        results[champ] = match.group(1).strip() if match else "Non trouvé"

    # ========================== AFFICHAGE DES RÉSULTATS ==========================
    st.markdown("---")
    st.subheader("📊 Informations extraites")
    df = pd.DataFrame(list(results.items()), columns=["Champ", "Valeur"])
    st.table(df)

    # ========================== EXPORT CSV ==========================
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Télécharger les résultats (CSV)",
        data=csv,
        file_name="facture_somelec.csv",
        mime="text/csv"
    )

    # ========================== GÉNÉRATION PDF (bonus) ==========================
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    def generer_pdf(data_dict):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        c.setTitle("Lecture Facture SOMELEC")
        c.setFont("Helvetica-Bold", 16)
        c.setFillColorRGB(0, 0.3, 0.6)
        c.drawString(180, 800, "FACTURE SOMELEC - OCR")

        c.setFont("Helvetica", 12)
        c.setFillColorRGB(0, 0, 0)
        y = 760
        for k, v in data_dict.items():
            c.drawString(80, y, f"{k} : {v}")
            y -= 25

        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer

    pdf_buffer = generer_pdf(results)
    st.download_button(
        "🧾 Télécharger le rapport PDF",
        data=pdf_buffer,
        file_name="facture_somelec.pdf",
        mime="application/pdf"
    )

    st.success("✅ Extraction terminée avec succès !")
else:
    st.info("➡️ Téléversez une facture pour commencer l’analyse.")

st.markdown('<div class="footer">Développé par 💻 Hamdinou Moulaye Driss — Projet OCR Streamlit</div>', unsafe_allow_html=True)
