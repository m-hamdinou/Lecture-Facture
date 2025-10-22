import streamlit as st
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import json

# ===================== CONFIGURATION =====================
st.set_page_config(page_title="AI Facture Reader (SOMELEC)", page_icon="⚡", layout="centered")

st.markdown("""
<style>
body {background-color: #f9fafc;}
h1 {color:#004b8d; text-align:center;}
.stButton>button {
    background-color:#004b8d;
    color:white;
    border-radius:10px;
    height:3em;
    width:100%;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ Lecture intelligente de facture SOMELEC")
st.write("Cette application IA utilise le modèle **Donut (Document Understanding Transformer)** "
         "pour lire et comprendre automatiquement les informations d'une facture, "
         "sans aucune règle manuelle ou regex.")

# ===================== UPLOAD IMAGE =====================
uploaded_file = st.file_uploader("📤 Importer une facture (JPG ou PNG)", type=["jpg", "png"])

@st.cache_resource
def load_model():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    return processor, model

processor, model = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🧾 Facture importée", use_container_width=True)
    st.info("⏳ Analyse IA en cours...")

    pixel_values = processor(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model.generate(pixel_values, max_length=512)

    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    try:
        json_result = json.loads(result)
    except:
        json_result = {"Résultat brut": result}

    st.success("✅ Analyse terminée ! Voici les informations détectées :")
    st.json(json_result)

    json_bytes = json.dumps(json_result, indent=4).encode("utf-8")
    st.download_button(
        "⬇️ Télécharger les résultats (JSON)",
        data=json_bytes,
        file_name="facture_somelec_ia.json",
        mime="application/json"
    )
else:
    st.info("➡️ Importez une image de facture pour lancer l'analyse IA.")
