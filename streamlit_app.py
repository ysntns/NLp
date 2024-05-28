# streamlit_app.py

import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import warnings
warnings.filterwarnings("ignore")

def pd_optns():
    pd.set_option('display.max_columns', None)
    pd.set_option("display.max_rows", None)
    pd.set_option('display.width', 500)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.float_format', lambda x: "%.1f" % x)
    warnings.simplefilter(action='ignore', category=Warning)

pd_optns()



# Modellerin isimleri
model_name = {
    "tr": "savasy/bert-base-turkish-squad",
    "en": "bert-large-uncased-whole-word-masking-finetuned-squad"
}

# Modelleri ve tokenizerları yükleme
models = {}
tokenizers = {}

for lang in ["tr", "en"]:
    models[lang] = AutoModelForQuestionAnswering.from_pretrained(model_name[lang])
    tokenizers[lang] = AutoTokenizer.from_pretrained(model_name[lang])


def question_answer(question, text, lang):
    if lang not in ["tr", "en"]:
        raise ValueError("Geçersiz dil seçimi. Sadece 'tr' ve 'en' desteklenmektedir.")

    tokenizer = tokenizers[lang]
    model = models[lang]

    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer


# Örnek kullanım
question = "Türkiye'nin başkenti neresidir?"
text = "Türkiye'nin başkenti Ankara'dır."
lang = "tr"

print(question_answer(question, text, lang))

# Streamlit Interface
st.sidebar.title("Developer")
st.sidebar.image("foto.png")
st.sidebar.write("YASIN TANIS")
st.sidebar.write("Data Science & Analytics / Machine Learning Expert / CRM / NLP / SQL /  ")
st.sidebar.write("Conctact : https://bento.me/ysntns")

st.title("Çok Dilli Soru-Cevap Uygulaması")
st.write("Bu uygulama, bir metne dayalı olarak sorularınıza cevap vermek için BERT modelini kullanır.")

text = st.text_area("Lütfen metni giriniz:", height=300)
question = st.text_input("Lütfen sorunuzu giriniz:")
lang = st.selectbox("Lütfen dili seçiniz:", ["tr", "en"])

if st.button("Soruyu Sor"):
    if text and question:
        answer = question_answer(question, text, lang)
        st.write("Cevap: ", answer)
    else:
        st.write("Lütfen metin ve soru alanlarını doldurunuz.")
