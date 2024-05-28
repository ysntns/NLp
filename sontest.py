# Gerekli Kütüphanelerin İçe Aktarılması ve Ayarların Yapılması

import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import streamlit as st
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

# Verinin Yüklenmesi ve Temizlenmesi

# Verinin Stanford web sitesinden yüklenmesi
coqa = pd.read_json('http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json')
coqa.head()

# 'version' sütununun silinmesi
del coqa["version"]

# Her soru-cevap çifti için ilgili hikayeyi ekleme
cols = ["text","question","answer"]
comp_list = []

for index, row in coqa.iterrows():
    for i in range(len(row["data"]["questions"])):
        temp_list = []
        temp_list.append(row["data"]["story"])
        temp_list.append(row["data"]["questions"][i]["input_text"])
        temp_list.append(row["data"]["answers"][i]["input_text"])
        comp_list.append(temp_list)

new_df = pd.DataFrame(comp_list, columns=cols)
# Veri çerçevesini CSV dosyasına kaydetme
new_df.to_csv("CoQA_data.csv", index=False)

# Yerel CSV dosyasından veriyi yükleme
data = pd.read_csv("CoQA_data.csv")
data.head()

print("Number of question and answers: ", len(data))


# Modellerin Yüklenmesi ve Soru-Cevap İşlevselliği

# Modellerin isimleri
model_name = {
    "tr": "savasy/bert-base-turkish-squad",
    "en": "bert-large-uncased-whole-word-masking-finetuned-squad"
}

# Modelleri ve tokenizer'ları yükleme
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


# Streamlit Arayüzü

# Streamlit Arayüzü
st.sidebar.title("Developer")
st.sidebar.image("foto.png")
st.sidebar.write("YASIN TANIS")
st.sidebar.write("Data Science & Analytics / Machine Learning Expert / CRM / NLP / SQL /")
st.sidebar.write("Conctact: https://bento.me/ysntns")

st.title("Çok Dilli Soru-Cevap Uygulaması")
st.write("Bu uygulama, bir metne dayalı olarak sorularınıza cevap vermek için BERT modelini kullanır.")

text = st.text_area("Lütfen metni giriniz:", height=300)
question = st.text_input("Lütfen sorunuzu giriniz:")
lang = st.selectbox("Lütfen dili seçiniz:", ["Türkçe", "İngilizce"])

if st.button("Soruyu Sor"):
    if text and question:
        answer = question_answer(question, text, lang[:2].lower())  # 'tr' veya 'en' olarak dil kodu gönderiyoruz
        st.write("Cevap: ", answer)
    else:
        st.write("Lütfen metin ve soru alanlarını doldurunuz.")
