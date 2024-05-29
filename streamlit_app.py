import transformers
from transformers import pipeline
import PyPDF2
import streamlit as st
from googletrans import Translator
from transformers import BartTokenizer, BartForConditionalGeneration

# Türkçe metin özetleme modeli ve tokenizer yükleniyor
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Streamlit arayüzü oluşturuluyor
st.title("Türkçe Metin Özetleme Uygulaması")
text_input = st.text_area("Metni buraya girin")

def summarize_pdf(file, target_language):
    # PDF'den metin oku
    text = read_pdf(file)

    # Metni özetle
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

    # Özeti hedef dile çevir
    translated_summary = translator.translate(summary, dest=languages[target_language]).text

    return translated_summary


def read_pdf(file):
    text = ""
    with open(file.name, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


if st.button("Metni Özetle"):
    input_ids = tokenizer(text_input, return_tensors="pt").input_ids
    summary_ids = model.generate(input_ids, num_beams=4, max_length=100, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    st.write("Özetlenmiş Metin:")
    st.write(summary)
