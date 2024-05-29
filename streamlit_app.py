import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

# Türkçe metin özetleme modeli ve tokenizer yükleniyor
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Streamlit arayüzü oluşturuluyor
st.title("Türkçe Metin Özetleme Uygulaması")
text_input = st.text_area("Metni buraya girin")

if st.button("Metni Özetle"):
    input_ids = tokenizer(text_input, return_tensors="pt").input_ids
    summary_ids = model.generate(input_ids, num_beams=4, max_length=100, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    st.write("Özetlenmiş Metin:")
    st.write(summary)