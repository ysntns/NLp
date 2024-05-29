import transformers
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import PyPDF2
import streamlit as st
from googletrans import Translator

# Türkçe metin özetleme modeli ve tokenizer yükleniyor
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
translator = Translator()
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Desteklenen diller
languages = {
    "Türkçe": "tr",
    "İngilizce": "en",
    "İspanyolca": "es",
    "Fransızca": "fr",
    "Almanca": "de",
    "Çince": "zh-cn",
    "Japonca": "ja",
    "Rusça": "ru",
    "Arapça": "ar"
}

def summarize_pdf(file, target_language):
    # PDF'den metin oku
    text = read_pdf(file)

    # Metin çok uzunsa parçalara böl
    max_chunk_size = 1000
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    summaries = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs["input_ids"]
        summary_ids = model.generate(input_ids, num_beams=4, max_length=150, min_length=30, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    # Tüm özetleri birleştir
    combined_summary = " ".join(summaries)

    # Özeti hedef dile çevir
    translated_summary = translator.translate(combined_summary, dest=languages[target_language]).text

    return translated_summary

def read_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def main():
    st.title("PDF ve Metin Özetleyici")
    st.markdown("PDF dosyası veya metin girin ve içeriğinin özetini belirlediğiniz dile çevirin.")

    option = st.radio("Ne özetlemek istiyorsunuz?", ('PDF', 'Metin'))

    if option == 'PDF':
        file = st.file_uploader("PDF dosyası yükleyin", type="pdf")
        if file is not None:
            target_language = st.selectbox("Dil Seçin", list(languages.keys()))
            if st.button("Özetle ve Çevir"):
                summary = summarize_pdf(file, target_language)
                st.write("Özetlenmiş ve Çevrilmiş Metin:")
                st.write(summary)

    elif option == 'Metin':
        text_input = st.text_area("Metni buraya girin")
        target_language = st.selectbox("Dil Seçin", list(languages.keys()), key="text")
        if st.button("Metni Özetle"):
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = inputs["input_ids"]
            summary_ids = model.generate(input_ids, num_beams=4, max_length=100, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            translated_summary = translator.translate(summary, dest=languages[target_language]).text

            st.write("Özetlenmiş ve Çevrilmiş Metin:")
            st.write(translated_summary)

if __name__ == "__main__":
    main()
