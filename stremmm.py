from nlpCoQa import *


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
