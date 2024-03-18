import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
# from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as fileobj:
        pdf_reader = PdfReader(fileobj)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def extract_tables_from_pdf(pdf_path):
    with open(pdf_path, "rb") as fileobj:
        pdf_reader = PdfReader(fileobj)
        num_pages = len(pdf_reader.pages)
        tables = []

        for page_num in range(num_pages):
            page_text = pdf_reader.pages[page_num].extract_text()
            # Here you can perform table extraction from page_text using camelot or any other library
            # Append extracted tables to the tables list

    return tables


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer will be very specific.Context of the pdf is  patient report about the breast cancer.Tell  the demographic information ,the gene identification,the variant and its type ,recommended treatments,clinical trial for a particular gene,the variants and their classificatin tiers from the report.
    If you dont know the answer, still you will give the answer from your knowledge base\n\n                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def suggest_websites(user_question):
    website_mapping = {
        "Alpelisib": ["https://medlineplus.gov/druginfo/meds/a619036.html", "https://go.drugbank.com/drugs/DB12015"],
        "Neratinib": ["https://go.drugbank.com/drugs/DB11828", "https://www.nationalbreastcancer.org"],
        "Lapatinib": ["https://en.wikipedia.org/wiki/Lapatinib#:~:text=Lapatinib%20(INN)%2C%20used%20in,factor%20receptor%20(EGFR)%20pathways.", "https://go.drugbank.com/drugs/DB01259"],
        "Pertuzumab":["https://www.perjeta.com/#:~:text=of%20coming%20back.-,PERJETA%C2%AE%20(pertuzumab)%20is%20a%20prescription%20medicine%20approved%20for%20use,chemotherapy%20for%20metastatic%20breast%20cancer"],
        "Trastuzumab":["https://www.cancerresearchuk.org/about-cancer/treatment/drugs/trastuzumab#:~:text=What%20is%20trastuzumab%3F,same%20side%20of%20the%20body."]
    }
    suggested_websites = []
    for topic, websites in website_mapping.items():
        if topic.lower() in user_question.lower():
            suggested_websites.extend(websites)
    return suggested_websites


def user_input(user_question, pdf_path):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Chatbot Reply: ", response["output_text"])

    # Get related websites based on user question
    related_websites = suggest_websites(user_question)

    if related_websites:
        st.write("Suggested Websites:")
        for website in related_websites:
            st.write(website)
    else:
        st.write("No relevant websites found.")


def main():
    st.set_page_config("Genesilico-HealthBOT")
    st.header("TumorGenie")

    user_question = st.text_input("Ask a Question from the PDF Files")
    pdf_file = st.file_uploader("Upload your PDF file")

    if pdf_file:
        st.write("PDF Uploaded Successfully")

    if st.button("Process PDF") and pdf_file:
        with st.spinner("Processing PDF..."):
            # Save the uploaded file to a temporary location
            with open("temp_pdf.pdf", "wb") as f:
                f.write(pdf_file.getvalue())

            # Extract text from the PDF
            pdf_text = extract_text_from_pdf("temp_pdf.pdf")
            text_chunks = get_text_chunks(pdf_text)
            get_vector_store(text_chunks)

            # Extract tables from the PDF
            pdf_tables = extract_tables_from_pdf("temp_pdf.pdf")

            st.success("PDF Processing Completed")

    if user_question:
        user_input(user_question, "temp_pdf.pdf")


if __name__ == "__main__":
    main()
