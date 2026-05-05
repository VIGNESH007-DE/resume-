import streamlit as st
from model import match_jobs
from utils import extract_text

st.title("🥈 Smart Job Matching AI")

uploaded_file = st.file_uploader("Upload Resume (PDF)")

if uploaded_file:
    resume_text = extract_text(uploaded_file)

    results = match_jobs(resume_text)

    st.subheader("Top Job Matches")

    for i, row in results.head(5).iterrows():
        st.write("###", row['title'])
        st.write("Score:", round(row['score'], 2))
        st.write("---")
