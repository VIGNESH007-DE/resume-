import streamlit as st

from utils import extract_text

from model import (
    match_jobs,
    extract_skills,
    missing_skills
)

# Page settings
st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="centered"
)

# Title
st.title(
    "📄 AI Resume Screening & Job Matching System"
)

st.write(
    "Upload your resume PDF and get AI-powered job recommendations."
)

# File upload
uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success(
        "Resume uploaded successfully!"
    )

    # Extract text
    resume_text = extract_text(
        uploaded_file
    )

    # Extract skills
    skills = extract_skills(
        resume_text
    )

    st.subheader(
        "🧠 Detected Skills"
    )

    if len(skills) > 0:

        for skill in skills:
            st.write(f"✅ {skill}")

    else:
        st.warning(
            "No skills detected."
        )

    # Match jobs
    results = match_jobs(
        resume_text
    )

    st.subheader(
        "🎯 Top Job Matches"
    )

    # Show top matches
    for index, row in results.head(5).iterrows():

        st.write(
            f"## {row['title']}"
        )

        st.progress(
            int(row['match_score'])
        )

        st.write(
            f"Match Score: {row['match_score']:.2f}%"
        )

        # Missing skills
        missing = missing_skills(
            resume_text,
            row['skills']
        )

        st.write(
            "### Missing Skills"
        )

        if len(missing) > 0:

            for m in missing:
                st.write(f"❌ {m}")

        else:
            st.write(
                "No missing skills."
            )

        st.write("---")
