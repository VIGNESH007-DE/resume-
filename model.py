import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


# Load dataset
jobs = pd.read_csv("jobs.csv")


# Skill database
skills_list = [

    "python",
    "machine learning",
    "deep learning",
    "sql",
    "power bi",
    "tableau",
    "tensorflow",
    "pytorch",
    "nlp",
    "computer vision",
    "react",
    "html",
    "css",
    "javascript",
    "django",
    "flask",
    "aws",
    "docker",
    "kubernetes",
    "linux",
    "ethical hacking",
    "network security",
]


# TF-IDF
vectorizer = TfidfVectorizer()

job_vectors = vectorizer.fit_transform(jobs['skills'])


# Extract skills
def extract_skills(text):

    found_skills = []

    for skill in skills_list:

        if skill in text:
            found_skills.append(skill)

    return found_skills


# Match jobs
def match_jobs(resume_text):

    resume_vector = vectorizer.transform([resume_text])

    similarity = cosine_similarity(
        resume_vector,
        job_vectors
    )

    jobs['match_score'] = similarity[0] * 100

    results = jobs.sort_values(
        by='match_score',
        ascending=False
    )

    return results


# Missing skills
def missing_skills(
    resume_text,
    job_skills
):

    resume_words = set(
        resume_text.split()
    )

    job_words = set(
        job_skills.split()
    )

    missing = job_words - resume_words

    return list(missing)
