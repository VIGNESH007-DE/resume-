import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

jobs = pd.read_csv("jobs.csv")

# 🔍 FIX: combine columns safely
if 'description' in jobs.columns:
    job_text = jobs['description'].astype(str)
elif 'skills' in jobs.columns and 'title' in jobs.columns:
    job_text = jobs['title'].astype(str) + " " + jobs['skills'].astype(str)
elif 'job_post' in jobs.columns:
    job_text = jobs['job_post'].astype(str)
else:
    raise ValueError("No valid job text column found in CSV")

job_embeddings = model.encode(job_text.tolist())

def match_jobs(resume_text):
    resume_vector = model.encode([resume_text])[0]

    scores = cosine_similarity([resume_vector], job_embeddings)[0]

    jobs['score'] = scores

    return jobs.sort_values(by='score', ascending=False)
