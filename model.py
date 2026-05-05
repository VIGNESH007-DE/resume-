import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

jobs = pd.read_csv("jobs.csv")

job_embeddings = model.encode(jobs['description'].tolist())

def match_jobs(resume_text):
    resume_vec = model.encode([resume_text])[0]

    scores = cosine_similarity([resume_vec], job_embeddings)[0]

    jobs['score'] = scores

    return jobs.sort_values(by='score', ascending=False)
