import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
jobs = pd.read_csv("jobs.csv")

# -----------------------------
# CREATE JOB TEXT FROM YOUR COLUMNS
# -----------------------------
job_text = (
    jobs['Company'].astype(str) + " " +
    jobs['Post'].astype(str) + " " +
    jobs['Location'].astype(str) + " " +
    jobs['Experience'].astype(str) + " " +
    jobs['Stipend'].astype(str)
)

# Create embeddings
job_embeddings = model.encode(job_text.tolist())

# -----------------------------
# MATCH FUNCTION
# -----------------------------
def match_jobs(resume_text):
    resume_vector = model.encode([resume_text])[0]

    scores = cosine_similarity([resume_vector], job_embeddings)[0]

    jobs['score'] = scores

    return jobs.sort_values(by='score', ascending=False)
