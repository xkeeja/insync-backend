from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .dl_from_bucket import blob_retrieval

app = FastAPI()
# app.state.model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/process")
def process(file_name: str):
    contents = blob_retrieval(file_name)
    return {'blob_contents': str(contents)}


@app.get("/")
def root():
    return {'greeting': 'Hello from in_sync'}
