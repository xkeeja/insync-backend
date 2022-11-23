from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
# app.state.model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# @app.get("/predict")
# def predict():
#     return {'fare_amount': float(3)}


@app.get("/")
def root():
    return {'greeting': 'Hello from in_sync'}
