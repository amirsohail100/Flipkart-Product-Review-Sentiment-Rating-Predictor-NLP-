from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from joblib import load
from fun import remove_num, remove_emoj, remove_other, remove_punc
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Literal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Max_len = 100
Max_words = 10000
Embedding_dim = 100

model = load('model.pkl')
tokenizer = load('tokenizer.pkl')
COLUMNS = load('columns.pkl')

class Inputs(BaseModel):

    Review : str = Field(
        ...,
        max_length=1000,
        min_length=1,
    )
    Summary : str = Field(
        ...,
        max_length=50,
        min_length=1,
    )


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post('/predict')
def predict(payload : Inputs):

    review = payload.Review
    summary = payload.Summary

    text = review + " " + summary

    text = remove_num(text)
    text = remove_emoj(text)
    text = remove_other(text)
    text = remove_punc(text)

    text_seq = tokenizer.texts_to_sequences(text)
    text_X_pad = pad_sequences(text_seq, maxlen=Max_len, padding="post", truncating="post")

    prediction = model.predict(text_X_pad)

    return {"prediction": prediction}