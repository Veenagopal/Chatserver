from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def root():
    return {"message": "Hello Boss! CSPRNG API is ready "}