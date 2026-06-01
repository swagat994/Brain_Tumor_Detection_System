from fastapi import FastAPI

from backend.routes import router


app = FastAPI(
    title="Brain Tumor Detection API",
    version="1.0"
)

app.include_router(
    router
)


@app.get("/")
def home():

    return {
        "message":
        "Brain Tumor Detection API Running"
    }