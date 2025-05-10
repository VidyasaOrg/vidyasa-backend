from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to Vidyasa Backend!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}