from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.predict_file import router as predict_file_router
from routes.predict_json import router as predict_json_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(predict_file_router, prefix="/api")
app.include_router(predict_json_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to ARIMA Prediction API (JSON version)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)