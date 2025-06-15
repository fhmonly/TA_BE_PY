from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.prediction import router as prediction_route
from routes.protected_prediction import router as protected_prediction

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(prediction_route, prefix="/api")
app.include_router(protected_prediction, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to ARIMA Prediction API (JSON version)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)