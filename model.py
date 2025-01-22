from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

app = FastAPI()

model = None
training_data = None

class PredictionInput(BaseModel):
    Temperature: float
    Run_Time: float

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    global training_data
    try:
        training_data = pd.read_csv(file.file)
        return {"message": "File uploaded successfully", "columns": list(training_data.columns)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/train")
async def train_model():
    global model, training_data
    if training_data is None:
        return {"error": "No data uploaded. Please upload data first."}

    try:
        X = training_data[["Temperature", "Run_Time"]]
        y = training_data["Downtime_Flag"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        joblib.dump(model, "model.pkl")

        return {"accuracy": accuracy, "f1_score": f1}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
async def predict(data: PredictionInput):
    global model
    if model is None:
        return {"error": "No model trained. Please train the model first."}

    try:
        input_data = pd.DataFrame([data.dict()])
        prediction = model.predict(input_data)[0]
        confidence = max(model.predict_proba(input_data)[0])
        return {"Downtime": "Yes" if prediction == 1 else "No", "Confidence": confidence}
    except Exception as e:
        return {"error": str(e)}


