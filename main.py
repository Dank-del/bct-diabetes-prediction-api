from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse
import pandas as pd
from predict_diabetes import predict
import joblib, uvicorn, logging
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

app = FastAPI(title="Diabetes Prediction API", version="1.0.0")
log = logging.getLogger("uvicorn.info")

@app.get("/")
async def root():
    return {"message": "API running"}

@app.post("/predict")
async def predict_diabetes(common = Depends(predict)):
    return common

@app.get("/get-heatmap")
async def get_heatmap():
    return FileResponse("heatmap.png")

@app.get("/get-histogram")
async def get_histogram():
    return FileResponse("histogram.png")
    


@app.on_event("startup")
async def startup_event():
    log.info("Reading the model")
    diabetes_df = pd.read_csv("diabetes.csv")
    p = sns.heatmap(diabetes_df.corr(), annot=True,cmap ='RdYlGn')
    log.info("Saved heatmap of the training data")
    p.get_figure().savefig("heatmap.png")
    hist = diabetes_df.plot.hist(figsize=(10,10))
    hist.get_figure().savefig("histogram.png")
    log.info("Saved histogram of the training data")
    X = diabetes_df.drop("Outcome", axis=1)
    y = diabetes_df["Outcome"]
    log.info("Training the model")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=7
    )
    fill_values = SimpleImputer(missing_values=0, strategy="mean")
    X_train = fill_values.fit_transform(X_train)
    X_test = fill_values.fit_transform(X_test)
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)
    log.info("Saving the model")
    joblib.dump(rfc, "model.joblib")
    log.info("Model saved")
    
if __name__ == "__main__":
    uvicorn.run(app)