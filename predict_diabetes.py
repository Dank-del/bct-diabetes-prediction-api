import pydantic, joblib
from numpy import int64


class Param(pydantic.BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float


def predict(params: Param):
    rfc = joblib.load("model.joblib")
    prediction: int64 = rfc.predict(
        [
            [
                params.pregnancies,
                params.glucose,
                params.blood_pressure,
                params.skin_thickness,
                params.insulin,
                params.bmi,
                params.diabetes_pedigree_function,
                params.age,
            ]
        ]
    )
    # print(prediction)
    return {
        "prediction": prediction.item(),
        "message": "Prediction successful",
        "params": params.dict(),
    }