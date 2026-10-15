import argparse
from pathlib import Path
import joblib
import pandas as pd

parser = argparse.ArgumentParser(description="Predict passenger survival on Titanic.")
parser.add_argument("--pclass", type=int, required=True, help="Ticket class (1, 2, 3)")
parser.add_argument("--sex", type=str, required=True, help="female or male")
parser.add_argument("--age", type=float, required=True, help="Age in years")
parser.add_argument("--fare", type=float, required=True, help="Passenger fare")
parser.add_argument("--sibsp", type=int, default=0, help="Number of siblings/spouses aboard")
parser.add_argument("--parch", type=int, default=0, help="Number of parents/children aboard")
parser.add_argument("--embarked", type=str, default="S", help="Port of embarkation (C, Q, S)")
args = parser.parse_args()

PROJECT_ROOT = Path.cwd().parent
MODEL_PATH = PROJECT_ROOT / "models" / "titanic_survival.joblib"

model = joblib.load(MODEL_PATH)

passenger = pd.DataFrame([{
    "Pclass": args.pclass,
    "Sex": args.sex,
    "Age": args.age,
    "SibSp": args.sibsp,
    "Parch": args.parch,
    "Fare": args.fare,
    "Embarked": args.embarked,
}])

prob = model.predict_proba(passenger)[0, 1]
pred = int(prob >= 0.5)

print(f"Predicted Class: {pred} ({'Survived' if pred == 1 else 'Did not survive'})")
print(f"Survival Probability: {prob:.4f}")
