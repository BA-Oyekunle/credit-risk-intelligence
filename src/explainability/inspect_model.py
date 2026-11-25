
import joblib
from pathlib import Path

MODEL_PATH = Path("artifacts/best_model.pkl")

print("=" * 70)
print("CREDIT RISK INTELLIGENCE")
print("MODEL PIPELINE INSPECTION")
print("=" * 70)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH.resolve()}"
    )

model = joblib.load(MODEL_PATH)

print("\nMODEL TYPE:")
print(type(model))

print("\nMODEL REPRESENTATION:")
print(model)

if hasattr(model, "named_steps"):
    print("\nPIPELINE STEPS:")

    for step_name, step_object in model.named_steps.items():
        print(f"\nStep: {step_name}")
        print(f"Class: {type(step_object)}")

        if hasattr(step_object, "get_feature_names_out"):
            try:
                names = step_object.get_feature_names_out()

                print("Transformed feature count:", len(names))
                print("First 30 transformed features:")

                for name in names[:30]:
                    print(" -", name)

            except Exception as error:
                print("Could not retrieve feature names:", error)

    final_step_name = list(model.named_steps.keys())[-1]
    estimator = model.named_steps[final_step_name]

    print("\nFINAL ESTIMATOR:")
    print(type(estimator))

    if hasattr(estimator, "feature_importances_"):
        print("\nFEATURE IMPORTANCE AVAILABLE")

    if hasattr(estimator, "get_booster"):
        print("\nXGBOOST MODEL DETECTED")

else:
    print("\nThe saved model is not a standard sklearn Pipeline.")

    if hasattr(model, "feature_importances_"):
        print("Feature importance is available.")

print("\nMODEL INSPECTION COMPLETED.")
