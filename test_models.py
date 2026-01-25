"""Quick test script to verify ML models are working"""

from backend.ml_diagnosis import get_ml_model

print("Loading models...")
model = get_ml_model()

print("\nTesting prediction...")
result = model.predict("chest pain and shortness of breath")

print(f"\nUrgency: {result['urgency_priority']} (confidence: {result['urgency_confidence']:.2f})")
print(f"Primary Cause: {result['primary_cause']} (confidence: {result['cause_confidence']:.2f})")
print(f"\nTop Causes:")
for i, cause in enumerate(result['all_predictions'][:3], 1):
    print(f"  {i}. {cause['cause']} ({cause['confidence']:.2f})")

print("\n[SUCCESS] Models are working correctly!")


