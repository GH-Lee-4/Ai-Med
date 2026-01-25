"""
Example usage of the Symptoms Diagnosis Agent
This file demonstrates how to use the symptoms diagnosis functionality
"""

from backend.symptoms_diagnosis import diagnose_symptoms, SymptomsDiagnosisAgent


def example_usage():
    """Example of how to use the symptoms diagnosis agent"""
    
    # Example 1: Simple diagnosis
    print("=" * 60)
    print("Example 1: Chest Pain Diagnosis")
    print("=" * 60)
    result = diagnose_symptoms("I have severe chest pain and shortness of breath")
    print(f"Urgency: {result['urgency_level']}")
    print(f"Matched Symptoms: {result['matched_symptoms']}")
    print(f"Possible Causes: {result['possible_causes']}")
    print(f"Possible Risks: {result['possible_risks']}")
    print("Recommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
    print()
    
    # Example 2: Diagnosis with patient age
    print("=" * 60)
    print("Example 2: Fever Diagnosis (Elderly Patient)")
    print("=" * 60)
    result = diagnose_symptoms("I have a high fever and chills", patient_age=72)
    print(f"Urgency: {result['urgency_level']}")
    print(f"Possible Causes: {result['possible_causes'][:3]}")  # Show first 3
    print(f"Possible Risks: {result['possible_risks']}")
    print()
    
    # Example 3: Using the agent directly
    print("=" * 60)
    print("Example 3: Using Agent Directly")
    print("=" * 60)
    agent = SymptomsDiagnosisAgent()
    result = agent.diagnose(
        "I'm experiencing dizziness and nausea after eating",
        patient_age=45,
        additional_info="No known allergies"
    )
    print(f"Urgency Priority: {result['urgency_priority']}")
    print(f"Urgency Level: {result['urgency_level']}")
    print(f"Matched {result['matched_symptom_count']} symptom(s)")
    print(f"Possible Causes: {', '.join(result['possible_causes'][:5])}")
    print()


if __name__ == "__main__":
    example_usage()

