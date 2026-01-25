from openai import OpenAI
import os

# Initialize OpenAI client (will use OPENAI_API_KEY from environment)
client = None
try:
    client = OpenAI()
except Exception as e:
    print(f"Warning: OpenAI client initialization failed: {e}")

def chat(symptoms, image_path=None):
    """
    Chat function for medical diagnosis using OpenAI and RAG
    
    Args:
        symptoms: Patient symptoms description
        image_path: Optional path to medical image
        
    Returns:
        AI assistant response string
    """
    if client is None:
        raise ValueError("OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")
    
    # Try to retrieve context (with fallback if vector store not available)
    try:
        from retrieve import retrieve_context
        context = retrieve_context(symptoms)
    except Exception as e:
        context = "No additional context available from knowledge base."
        print(f"Warning: Context retrieval failed: {e}")
    
    # Try to analyze image if provided
    image_info = ""
    if image_path:
        try:
            from image_model import predict_image
            preds = predict_image(image_path)
            image_info = f"Image analysis: {preds}"
        except Exception as e:
            image_info = "Image analysis unavailable."
            print(f"Warning: Image analysis failed: {e}")

    prompt = f"""
You are a medical assistant.
Use ONLY the information below.

{context}

{image_info}

User symptoms: {symptoms}

Give a cautious, non-diagnostic response.
"""

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",  # Fixed model name (gpt-4.1-mini doesn't exist)
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return res.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")
