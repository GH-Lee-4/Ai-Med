import streamlit as st
import numpy as np
import cv2
from PIL import Image
from backend.camera import capture_image
from backend.img_process import process_image
from backend.model import (
    predict_diagnosis, classify_normal_abnormal, segment_problem_area,
    calculate_risk_score, classify_chest_xray, classify_skin_lesion,
    classify_bone_fracture, get_image_type,
    detect_brain_abnormalities, detect_abdominal_abnormalities,
    detect_retinal_abnormalities, detect_mammography_abnormalities,
    detect_chest_abnormalities, detect_bone_abnormalities
)
from backend.symptoms_diagnosis import diagnose_symptoms

# Import chat function (with error handling in case dependencies are missing)
try:
    from chat import chat as chat_with_ai
    CHAT_AVAILABLE = True
except ImportError:
    CHAT_AVAILABLE = False
    chat_with_ai = None

# Set page config and style
st.set_page_config(page_title="AI Medical Diagnosis", layout="wide")

# Custom CSS to make it look more like the React version
st.markdown("""
    <style>
    .stApp {
        background-color: #111827;
        color: white;
    }
    .stButton>button {
        width: 200px;
        margin: 5px;
        border-radius: 8px;
    }
    .filter-button {
        background-color: transparent;
        border: none;
        color: #9CA3AF;
        padding: 8px 16px;
        border-radius: 8px;
        cursor: pointer;
    }
    .filter-button:hover {
        color: white;
    }
    .filter-button.active {
        background-color: white;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 AI Medical Diagnosis System")

# Create tabs for different features
tab1, tab2 = st.tabs(["📷 Image Processing", "🩹 Symptoms Diagnosis"])

# Tab 1: Image Processing
with tab1:
    st.header("Image Processing & Object Detection")
    
    # Store the selected filter in session state
    if 'selected_filter' not in st.session_state:
        st.session_state.selected_filter = "Original"

    # Create a container for the main content
    with st.container():
        # Center-align the content
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Camera and Capture buttons in one row
            col_cam1, col_cam2 = st.columns(2)
            with col_cam1:
                camera_btn = st.button("📷 Use Camera", use_container_width=True)
            with col_cam2:
                capture_btn = st.button("🖼️ Capture Image", use_container_width=True)

            # Filter selection
            st.markdown("### Select Filter")
            st.session_state.selected_filter = st.select_slider(
                "",
                options=["Original", "Grayscale", "Infrared", "Xray"]
            )

            # Image display area
            image_placeholder = st.empty()
            if not camera_btn and not capture_btn:
                st.markdown("### No image selected")
            
            # Process and Download buttons
            col_proc1, col_proc2 = st.columns(2)
            with col_proc1:
                process_btn = st.button("⚡ Process Image", use_container_width=True)
            with col_proc2:
                download_btn = st.button("⬇️ Download Image", use_container_width=True)

    # Handle camera capture
    if camera_btn:
        captured_image = capture_image()
        if captured_image:
            # Show original image
            image_placeholder.image(captured_image, caption="Captured Image", use_container_width=True)
            
            # Process image with selected filter when Process button is clicked
            if process_btn:
                processed_image = process_image(captured_image, st.session_state.selected_filter)
                image_placeholder.image(
                    processed_image, 
                    caption=f"Processed Image ({st.session_state.selected_filter})", 
                    use_container_width=True
                )

    # Handle image upload
    uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    # Store current image in session state and save path for chat
    current_image = None
    if uploaded_image:
        # Save uploaded image temporarily for chat function
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"uploaded_image_{st.session_state.get('form_submit_count', 0)}.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        st.session_state.uploaded_image_path = temp_path
        current_image = uploaded_image
        # Show original image
        image_placeholder.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        
        # Process image with selected filter when Process button is clicked
        if process_btn:
            processed_image = process_image(uploaded_image, st.session_state.selected_filter)
            image_placeholder.image(
                processed_image, 
                caption=f"Processed Image ({st.session_state.selected_filter})", 
                use_container_width=True
            )
    
    # Medical Image Classification Section
    if uploaded_image:
        st.markdown("---")
        st.markdown("### 🔬 Medical Image Analysis")
        
        # Image type selection
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            [
                "General Classification",
                "Chest X-ray (Pneumonia/TB)",
                "Enhanced Chest Abnormalities",
                "Skin Lesion (Moles)",
                "Bone Fracture",
                "Enhanced Bone Abnormalities",
                "Brain MRI/CT Abnormalities",
                "Abdominal Imaging Abnormalities",
                "Retinal/Eye Abnormalities",
                "Mammography Abnormalities"
            ],
            key="analysis_type"
        )
        
        # Analysis buttons
        col_analyze1, col_analyze2, col_analyze3, col_analyze4 = st.columns(4)
        
        with col_analyze1:
            classify_btn = st.button("🔍 Classify", use_container_width=True)
        with col_analyze2:
            segment_btn = st.button("🎯 Segment", use_container_width=True)
        with col_analyze3:
            risk_btn = st.button("⚠️ Risk Score", use_container_width=True)
        with col_analyze4:
            normal_abnormal_btn = st.button("✅ Normal/Abnormal", use_container_width=True)
        
        # Perform analysis based on button clicks
        if classify_btn or segment_btn or risk_btn or normal_abnormal_btn:
            # Convert uploaded image to numpy array for processing
            img_bytes = uploaded_image.read()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get image type
            img_type = get_image_type(img_rgb)
            st.info(f"📸 Detected Image Type: {img_type}")
            
            # Classification
            if classify_btn:
                st.markdown("#### 🔍 Classification Results")
                with st.spinner("Analyzing image..."):
                    if analysis_type == "Chest X-ray (Pneumonia/TB)":
                        result = classify_chest_xray(img_rgb)
                    elif analysis_type == "Enhanced Chest Abnormalities":
                        result = detect_chest_abnormalities(img_rgb)
                    elif analysis_type == "Skin Lesion (Moles)":
                        result = classify_skin_lesion(img_rgb)
                    elif analysis_type == "Bone Fracture":
                        result = classify_bone_fracture(img_rgb)
                    elif analysis_type == "Enhanced Bone Abnormalities":
                        result = detect_bone_abnormalities(img_rgb)
                    elif analysis_type == "Brain MRI/CT Abnormalities":
                        result = detect_brain_abnormalities(img_rgb)
                    elif analysis_type == "Abdominal Imaging Abnormalities":
                        result = detect_abdominal_abnormalities(img_rgb)
                    elif analysis_type == "Retinal/Eye Abnormalities":
                        result = detect_retinal_abnormalities(img_rgb)
                    elif analysis_type == "Mammography Abnormalities":
                        result = detect_mammography_abnormalities(img_rgb)
                    else:
                        result = predict_diagnosis(img_rgb)
                        if isinstance(result, str):
                            result = {"diagnosis": result, "confidence": 0.0, "message": result}
                    
                    if isinstance(result, dict):
                        # Display diagnosis or abnormalities
                        if result.get("is_abnormal") is not None:
                            # New abnormality detection format
                            if result.get("is_abnormal"):
                                st.error(f"⚠️ **Abnormalities Detected**")
                                if result.get("abnormalities"):
                                    for abn in result.get("abnormalities", []):
                                        st.write(f"- **{abn.get('condition', 'Unknown')}** ({abn.get('confidence', 0)*100:.2f}% confidence, {abn.get('severity', 'Unknown')} severity)")
                                elif result.get("primary_finding"):
                                    pf = result.get("primary_finding", {})
                                    st.write(f"- **{pf.get('condition', 'Unknown')}** ({pf.get('confidence', 0)*100:.2f}% confidence)")
                            else:
                                st.success("✅ **No Abnormalities Detected** - Image appears normal")
                        elif result.get("diagnosis"):
                            # Traditional diagnosis format
                            if result.get("confidence", 0) > 0.7:
                                st.success(f"**Diagnosis:** {result['diagnosis']} ({result.get('confidence', 0)*100:.2f}% confidence)")
                            elif result.get("confidence", 0) > 0.5:
                                st.warning(f"**Diagnosis:** {result['diagnosis']} ({result.get('confidence', 0)*100:.2f}% confidence)")
                            else:
                                st.info(f"**Diagnosis:** {result['diagnosis']} ({result.get('confidence', 0)*100:.2f}% confidence)")
                        
                        # Display all predictions
                        if result.get("all_predictions"):
                            st.markdown("**Top Predictions:**")
                            for i, pred in enumerate(result.get("all_predictions", [])[:5], 1):
                                st.write(f"{i}. {pred.get('class', 'Unknown')}: {pred.get('confidence', 0)*100:.2f}%")
                        
                        # Display recommendation
                        if result.get("recommendation"):
                            st.markdown("**💡 Recommendation:**")
                            st.write(result["recommendation"])
            
            # Segmentation
            if segment_btn:
                st.markdown("#### 🎯 Problem Area Segmentation")
                with st.spinner("Segmenting problem areas..."):
                    segmented_img, seg_metadata = segment_problem_area(img_rgb)
                    
                    col_seg1, col_seg2 = st.columns(2)
                    with col_seg1:
                        st.image(img_rgb, caption="Original Image", use_container_width=True)
                    with col_seg2:
                        st.image(segmented_img, caption="Segmented Image (Problem Areas Highlighted)", use_container_width=True)
                    
                    if seg_metadata.get("has_problem_area"):
                        st.warning(f"⚠️ Problem area detected: {seg_metadata.get('problem_area_percentage', 0):.2f}% of image")
                        if seg_metadata.get("bounding_box"):
                            bbox = seg_metadata["bounding_box"]
                            st.write(f"📍 Bounding Box: X={bbox['x']}, Y={bbox['y']}, Width={bbox['width']}, Height={bbox['height']}")
                    else:
                        st.success("✅ No significant problem areas detected")
            
            # Risk Score
            if risk_btn:
                st.markdown("#### ⚠️ Risk Assessment")
                with st.spinner("Calculating risk score..."):
                    # Get diagnosis first
                    if analysis_type == "Chest X-ray (Pneumonia/TB)":
                        diag_result = classify_chest_xray(img_rgb)
                        diagnosis = diag_result.get("diagnosis")
                    elif analysis_type == "Enhanced Chest Abnormalities":
                        diag_result = detect_chest_abnormalities(img_rgb)
                        diagnosis = diag_result.get("primary_finding", {}).get("condition") if diag_result.get("is_abnormal") else "Normal"
                    elif analysis_type == "Skin Lesion (Moles)":
                        diag_result = classify_skin_lesion(img_rgb)
                        diagnosis = diag_result.get("diagnosis")
                    elif analysis_type == "Bone Fracture":
                        diag_result = classify_bone_fracture(img_rgb)
                        diagnosis = diag_result.get("diagnosis")
                    elif analysis_type == "Enhanced Bone Abnormalities":
                        diag_result = detect_bone_abnormalities(img_rgb)
                        diagnosis = diag_result.get("primary_finding", {}).get("condition") if diag_result.get("is_abnormal") else "Normal"
                    elif analysis_type == "Brain MRI/CT Abnormalities":
                        diag_result = detect_brain_abnormalities(img_rgb)
                        diagnosis = diag_result.get("primary_finding", {}).get("condition") if diag_result.get("is_abnormal") else "Normal"
                    elif analysis_type == "Abdominal Imaging Abnormalities":
                        diag_result = detect_abdominal_abnormalities(img_rgb)
                        diagnosis = diag_result.get("primary_finding", {}).get("condition") if diag_result.get("is_abnormal") else "Normal"
                    elif analysis_type == "Retinal/Eye Abnormalities":
                        diag_result = detect_retinal_abnormalities(img_rgb)
                        diagnosis = diag_result.get("primary_finding", {}).get("condition") if diag_result.get("is_abnormal") else "Normal"
                    elif analysis_type == "Mammography Abnormalities":
                        diag_result = detect_mammography_abnormalities(img_rgb)
                        diagnosis = diag_result.get("primary_finding", {}).get("condition") if diag_result.get("is_abnormal") else "Normal"
                    else:
                        diag_result = predict_diagnosis(img_rgb)
                        if isinstance(diag_result, dict):
                            diagnosis = diag_result.get("diagnosis")
                        else:
                            diagnosis = None
                    
                    risk_result = calculate_risk_score(img_rgb, diagnosis)
                    
                    risk_score = risk_result.get("risk_score", 0)
                    risk_level = risk_result.get("risk_level", "Unknown")
                    
                    # Display risk score with progress bar
                    st.markdown(f"**Risk Level:** {risk_level}")
                    st.progress(risk_score)
                    st.write(f"**Risk Score:** {risk_score*100:.1f}%")
                    
                    # Color code based on risk level
                    if risk_level == "Critical":
                        st.error(f"🚨 {risk_result.get('message', 'Critical risk detected')}")
                    elif risk_level == "High" or risk_level == "Very High":
                        st.warning(f"⚠️ {risk_result.get('message', 'High risk detected')}")
                    elif risk_level == "Moderate":
                        st.info(f"📋 {risk_result.get('message', 'Moderate risk detected')}")
                    else:
                        st.success(f"✅ {risk_result.get('message', 'Low risk detected')}")
            
            # Normal/Abnormal Classification
            if normal_abnormal_btn:
                st.markdown("#### ✅ Normal/Abnormal Classification")
                with st.spinner("Classifying..."):
                    result = classify_normal_abnormal(img_rgb)
                    
                    classification = result.get("classification", "Unknown")
                    confidence = result.get("confidence", 0)
                    is_normal = result.get("is_normal", False)
                    
                    if is_normal:
                        st.success(f"✅ **Classification: Normal** ({confidence*100:.2f}% confidence)")
                    else:
                        st.error(f"⚠️ **Classification: Abnormal** ({confidence*100:.2f}% confidence)")
                    
                    st.write(result.get("message", ""))

# Tab 2: Symptoms Diagnosis
with tab2:
    st.header("🩹 Symptoms Diagnosis Agent")
    st.markdown("Enter your symptoms below and our AI agent will analyze them to determine urgency, possible causes, and risks.")
    
    # Warning disclaimer
    st.warning("⚠️ **Important**: This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions you may have regarding a medical condition.")
    
    # Diagnosis mode selection
    if 'diagnosis_mode' not in st.session_state:
        st.session_state.diagnosis_mode = "ML-Based"
    
    if CHAT_AVAILABLE:
        diagnosis_mode = st.radio(
            "Select Diagnosis Mode:",
            ["ML-Based", "AI Chat Assistant"],
            horizontal=True,
            key="diagnosis_mode_selector"
        )
        st.session_state.diagnosis_mode = diagnosis_mode
    else:
        st.info("💡 AI Chat Assistant mode requires additional setup (OpenAI API key, vector store). Using ML-Based mode.")
        st.session_state.diagnosis_mode = "ML-Based"
    
    # Initialize session state for form refresh
    if 'form_submit_count' not in st.session_state:
        st.session_state.form_submit_count = 0
    if 'last_diagnosis_result' not in st.session_state:
        st.session_state.last_diagnosis_result = None
    if 'should_refresh' not in st.session_state:
        st.session_state.should_refresh = False
    
    # Display previous diagnosis result if available
    if st.session_state.last_diagnosis_result:
        result = st.session_state.last_diagnosis_result
        
        # Handle chat-based responses
        if result.get("mode") == "chat":
            if "error" not in result:
                st.markdown("### 💬 AI Assistant Response")
                st.markdown(result.get("chat_response", "No response generated"))
                if st.button("🔄 Enter New Symptoms", use_container_width=True):
                    st.session_state.last_diagnosis_result = None
                    st.session_state.form_submit_count += 1
                    st.rerun()
            else:
                st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
                if st.button("🔄 Try Again", use_container_width=True):
                    st.session_state.last_diagnosis_result = None
                    st.session_state.form_submit_count += 1
                    st.rerun()
        elif "error" not in result:
            # Display urgency level with color coding
            urgency_priority = result.get('urgency_priority', 'ROUTINE')
            urgency_level = result.get('urgency_level', '')
            
            if urgency_priority == 'CRITICAL':
                st.error(f"🚨 **URGENCY LEVEL: {urgency_level}**")
            elif urgency_priority == 'HIGH':
                st.warning(f"⚠️ **URGENCY LEVEL: {urgency_level}**")
            elif urgency_priority == 'MODERATE':
                st.info(f"📋 **URGENCY LEVEL: {urgency_level}**")
            else:
                st.success(f"✅ **URGENCY LEVEL: {urgency_level}**")
            
            # Display matched symptoms with explanations
            if result.get('matched_symptoms'):
                st.markdown("### 📊 Matched Symptoms")
                explanations = result.get('explanations', {})
                symptom_explanations = explanations.get('symptom_explanations', [])
                
                if symptom_explanations:
                    for symptom_exp in symptom_explanations:
                        st.write(f"• **{symptom_exp['term']}**")
                        with st.expander(f"💡 What does '{symptom_exp['term']}' mean? (Click to understand)"):
                            st.info(symptom_exp['explanation'])
                else:
                    st.write(", ".join([s.replace('_', ' ').title() for s in result['matched_symptoms']]))
            
            # Display possible causes with explanations
            st.markdown("### 🔍 Possible Causes")
            causes = result.get('possible_causes', [])
            explanations = result.get('explanations', {})
            cause_explanations = explanations.get('cause_explanations', [])
            
            if causes:
                for i, cause in enumerate(causes, 1):
                    # Find matching explanation
                    explanation = next((exp for exp in cause_explanations if exp['term'] == cause), None)
                    
                    if explanation:
                        st.write(f"{i}. **{cause}**")
                        with st.expander(f"💡 What does '{cause}' mean? (Click for simple explanation)", expanded=False):
                            st.success(f"**Simple Explanation:**\n\n{explanation['explanation']}")
                    else:
                        st.write(f"{i}. {cause}")
            else:
                st.write("Unable to determine specific causes.")
            
            # Display possible risks with explanations
            st.markdown("### ⚠️ Possible Risks")
            risks = result.get('possible_risks', [])
            risk_explanations = explanations.get('risk_explanations', [])
            
            if risks:
                for i, risk in enumerate(risks, 1):
                    # Find matching explanation
                    explanation = next((exp for exp in risk_explanations if exp['term'] == risk), None)
                    
                    if explanation:
                        st.write(f"{i}. **{risk}**")
                        with st.expander(f"💡 What does '{risk}' mean? (Click for simple explanation)", expanded=False):
                            st.warning(f"**Simple Explanation:**\n\n{explanation['explanation']}")
                    else:
                        st.write(f"{i}. {risk}")
            else:
                st.write("General health monitoring recommended.")
            
            # Display recommendations
            st.markdown("### 💡 Recommendations")
            recommendations = result.get('recommendations', [])
            if recommendations:
                for rec in recommendations:
                    st.write(f"• {rec}")
            
            # Add a button to clear results and enter new symptoms
            if st.button("🔄 Enter New Symptoms", use_container_width=True):
                st.session_state.last_diagnosis_result = None
                st.session_state.form_submit_count += 1
                st.rerun()
        else:
            st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
    
    # Input form with dynamic key to force refresh
    form_key = f"symptoms_form_{st.session_state.form_submit_count}"
    with st.form(form_key, clear_on_submit=True):
        symptoms_text = st.text_area(
            "Describe your symptoms:",
            placeholder="e.g., I have chest pain and shortness of breath, feeling dizzy...",
            height=150,
            key=f"symptoms_text_{st.session_state.form_submit_count}"
        )
        
        col_age, col_info = st.columns(2)
        with col_age:
            patient_age = st.number_input(
                "Age (optional):", 
                min_value=0, 
                max_value=120, 
                value=None, 
                step=1,
                key=f"patient_age_{st.session_state.form_submit_count}"
            )
        with col_info:
            additional_info = st.text_input(
                "Additional information (optional):", 
                placeholder="e.g., No known allergies",
                key=f"additional_info_{st.session_state.form_submit_count}"
            )
        
        submit_btn = st.form_submit_button("🔍 Analyze Symptoms", use_container_width=True)
    
    # Process diagnosis
    if submit_btn and symptoms_text:
        if st.session_state.diagnosis_mode == "AI Chat Assistant" and CHAT_AVAILABLE:
            # Use chat-based diagnosis
            with st.spinner("Consulting AI assistant..."):
                try:
                    # Check if there's an uploaded image in tab1
                    chat_image_path = None
                    if 'uploaded_image_path' in st.session_state:
                        chat_image_path = st.session_state.uploaded_image_path
                    
                    chat_response = chat_with_ai(symptoms_text, chat_image_path)
                    
                    # Format as result for display
                    result = {
                        "chat_response": chat_response,
                        "mode": "chat"
                    }
                except Exception as e:
                    result = {
                        "error": f"Chat error: {str(e)}",
                        "mode": "chat"
                    }
        else:
            # Use ML-based diagnosis
            with st.spinner("Analyzing symptoms..."):
                result = diagnose_symptoms(symptoms_text, patient_age, additional_info)
                result["mode"] = "ml"
        
        # Store result in session state
        st.session_state.last_diagnosis_result = result
        st.session_state.form_submit_count += 1
        
        # Auto-refresh to clear form and show results
        st.rerun()
    
    elif submit_btn and not symptoms_text:
        st.error("Please enter your symptoms to get a diagnosis.")
        # Still refresh to clear the form
        st.session_state.form_submit_count += 1
        st.rerun()
    
    # Example symptoms
    with st.expander("💡 Example Symptoms to Try"):
        st.markdown("""
        - "I have severe chest pain and shortness of breath"
        - "High fever with chills and body aches"
        - "Dizziness and nausea after eating"
        - "Persistent cough and fatigue"
        - "Abdominal pain with vomiting"
        """)

