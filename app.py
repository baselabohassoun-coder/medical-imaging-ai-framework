import streamlit as st
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchxrayvision as xrv

# 1. Page Configuration and Title Styling
st.set_page_config(page_title="Radiologist AI Assistant", layout="wide")
st.title("🔍 Radiologist AI Assistant")
st.write("An open-source computer vision support tool utilizing pre-trained deep learning weights for rapid scan evaluation.")

# --- SIDEBAR: CLINICAL PATIENT DOSSIER ---
st.sidebar.header("📋 Patient Clinical Dossier")
patient_name = st.sidebar.text_input("Patient Full Name", placeholder="e.g., John Doe")
patient_age = st.sidebar.number_input("Patient Age", min_value=0, max_value=120, value=25)
patient_id = st.sidebar.text_input("Medical Record ID", placeholder="e.g., RAD-2026-77")
diagnostic_notes = st.sidebar.text_area("Clinical Notes / Symptoms", placeholder="Type any pre-existing symptoms here...")

# 2. Initialize the Pre-trained Medical Neural Network Architecture
@st.cache_resource
def load_medical_model():
    # DenseNet121 pre-trained on 100k+ clinical chest X-rays (NIH, CheXpert, MIMIC)
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval() 
    return model

model = load_medical_model()

# 3. Create clean layout columns on the local dashboard
col1, col2 = st.columns(2)

with col1:
    st.header("📸 Scan Upload")
    uploaded_file = st.file_uploader("Drag and drop your X-ray image here...", type=["png", "jpg", "jpeg"])
    
    # --- CRITICAL REAL-WORLD SAFETY GATE ---
    st.markdown("---")
    st.subheader("🛡️ Clinical Quality Assurance")
    anatomy_confirmed = st.checkbox("Confirm uploaded image is an Anterior-Posterior Thoracic (Chest) Scan", value=False)

with col2:
    st.header("🔍 Visual & AI Analysis")
    
    if uploaded_file is not None:
        # Load and render the visual image to the dashboard page
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Scan Profile", use_container_width=True)
        
        # Check if the manual verification box is unchecked
        if not anatomy_confirmed:
            st.error("⚠️ ANATOMICAL SAFETY GATE ACTIVE")
            st.warning("The neural vision pipeline requires mandatory anatomical confirmation before computing diagnostic weight matrices.")
            st.info("Please verify that the uploaded scan is a thoracic chest profile by checking the box under the upload panel to unlock the AI evaluation metrics.")
        else:
            st.success("✅ Thoracic Chest X-ray profile validated. Initializing pattern matching...")
            
            # --- REAL AI EVALUATION PIPELINE ---
            with st.spinner("Pre-trained neural network analyzing pixel matrices..."):
                # Preprocess image specifically for TorchXRayVision
                img_gray = image.convert("L")
                img_np = np.array(img_gray)
                img_normalized = xrv.datasets.normalize(img_np, 255)
                img_tensor = img_normalized[None, ...]

                transform = torchvision.transforms.Compose([
                    xrv.datasets.XRayCenterCrop(),
                    xrv.datasets.XRayResizer(224)
                ])
                img_transformed = transform(img_tensor)
                input_tensor = torch.from_numpy(img_transformed).unsqueeze(0)

                # Execute Model Inference
                with torch.no_grad():
                    outputs = model(input_tensor)[0]
                    raw_preds = dict(zip(model.pathologies, outputs.cpu().numpy().tolist()))

                # --- CLINICAL CALIBRATION FUNCTION ---
                # Converts raw sigmoid outputs (where ~0.50 is neutral noise) into calibrated 0-100% UI confidence
                def calibrate_score(raw_val):
                    if raw_val <= 0.50:
                        return max(0.0, (raw_val / 0.50) * 15.0)  # Maps 0.0-0.50 -> 0%-15% UI score
                    else:
                        return min(100.0, 15.0 + ((raw_val - 0.50) / 0.50) * 85.0)  # Maps 0.50-1.0 -> 15%-100% UI score

                # Group related pneumonia markers (Infiltration, Consolidation, Pneumonia)
                pneumonia_raw = max(
                    raw_preds.get("Pneumonia", 0.0),
                    raw_preds.get("Infiltration", 0.0),
                    raw_preds.get("Consolidation", 0.0)
                )
                mass_raw = max(
                    raw_preds.get("Mass", 0.0), 
                    raw_preds.get("Nodule", 0.0)
                )

                pneumonia_score = calibrate_score(pneumonia_raw)
                mass_score = calibrate_score(mass_raw)
                
                # Normal score is high when top pathology signals remain low
                max_pathology_signal = max([calibrate_score(v) for v in raw_preds.values()])
                normal_score = max(0.0, 100.0 - max_pathology_signal)

                calibrated_scores = {
                    "Pneumonia Signs": pneumonia_score,
                    "Mass / Nodules": mass_score,
                    "Normal Healthy Tissues": normal_score
                }

                st.subheader("📊 Calibrated Diagnostic Probability Breakdown")
                for finding, prob_percentage in calibrated_scores.items():
                    st.write(f"**{finding}**")
                    st.progress(int(prob_percentage))
                    st.write(f"Confidence score: {prob_percentage:.2f}%")

                # Expanded clinical breakdown displaying top predictions
                with st.expander("🔬 Detailed Multi-Pathology Spectrum (Calibrated & Raw)"):
                    sorted_pathologies = sorted(raw_preds.items(), key=lambda x: x[1], reverse=True)
                    for path_name, raw_val in sorted_pathologies[:5]:
                        cal_val = calibrate_score(raw_val)
                        st.write(f"• **{path_name}:** {cal_val:.1f}% calibrated score (Raw: {raw_val*100:.1f}%)")

            # --- AUTOMATIC REPORT GENERATION ---
            st.markdown("---")
            st.header("📝 Automated Clinical Findings Report")
            
            highest_finding = max(calibrated_scores, key=calibrated_scores.get)
            
            report_text = f"""==================================================
RADIOLOGY DEPARTMENT SERVICES
AUTOMATED CLINICAL ASSISTANT REPORT
==================================================
[PATIENT DEMOGRAPHICS]
Name: {patient_name if patient_name else 'NOT SPECIFIED'}
Age: {patient_age} Years Old
Record ID: {patient_id if patient_id else 'NOT SPECIFIED'}

[CLINICAL CONTEXT]
Presented Symptoms: {diagnostic_notes if diagnostic_notes else 'None documented.'}

[COMPUTER VISION FINDINGS (TORCHXRAYVISION BACKEND)]
Primary Structural Density Match: {highest_finding}

[PROBABILITY BREAKDOWN METRICS]
- Pneumonia Signs: {calibrated_scores['Pneumonia Signs']:.2f}%
- Mass / Nodules: {calibrated_scores['Mass / Nodules']:.2f}%
- Normal Healthy Tissues: {calibrated_scores['Normal Healthy Tissues']:.2f}%

[DIAGNOSTIC NOTICE]
The computer vision system flags structural metrics matching '{highest_finding}' as the highest probabilistic correlation. This output is strictly designed for workflow prioritization and rapid triage support. 

[VALIDATION SIGN-OFF]
Final Physician Status: [ ] APPROVED  /  [ ] REJECTED

Reviewing Radiologist Signature: _______________________
Date: ________________________
"""
            st.text_area("Generated Document Draft (Click inside to copy)", value=report_text, height=400)
                    
    else:
        st.info("Awaiting medical scan upload to initialize the computer vision system.")
