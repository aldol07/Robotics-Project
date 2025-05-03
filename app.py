import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import io
import time


st.set_page_config(
    page_title="Human Detection with YOLOv5",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    model.eval()
    return model
def detect_humans(image, model, confidence=0.5):

    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
   
    results = model(img_array)
    
    people_detections = results.pandas().xyxy[0]
    people_detections = people_detections[people_detections['name'] == 'person']
    people_detections = people_detections[people_detections['confidence'] >= confidence]
    
    return people_detections, results.render()[0]

def draw_boxes(image, detections):
    img = np.array(image.copy())
    
    for _, detection in detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"Person: {conf:.2f}"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return Image.fromarray(img)

def process_webcam(model, confidence_threshold, stop_button):
    
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        st.error("Could not access webcam. Please check your camera connection.")
        return

    webcam_placeholder = st.empty()
    stats_placeholder = st.empty()
    details_placeholder = st.empty()

    frame_time = time.time()
    fps = 0
    fps_counter = 0
    fps_update_time = frame_time

    while not stop_button:
        ret, frame = cap.read()
        
        if not ret:
            st.error("Failed to capture frame from webcam.")
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        start_time = time.time()
        detections, rendered_img = detect_humans(frame_rgb, model, confidence_threshold)
        
        
        current_time = time.time()
        fps_counter += 1
        
        if current_time - fps_update_time >= 1.0:  
            fps = fps_counter / (current_time - fps_update_time)
            fps_counter = 0
            fps_update_time = current_time
        
        webcam_placeholder.image(rendered_img, caption="Live Webcam Feed",use_container_width=True)
        
        with stats_placeholder.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Humans Detected", len(detections))
            
            with col2:
                processing_time = time.time() - start_time
                st.metric("Processing Time", f"{processing_time:.3f} seconds")
            
            with col3:
                st.metric("FPS", f"{fps:.1f}")
        
        if not detections.empty:
            with details_placeholder.container():
                
                display_data = detections.copy()
                
                for col in ['xmin', 'ymin', 'xmax', 'ymax']:
                    display_data[col] = display_data[col].round(0).astype(int)
                
                display_data['confidence'] = (display_data['confidence'] * 100).round(1).astype(str) + '%'
                
                
                st.dataframe(display_data[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']], height=150)
    
    cap.release()

def main():
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #306998;
        text-align: center;
    }
    .stapp > header {
        background-color: transparent;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4B8BBE !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    
    st.markdown('<h1 class="main-header">Human Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Powered by YOLOv5</h2>', unsafe_allow_html=True)
    
   
    st.sidebar.title("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This application uses YOLOv5 to detect humans in images and webcam feed. "
        "Upload an image or use your webcam to identify and highlight people in real-time."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")
    st.sidebar.markdown("""
    - **Model**: YOLOv5s
    - **Framework**: PyTorch
    - **Classes**: Filtered for 'person'
    """)
    
    
    with st.spinner("Loading YOLOv5 model..."):
        model = load_model()
        st.success("Model loaded successfully!")
    
   
    tab1, tab2 = st.tabs(["📷 Webcam Detection", "🖼️ Image Upload"])
    
    with tab1:
        st.subheader("Real-time Human Detection")
        st.markdown("""
        This mode uses your webcam for real-time human detection. Click the start button below to begin.
        """)
        
        
        start_webcam = st.button("Start Webcam")
        
        if start_webcam:
            stop_webcam = st.button("Stop Webcam")
            
            process_webcam(model, confidence_threshold, stop_webcam)
            
    # Image Upload Tab
    with tab2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
           
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
           
            with st.spinner("Detecting humans..."):
                progress_bar = st.progress(0)
                
                
                for percent_complete in range(0, 101, 10):
                    time.sleep(0.1)
                    progress_bar.progress(percent_complete / 100)
                
              
                start_time = time.time()
                detections, rendered_img = detect_humans(image, model, confidence_threshold)
                end_time = time.time()
                
                progress_bar.progress(1.0)
            
            
            with col2:
                st.subheader("Detection Result")
                st.image(rendered_img, caption="Detection Result", use_column_width=True)
            
            
            st.markdown("---")
            st.subheader("Detection Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Humans Detected", len(detections))
            
            with col2:
                st.metric("Processing Time", f"{(end_time - start_time):.3f} seconds")
            
            with col3:
                avg_confidence = detections['confidence'].mean() if not detections.empty else 0
                st.metric("Average Confidence", f"{avg_confidence:.2f}")
            
           
            if not detections.empty:
                st.markdown("---")
                st.subheader("Detection Details")
                
               
                display_data = detections.copy()
               
                for col in ['xmin', 'ymin', 'xmax', 'ymax']:
                    display_data[col] = display_data[col].round(2)
               
                display_data['confidence'] = (display_data['confidence'] * 100).round(2).astype(str) + '%'
                
                
                st.dataframe(display_data[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])
            
            else:
                st.info("No humans detected in the image with the current confidence threshold.")

if __name__ == "__main__":
    main()