import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import io
import time
import random


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

def compute_target_distance(robot_position, target_position):
    """Compute Euclidean distance between robot and target"""
    return np.sqrt((robot_position[0] - target_position[0])**2 + 
                   (robot_position[1] - target_position[1])**2)

def robot_decision_making(image, detections, robot_state):
    """Implement decision-making logic for the robot based on detected humans"""
    img = np.array(image.copy())
    height, width = img.shape[:2]
    
    # Robot parameters
    robot_radius = 20
    robot_position = robot_state.get('position', (width // 4, height // 2))
    robot_target = robot_state.get('target', None)
    robot_speed = robot_state.get('speed', 5)
    
    # Define approach and avoid criteria
    approach_criteria = robot_state.get('approach_criteria', 'tallest')  # Options: tallest, nearest, center
    avoid_threshold = robot_state.get('avoid_threshold', 0.7)  # Confidence threshold for avoidance
    
    # Extract person boxes
    person_boxes = []
    for _, detection in detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width_person = x2 - x1
        height_person = y2 - y1
        area = width_person * height_person
        distance = compute_target_distance(robot_position, (center_x, center_y))
        
        person_boxes.append({
            'bbox': (x1, y1, x2, y2),
            'center': (center_x, center_y),
            'confidence': conf,
            'area': area,
            'distance': distance,
            'height': height_person,
            'decision': 'neutral'  # Default decision
        })
    
    # Make decisions for each person
    targets_to_approach = []
    targets_to_avoid = []
    
    for person in person_boxes:
        # Determine if the person should be avoided based on confidence
        if person['confidence'] > avoid_threshold:
            person['decision'] = 'avoid'
            targets_to_avoid.append(person)
        else:
            person['decision'] = 'approach'
            targets_to_approach.append(person)
    
    # Determine the primary target to approach
    target_to_approach = None
    if targets_to_approach:
        if approach_criteria == 'tallest':
            target_to_approach = max(targets_to_approach, key=lambda x: x['height'])
        elif approach_criteria == 'nearest':
            target_to_approach = min(targets_to_approach, key=lambda x: x['distance'])
        elif approach_criteria == 'center':
            # Find the person closest to image center
            image_center = (width // 2, height // 2)
            for person in targets_to_approach:
                person['center_distance'] = compute_target_distance(image_center, person['center'])
            target_to_approach = min(targets_to_approach, key=lambda x: x['center_distance'])
    
    # Update robot target
    if target_to_approach:
        robot_target = target_to_approach['center']
        robot_state['target'] = robot_target
    
    # Move robot towards target (simple direct movement)
    if robot_target:
        dx = robot_target[0] - robot_position[0]
        dy = robot_target[1] - robot_position[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > robot_speed:
            # Normalize direction vector and multiply by speed
            dx = dx / distance * robot_speed
            dy = dy / distance * robot_speed
            
            # Check if the new position would collide with any person to avoid
            new_position = (int(robot_position[0] + dx), int(robot_position[1] + dy))
            collision = False
            
            for person in targets_to_avoid:
                x1, y1, x2, y2 = person['bbox']
                # Add padding around the person
                padding = robot_radius + 10
                if (new_position[0] >= (x1 - padding) and new_position[0] <= (x2 + padding) and
                    new_position[1] >= (y1 - padding) and new_position[1] <= (y2 + padding)):
                    collision = True
                    break
            
            if not collision:
                robot_position = new_position
                robot_state['position'] = robot_position
    
    # Draw all person boxes with decision colors
    for person in person_boxes:
        x1, y1, x2, y2 = person['bbox']
        conf = person['confidence']
        
        # Color based on decision
        if person['decision'] == 'approach':
            color = (0, 255, 0)  # Green for approach
            label = f"Approach: {conf:.2f}"
        else:
            color = (255, 0, 0)  # Red for avoid
            label = f"Avoid: {conf:.2f}"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw robot
    cv2.circle(img, robot_position, robot_radius, (0, 0, 255), -1)  # Robot body
    
    # Draw target line if target exists
    if robot_target:
        cv2.line(img, robot_position, robot_target, (255, 255, 0), 2)
        cv2.circle(img, robot_target, 5, (255, 255, 0), -1)
    
    # Add decision explanation text
    cv2.putText(img, f"Approach strategy: {approach_criteria}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display the number of people to approach/avoid
    cv2.putText(img, f"Approach: {len(targets_to_approach)}, Avoid: {len(targets_to_avoid)}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return Image.fromarray(img), robot_state

def draw_boxes(image, detections):
    img = np.array(image.copy())
    
    for _, detection in detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"Person: {conf:.2f}"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return Image.fromarray(img)

def process_webcam(model, confidence_threshold, stop_button, robot_state):
    
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        st.error("Could not access webcam. Please check your camera connection.")
        return

    webcam_placeholder = st.empty()
    stats_placeholder = st.empty()
    details_placeholder = st.empty()
    decision_placeholder = st.empty()

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
        
        # Apply robotics decision making
        robotics_img, robot_state = robot_decision_making(rendered_img, detections, robot_state)
        
        current_time = time.time()
        fps_counter += 1
        
        if current_time - fps_update_time >= 1.0:  
            fps = fps_counter / (current_time - fps_update_time)
            fps_counter = 0
            fps_update_time = current_time
        
        webcam_placeholder.image(robotics_img, caption="Live Webcam Feed with Robotics Decisions",
                              use_container_width=True)
        
        with stats_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Humans Detected", len(detections))
            
            with col2:
                processing_time = time.time() - start_time
                st.metric("Processing Time", f"{processing_time:.3f} seconds")
            
            with col3:
                st.metric("FPS", f"{fps:.1f}")
                
            with col4:
                st.metric("Robot Status", "Active" if robot_state['target'] else "Idle")
        
        # Robot decision details
        with decision_placeholder.container():
            st.subheader("Robot Decision Making")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Current Strategy:** {robot_state['approach_criteria']}")
                st.write(f"**Avoidance Threshold:** {robot_state['avoid_threshold']}")
                if robot_state['target']:
                    st.write(f"**Current Target Position:** ({robot_state['target'][0]}, {robot_state['target'][1]})")
                else:
                    st.write("**No target currently selected**")
            
            with col2:
                if len(detections) > 0:
                    approach_count = sum(1 for _, det in detections.iterrows() 
                                      if det['confidence'] <= robot_state['avoid_threshold'])
                    avoid_count = len(detections) - approach_count
                    st.write(f"**Humans to approach:** {approach_count}")
                    st.write(f"**Humans to avoid:** {avoid_count}")
                else:
                    st.write("**No humans detected**")
        
        if not detections.empty:
            with details_placeholder.container():
                
                display_data = detections.copy()
                
                # Add decision column
                display_data['decision'] = ['Avoid' if conf > robot_state['avoid_threshold'] else 'Approach' 
                                         for conf in display_data['confidence']]
                
                for col in ['xmin', 'ymin', 'xmax', 'ymax']:
                    display_data[col] = display_data[col].round(0).astype(int)
                
                display_data['confidence'] = (display_data['confidence'] * 100).round(1).astype(str) + '%'
                
                st.dataframe(display_data[['name', 'confidence', 'decision', 'xmin', 'ymin', 'xmax', 'ymax']], 
                          height=150)
    
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
    st.markdown('<h2 class="sub-header">Powered by YOLOv5 with Robotics Integration</h2>', unsafe_allow_html=True)
    
    # Sidebar settings
    st.sidebar.title("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Robotics decision-making settings
    st.sidebar.title("Robot Settings")
    approach_criteria = st.sidebar.selectbox(
        "Approach Strategy",
        ["tallest", "nearest", "center"],
        help="Determines which detected human the robot should approach"
    )
    
    avoid_threshold = st.sidebar.slider(
        "Avoidance Threshold", 
        0.5, 1.0, 0.7, 0.05,
        help="Confidence threshold above which the robot will avoid a person"
    )
    
    # Initialize robot state
    robot_state = {
        'position': None,  # Will be set based on image dimensions
        'target': None,
        'speed': 5,
        'approach_criteria': approach_criteria,
        'avoid_threshold': avoid_threshold
    }
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This application uses YOLOv5 to detect humans and implements robotic decision making. "
        "The system determines which people to approach and which to avoid."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")
    st.sidebar.markdown("""
    - **Model**: YOLOv5s
    - **Framework**: PyTorch
    - **Classes**: Filtered for 'person'
    - **Robot Logic**: Smart decision-making based on confidence levels
    """)
    
    # Load model
    with st.spinner("Loading YOLOv5 model..."):
        model = load_model()
        st.success("Model loaded successfully!")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📷 Webcam Detection", "🖼️ Image Upload"])
    
    with tab1:
        st.subheader("Real-time Human Detection with Robotics")
        st.markdown("""
        This mode uses your webcam for real-time human detection with robotic decision-making.
        The robot (red circle) will decide which humans to approach (green boxes) and which to avoid (red boxes)
        based on the confidence level and your selected strategy.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("Robot will approach humans with lower confidence scores")
        with col2:
            st.warning("Robot will avoid humans with higher confidence scores")
            
        # Start button
        start_webcam = st.button("Start Webcam")
        
        if start_webcam:
            # Initialize robot position for webcam
            if robot_state['position'] is None:
                # We'll set this later when we get the first frame
                robot_state['position'] = (100, 100)
                
            stop_webcam = st.button("Stop Webcam")
            
            process_webcam(model, confidence_threshold, stop_webcam, robot_state)
            
    # Image Upload Tab
    with tab2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Reset robot state for new image
            height, width = np.array(image).shape[:2]
            robot_state['position'] = (width // 4, height // 2)  # Start from left side
            robot_state['target'] = None
            
            # Original and detection columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process image
            with st.spinner("Detecting humans..."):
                progress_bar = st.progress(0)
                
                # Simulate progress
                for percent_complete in range(0, 101, 10):
                    time.sleep(0.1)
                    progress_bar.progress(percent_complete / 100)
                
                # Actual detection
                start_time = time.time()
                detections, rendered_img = detect_humans(image, model, confidence_threshold)
                
                # Apply robotics decision making
                robotics_img, robot_state = robot_decision_making(rendered_img, detections, robot_state)
                
                end_time = time.time()
                
                progress_bar.progress(1.0)
            
            # Display result
            with col2:
                st.subheader("Robotics Decision Result")
                st.image(robotics_img, caption="Robot Decision Making", use_container_width=True)
            
            # Statistics
            st.markdown("---")
            st.subheader("Detection Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Humans Detected", len(detections))
            
            with col2:
                st.metric("Processing Time", f"{(end_time - start_time):.3f} seconds")
            
            with col3:
                approach_count = sum(1 for _, det in detections.iterrows() 
                                 if det['confidence'] <= robot_state['avoid_threshold'])
                st.metric("Humans to Approach", approach_count)
            
            with col4:
                avoid_count = len(detections) - approach_count
                st.metric("Humans to Avoid", avoid_count)
            
            # Robot decision explanation
            st.markdown("---")
            st.subheader("Robot Decision Explanation")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"The robot uses the **{approach_criteria}** strategy to decide which human to approach")
                st.write("- **Tallest**: Prioritizes the tallest detected person")
                st.write("- **Nearest**: Prioritizes the closest person to the robot")
                st.write("- **Center**: Prioritizes the person closest to the center of the image")
            
            with col2:
                st.warning(f"Persons with confidence above {avoid_threshold} are marked as 'avoid'")
                st.write("The robot (red circle) will plan a path towards the target while avoiding high-confidence detections")
                if robot_state['target']:
                    st.write(f"Current target position: ({robot_state['target'][0]}, {robot_state['target'][1]})")
            
            # Detection details
            if not detections.empty:
                st.markdown("---")
                st.subheader("Detection Details")
                
                # Prepare data for display
                display_data = detections.copy()
                
                # Add decision column
                display_data['decision'] = ['Avoid' if conf > robot_state['avoid_threshold'] else 'Approach' 
                                         for conf in display_data['confidence']]
                
                # Format numbers
                for col in ['xmin', 'ymin', 'xmax', 'ymax']:
                    display_data[col] = display_data[col].round(2)
                
                display_data['confidence'] = (display_data['confidence'] * 100).round(2).astype(str) + '%'
                
                # Display data
                st.dataframe(display_data[['name', 'confidence', 'decision', 'xmin', 'ymin', 'xmax', 'ymax']])
            
            else:
                st.info("No humans detected in the image with the current confidence threshold.")

if __name__ == "__main__":
    main()