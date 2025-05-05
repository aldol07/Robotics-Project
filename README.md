# 🤖 Human Detection & Robot Decision-Making using YOLOv5

This project is a robotics application that uses YOLOv5 for real-time human detection and simulates a robot's decision-making behavior based on detected humans. It runs as a web app using **Streamlit**, allowing users to upload images and visualize how a robot would respond—whether to approach or avoid people based on various criteria.

---

## 🚀 Features

- 🧠 **YOLOv5 Integration**: Detect humans in an uploaded image using a pre-trained YOLOv5s model.
- 🎯 **Robot Simulation**: The robot "moves" toward selected human targets or avoids them based on detection confidence.
- 🛡️ **Decision Strategies**: 
  - Approach tallest, nearest, or center-most person.
  - Avoid individuals with high detection confidence.
- 🎨 **Visual Feedback**: Bounding boxes for approach (green) or avoid (red), robot position, and decision paths drawn on the image.

---

## 🧪 Tech Stack

- [Streamlit](https://streamlit.io/)
- [YOLOv5](https://github.com/ultralytics/yolov5) via `torch.hub`
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Pillow (PIL)](https://python-pillow.org/)

---

## 🧰 Setup Instructions

1. **Clone the repository**

    ```bash
    https://github.com/aldol07/Robotics-Project
    cd Robotics-Project
    ```

2. **Create a virtual environment**

    ```bash
    python -m venv env
    ```

3. **Activate the environment**

    - On Windows:
      ```bash
      .\env\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source env/bin/activate
      ```

4. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

---

## 📂 File Structure

├── app.py # Main Streamlit app file
├── requirements.txt # Python dependencies
├── env/ # Virtual environment folder
└── README.md # This file


---

## 🖼️ How to Use

1. Run the app:

    ```bash
    streamlit run app.py
    ```

2. Upload an image containing humans or start webcam

3. Watch the robot:
    - Analyze the detections.
    - Decide whom to approach or avoid.
    - Move accordingly (simulated on image).

---

## 📌 Notes

- The robot is simulated and does **not** control a physical device.
- All detection and decision-making is visualized on the image itself.
- The app uses the **`yolov5s`** variant of YOLOv5 for fast and lightweight inference.

---

---

## 📄 License

This project is for academic and learning purposes.

---

## 👨‍💻 Author

**Kartikay Dubey**  
*Third-year ECE Student, School of Engineering, JNU*

---

