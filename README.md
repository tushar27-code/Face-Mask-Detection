# Face Mask Detection System

## Project Overview

The Face Mask Detection System is a Python-based computer vision application that detects whether a person is wearing a face mask.

It uses a Convolutional Neural Network (CNN) trained with TensorFlow/Keras and OpenCV for face detection. The application provides multiple detection modes through an interactive Streamlit interface.

## Features

- **Image Detection:** Upload an image and detect faces with or without masks.
- **Video Detection:** Process uploaded videos and identify mask/no-mask faces.
- **Live Webcam Detection:** Perform real-time detection using a computer webcam.
- **IP Webcam Detection:** Connect an Android phone camera using an IP Webcam video URL.
- **CNN Model:** Predict mask/no-mask status using a trained deep-learning model.
- **Automatic Image Saving:** Save detected no-mask face crops.
- **Dashboard:** View detection statistics, charts, and detection history.
- **CSV Export:** Download detection history reports.

## Technologies Used

- Python
- TensorFlow and Keras
- OpenCV
- NumPy
- Pandas
- Streamlit
- Streamlit-WebRTC

## Model Training

The CNN model was trained using image datasets organized into training and validation directories.

Reported training results:

- Training accuracy: 97.85%
- Validation accuracy: 95.88%
- Input image size: 150 × 150 pixels
- Classification: With Mask / Without Mask

## Project Structure

```text
Face_Mask/
├── main.py
├── backend.py
├── train.py
├── history.py
├── mask.keras
├── face.xml
├── requirements.txt
├── README.md
├── .gitignore
├── train/
├── test/
├── data/
└── Fetched_img/
```

## Installation

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Open the project directory:

```bash
cd Face_Mask
```

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate it on Windows:

```bash
venv\Scripts\activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run main.py
```

The application will open in your browser.

## Model Training

To train the model again, run:

```bash
python train.py
```

## Important Notes

- Ensure `mask.keras` and `face.xml` are available in the project directory.
- Keep the webcam and Android IP Webcam device connected and accessible.
- Detection counts represent face detections across frames, not unique individuals.
- Do not upload private images, personal videos, or sensitive detection records.

## Future Improvements

- Improve detection accuracy under different lighting conditions.
- Add unique-person tracking.
- Improve webcam session management.
- Add more detailed analytics and reporting.

## Author

Developed as a B.Tech Computer Science and Engineering project.