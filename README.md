# Face Recognition Project

This project is a simple face recognition system built with OpenCV, designed to train on a dataset of face images organized in subfolders. The model uses the Local Binary Patterns Histogram (LBPH) algorithm for face recognition.

## Project Structure
```
Face-Recognition-with-OpenCV/
│
├── faces/                           
│   ├── person1/                     
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── person2/
│   │   └── ...
│   └── ...                         
│
├── test_images/                     
│   └── test1.jpg
│
├── Train.py                         
├── Recognize.py                   
├── TakePhotos.py                  
├── id-names.csv                  
└── README.md                     
```

## Getting Started

### Prerequisites
- Python 3.x
- Libraries: `numpy`, `opencv-python`, `pandas`

### Installation
1. Clone this repository or download the project files.
2. Install required Python packages:
   ```bash
   pip install numpy opencv-python pandas
   ```

### Dataset
- The training dataset should be stored in the `faces/` folder.
- Each subfolder inside `faces/` should be named after the individual it represents and should contain images of that person.
- You can download a dataset from [Kaggle](https://www.kaggle.com/). After downloading, extract it and organize it into the `faces/` directory as described above.

## How to Use

### 1. Training the Model
Run `Train.py` to train the face recognition model:
```bash
python Train.py
```

This script processes images in the `faces/` folder, extracts facial features, and trains an LBPH face recognizer. The model and label information will be saved for later use.

### 2. Recognizing Faces
Run `Recognize.py` to recognize faces using a webcam or test images:
```bash
python Recognize.py
```

This script loads the trained model and detects faces in real-time or in test images, displaying the name of the recognized person.



## Tips for Improved Accuracy
- **Image Quality**: Use clear, high-resolution images for training.
- **Diverse Images**: Train with images from different angles and lighting conditions.
- **Preprocessing**: Convert images to grayscale and resize them for consistency.

## Acknowledgments
- This project uses OpenCV for face detection and recognition.
- Data for training was sourced from Kaggle's face recognition datasets.

