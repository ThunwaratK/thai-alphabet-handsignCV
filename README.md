# Thai Alphabet Hand Sign Recognition

A computer vision project for recognizing Thai alphabet hand signs using deep learning and MediaPipe hand tracking.

## Project Overview

This project uses a trained CNN model (MobileNetV2-based) to recognize Thai alphabet hand signs in real-time through a webcam interface. The system detects hands using MediaPipe and classifies the hand signs into Thai alphabet characters.

## Features

- Real-time hand sign detection using MediaPipe
- Thai alphabet character recognition (15 classes: ก, ด, ต, น, บ, พ, ฟ, ม, ย, ร, ล, ว, ษ, ห, อ)
- Webcam interface for live predictions
- Pre-trained models included
- Jupyter notebook for training and evaluation

## Project Structure

```
thai-alphabet-handsignCV/
├── dataset/
│   ├── image/              # Training images dataset
│   ├── model/              # Trained models
│   │   ├── best_sign_model.h5
│   │   ├── final_sign_model.h5
│   │   └── evaluation/
│   │       └── classification_report.txt
│   └── unseen/             # Test dataset
│       └── DatasetKen/     # Unseen test images
├── VisionFinal_Handsign.ipynb  # Training notebook
├── webcam.py               # Real-time webcam application
└── requirements.txt        # Python dependencies
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time recognition)
- (Optional) CUDA-compatible GPU for faster training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ThunwaratK/thai-alphabet-handsignCV.git
   cd thai-alphabet-handsignCV
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - Windows (Command Prompt):
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

The project requires the following main libraries:
- TensorFlow >= 2.13.0
- OpenCV >= 4.8.0
- MediaPipe >= 0.10.0
- NumPy >= 1.24.0
- Pillow >= 10.0.0
- scikit-learn >= 1.3.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- pandas >= 2.0.0
- tqdm >= 4.65.0
- pyperclip >= 1.8.2 (optional, for clipboard functionality)
- gdown >= 4.7.0

## Usage

### Real-time Hand Sign Recognition

Run the webcam application to start recognizing hand signs in real-time:

```bash
python webcam.py
```

**Controls:**
- Press `q` to quit the application
- The application will display the predicted Thai character and confidence score
- Ensure your hand is clearly visible within the camera frame

### Training the Model

To train the model from scratch or retrain with your own dataset:

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook VisionFinal_Handsign.ipynb
   ```

2. Follow the cells in the notebook sequentially:
   - Data loading and preprocessing
   - Model architecture definition
   - Training with callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
   - Evaluation and metrics generation

### Model Files

The project includes pre-trained models in `dataset/model/`:
- `best_sign_model.h5` - Best performing model checkpoint
- `final_sign_model.h5` - Final trained model
- `evaluation/classification_report.txt` - Model performance metrics

## Thai Alphabet Classes

The model recognizes the following 15 Thai alphabet characters:
- ก (Ko Kai)
- ด (Do Dek)
- ต (To Tao)
- น (No Nu)
- บ (Bo Baimai)
- พ (Pho Phan)
- ฟ (Fo Fan)
- ม (Mo Ma)
- ย (Yo Yak)
- ร (Ro Ruea)
- ล (Lo Ling)
- ว (Wo Waen)
- ษ (So Rue Si)
- ห (Ho Hip)
- อ (O Ang)

## Dataset

The dataset used in this project is sourced from:
- **Thai Alphabet Hand Sign Dataset**: [Mendeley Data](https://data.mendeley.com/datasets/rknd3wbz42/1)

## References

This project is based on research in Thai sign language recognition:
- Wongkanya, R., et al. "Thai Alphabet Sign Language Recognition Using Deep Learning" *English Language and Linguistics* 16(5), 2022. [PDF](http://www.icicel.org/ell/contents/2022/5/el-16-05-10.pdf)

## License

This project is for educational purposes.
