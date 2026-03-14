# Brightness Classifier

A deep learning application that classifies images based on their brightness levels using MobileNetV2. The project includes both a training pipeline and a FastAPI-based REST API with a modern web interface.

## Features

- **Image Classification**: Classifies images as "dark" or "very dark"
- **Smart Brightness Detection**: Images with brightness > 80 are pre-classified as "dark" without model inference
- **Real-time API**: FastAPI-based REST endpoint for image classification
- **Web UI**: Modern, responsive user interface with drag-and-drop support
- **Lightweight Model**: Uses MobileNetV2 for fast inference
- **GPU Support**: Automatic GPU detection and usage
- **Kaggle Deployment Ready**: Includes notebook for Kaggle deployment with public URL sharing via ngrok

## Project Structure

```
.
├── app.py                      # FastAPI application with classification endpoint
├── model.py                    # MobileNetV2 model definition and loading
├── dataset.py                  # Custom PyTorch Dataset class
├── train.py                    # Training script
├── test_api.py                 # API testing script
├── kaggle_deployment.ipynb     # Kaggle notebook for deployment
├── model.pth                   # Trained model weights
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── static/
│   └── index.html             # Web UI (LUMIAI interface)
└── dataset/
    ├── dark/                  # Dark images for training
    └── very_dark/             # Very dark images for training
```

## Installation

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Brightness\ clasifierV2
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the API Server

Start the FastAPI server:
```bash
python app.py
```

The API will be available at `http://localhost:8000`
- **Web UI**: http://localhost:8000/
- **Classification Endpoint**: POST http://localhost:8000/classify

### Training the Model

Prepare your dataset in the following structure:
```
dataset/
├── dark/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── very_dark/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

Then run the training script:
```bash
python train.py
```

**Training Configuration** (editable in `train.py`):
- Epochs: 10
- Batch Size: 16
- Learning Rate: 0.001
- Train/Val Split: 80/20

### Testing the API

Use the included test script:
```bash
python test_api.py
```

Or use curl:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/classify
```

## API Documentation

### POST `/classify`

Classify an image based on brightness.

**Request:**
- **Content-Type**: multipart/form-data
- **Parameters**:
  - `file` (required): Image file (JPG, PNG, WEBP)

**Response:**
```json
{
  "category": "dark",
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response Codes**:
- `200`: Success
- `400`: Invalid image file

### GET `/`

Returns the web UI (index.html)

## Model Architecture

The model uses **MobileNetV2** with the following configuration:

- **Base Architecture**: Pre-trained ImageNet weights
- **Input Size**: 224×224 pixels
- **Feature Extraction**: Frozen early layers + unfrozen last 2 layers
- **Classification Head**: Dropout (0.5) + Linear layer (2 classes)
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss

## Brightness Classification Logic

1. **Brightness Check**: Calculate average luminance of the image
   - If brightness > 80: Classify as "dark" (skip model inference)
   - If brightness ≤ 80: Use the trained model for classification

2. **Model Classification**: Returns either "dark" or "very_dark"

## Deployment on Kaggle

A Kaggle deployment notebook is included (`kaggle_deployment.ipynb`):

1. Upload your dataset and code to Kaggle
2. Install dependencies
3. Train the model (optional if using pre-trained weights)
4. Set up ngrok for public URL access
5. Launch the FastAPI server

The public URL will be accessible externally for API calls.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA support (optional, CPU fallback available)
- 500MB+ free disk space (for model weights)

## File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server with image classification endpoint |
| `model.py` | MobileNetV2 model initialization and loading |
| `dataset.py` | Custom PyTorch dataset for image loading |
| `train.py` | Training loop with validation and checkpointing |
| `test_api.py` | Test script to verify API functionality |
| `kaggle_deployment.ipynb` | Kaggle notebook for training and deployment |
| `static/index.html` | Modern web interface (LUMIAI) |

## Performance

- **Inference Time**: ~50-100ms per image (CPU), ~10-20ms (GPU)
- **Model Size**: ~13MB
- **Input Processing**: Automatic resize to 224×224 with ImageNet normalization

## Troubleshooting

**Model not found error:**
- Ensure `model.pth` exists in the project root
- If missing, run `python train.py` to train a new model

**Port already in use:**
- Change the port in `app.py`: `uvicorn.run(app, host="127.0.0.1", port=8001)`

**CUDA out of memory:**
- Reduce batch size in `train.py`
- Or use CPU by setting: `device = torch.device('cpu')`

## License

MIT License - Feel free to use this project for personal or commercial purposes.

## Author

Brightness Classifier Project
