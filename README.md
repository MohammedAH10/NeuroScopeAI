# Brain MRI Analysis System

A Flask-based web application that analyzes brain MRI images for tumor classification and segmentation.

## Features

- **MRI Image Upload**: Upload brain MRI images for analysis
- **Tumor Classification**: Identifies the type of tumor (glioma, meningioma, pituitary) or confirms no tumor
- **Tumor Segmentation**: Visualizes the tumor area with an overlay if present
- **Medical Summary**: Provides a brief summary of the findings
- **Analysis History**: Stores all analyses for future reference

## Technical Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, Tailwind CSS
- **Database**: SQLite
- **Machine Learning**: TensorFlow/Keras
- **Models**: 
  - Brain MRI classification model (brain_mri.h5)
  - U-Net segmentation model (Unet_model.h5)

## Setup Instructions

1. **Clone the repository**

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Download the pre-trained models**
   - Place the models in the root directory:
     - `brain_mri.h5` (classification model)
     - `Unet_model.h5` (segmentation model)

4. **Initialize the database**
   - The database will be automatically created when you run the application for the first time

5. **Run the application**
   ```
   export OPENROUTER_API_KEY=your_openrouter_api_key
   python app.py
   ```

6. **Access the application**
   - Open a web browser and go to `http://127.0.0.1:5000/`

## Project Structure

```
├── app.py                  # Main Flask application file
├── brain_mri.db            # SQLite database (created automatically)
├── brain_mri.h5            # Classification model
├── Unet_model.h5           # Segmentation model
├── requirements.txt        # Dependencies
├── static/                 # Static files
│   ├── uploads/            # Uploaded MRI images
│   └── results/            # Generated results
└── templates/              # HTML templates
    ├── base.html           # Base template
    ├── index.html          # Homepage
    ├── result.html         # Results page
    └── history.html        # Analysis history page
```

## Notes

- This application is for educational purposes only and should not be used for actual medical diagnosis.
- The result summary uses OpenRouter through the OpenAI-compatible Python client and expects `OPENROUTER_API_KEY` to be set.
