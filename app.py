import os
import numpy as np
import cv2
import sqlite3
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from datetime import datetime
from markupsafe import escape

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
app.config['OPENROUTER_MODEL'] = 'openai/gpt-oss-120b'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)


# Class names for the classification model
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Database initialization
def init_db():
    conn = sqlite3.connect('brain_mri.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        original_path TEXT NOT NULL,
        result_path TEXT NOT NULL,
        classification TEXT NOT NULL,
        confidence REAL NOT NULL,
        summary TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

# Helper functions for model inference
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def iou(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    total = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
    union = total - intersection
    return (intersection + smooth) / (union + smooth)

# Load the models
classification_model = load_model('brain_mri.h5', compile=False)
segmentation_model = load_model('Unet_model.h5', custom_objects={
    'dice_coefficient': dice_coefficient,
    'dice_loss': dice_loss,
    'iou': iou
})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image_for_classification(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_image_for_segmentation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=(0, -1))


def format_summary_html(text):
    paragraphs = [segment.strip() for segment in text.split('\n') if segment.strip()]
    if not paragraphs:
        return "<p>No summary available.</p>"

    formatted = ''.join(f"<p>{escape(paragraph)}</p>" for paragraph in paragraphs)
    disclaimer = (
        "<p><strong>Note:</strong> This explanation is educational and must not be used "
        "as a medical diagnosis or treatment plan.</p>"
    )
    return formatted + disclaimer


def get_fallback_summary(classification, confidence):
    base_summaries = {
        'glioma': (
            "Glioma is a tumor arising from glial tissue in the brain. Clinical significance usually depends on tumor grade, location, and surrounding brain involvement. "
            "Typical follow-up in real care includes radiology review, symptom correlation, and specialist evaluation."
        ),
        'meningioma': (
            "Meningioma usually develops from the membranes surrounding the brain and is often slow-growing, though behavior varies by subtype and location. "
            "Real-world next steps often include reviewing size, pressure effect, growth pattern, and need for observation versus intervention."
        ),
        'pituitary': (
            "Pituitary tumors involve the pituitary region and may matter because of hormone effects, visual pathway compression, or local mass effect. "
            "Clinical workup commonly includes endocrine assessment and focused review of symptoms such as headache or visual disturbance."
        ),
        'no_tumor': (
            "No tumor class was detected by the model for this image. That does not rule out other abnormalities, image quality issues, or findings outside the model's scope. "
            "Formal interpretation should still depend on a qualified radiology or neurology workflow when clinically needed."
        )
    }

    if confidence > 0.9:
        confidence_note = "The model confidence is high for this predicted class."
    elif confidence > 0.7:
        confidence_note = "The model confidence is moderate for this predicted class."
    else:
        confidence_note = "The model confidence is limited, so the output should be treated cautiously."

    combined = f"{base_summaries.get(classification, 'The predicted class is not recognized by the summary helper.')} {confidence_note}"
    return format_summary_html(combined)


def get_openrouter_summary(classification, confidence):
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key or OpenAI is None:
        return get_fallback_summary(classification, confidence)

    client = OpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=api_key,
        default_headers={
            'HTTP-Referer': 'http://127.0.0.1:5000',
            'X-OpenRouter-Title': 'NeuroScope MRI'
        }
    )

    prompt = (
        f"Brain MRI model output:\n"
        f"- Predicted classification: {classification}\n"
        f"- Confidence: {confidence:.4f}\n\n"
        "Write a concise educational explanation for a web app result page. "
        "Explain what this tumor class generally means, why the confidence level matters, "
        "what clinical factors are usually reviewed next, and one caution about not using AI output alone. "
        "If the class is no_tumor, explain that no tumor was detected by the model but other abnormalities may still require professional review. "
        "Keep it to 3 short paragraphs in plain text. Do not claim a diagnosis. Do not mention that you are an AI model."
    )

    try:
        response = client.chat.completions.create(
            model=app.config['OPENROUTER_MODEL'],
            messages=[
                {
                    'role': 'system',
                    'content': (
                        "You write careful educational MRI result summaries for a student medical imaging app. "
                        "Be clinically literate, concise, and explicit that the output is not a diagnosis."
                    )
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            temperature=0.4,
            max_tokens=350
        )
        content = (response.choices[0].message.content or '').strip()
        if not content:
            return get_fallback_summary(classification, confidence)
        return format_summary_html(content)
    except Exception:
        return get_fallback_summary(classification, confidence)

def save_to_database(filename, original_path, result_path, classification, confidence, summary):
    conn = sqlite3.connect('brain_mri.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO analyses (filename, original_path, result_path, classification, confidence, summary)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (filename, original_path, result_path, classification, confidence, summary))
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return analysis_id

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'mri_image' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('analyze'))
    
    file = request.files['mri_image']
    
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('analyze'))
    
    if file and allowed_file(file.filename):
        # Save original image
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{timestamp}_{filename}"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(original_path)
        
        # Classify the image
        classification_input = preprocess_image_for_classification(original_path)
        predictions = classification_model.predict(classification_input)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Segment the image
        segmentation_input = preprocess_image_for_segmentation(original_path)
        segmentation_mask = segmentation_model.predict(segmentation_input)
        segmentation_mask = (segmentation_mask > 0.5).astype(np.uint8)
        
        # Create overlay image
        plt.figure(figsize=(10, 8))
        
        # Original image
        img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (128, 128))
        
        # Display original and mask overlay
        plt.subplot(1, 2, 1)
        plt.imshow(img_resized, cmap='gray')
        plt.title('Original MRI')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_resized, cmap='gray')
        plt.imshow(segmentation_mask[0, :, :, 0], alpha=0.5, cmap='jet')
        plt.title('Tumor Segmentation')
        plt.axis('off')
        
        # Save the result
        result_filename = f"result_{saved_filename.split('.')[0]}.png"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        plt.savefig(result_path, bbox_inches='tight')
        plt.close()
        
        # Get an educational insight summary from OpenRouter via an OpenAI-compatible API.
        summary = get_openrouter_summary(predicted_class, confidence)
        
        # Save to database
        analysis_id = save_to_database(
            saved_filename, 
            original_path, 
            result_path, 
            predicted_class,
            confidence,
            summary
        )
        
        # Redirect to results page
        return redirect(url_for('result', analysis_id=analysis_id))
    
    flash('Invalid file type. Please upload JPG, JPEG, or PNG files.', 'error')
    return redirect(url_for('analyze'))

@app.route('/result/<int:analysis_id>')
def result(analysis_id):
    conn = sqlite3.connect('brain_mri.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM analyses WHERE id = ?', (analysis_id,))
    analysis = cursor.fetchone()
    conn.close()
    
    if analysis:
        return render_template('result.html', analysis=analysis)
    else:
        flash('Analysis not found', 'error')
        return redirect(url_for('index'))

@app.route('/history')
def history():
    conn = sqlite3.connect('brain_mri.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM analyses ORDER BY created_at DESC LIMIT 20')
    analyses = cursor.fetchall()
    conn.close()
    
    return render_template('history.html', analyses=analyses)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
