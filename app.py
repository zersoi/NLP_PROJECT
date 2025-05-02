from flask import Flask, request, render_template_string
import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_DIR = "saved_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Poppins', sans-serif;
        }

        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 2rem;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 2rem;
            font-size: 2.5rem;
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        textarea {
            width: 100%;
            padding: 1rem;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            resize: vertical;
            min-height: 150px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }

        textarea:focus {
            outline: none;
            border-color: #3498db;
        }

        .submit-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 8px;
            font-size: 1.1rem;
            cursor: pointer;
            transition: transform 0.2s ease, background 0.3s ease;
            display: block;
            margin: 0 auto;
        }

        .submit-btn:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }

        .result-container {
            margin-top: 2rem;
            text-align: center;
            padding: 1.5rem;
            border-radius: 10px;
            animation: fadeIn 0.5s ease;
        }

        .positive {
            background: #e8f5e9;
            color: #2e7d32;
        }

        .negative {
            background: #ffebee;
            color: #c62828;
        }

        .result-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 768px) {
            .container {
                padding: 1.5rem;
                margin: 1rem;
            }
            
            h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1> Sentiment Analyzer For social media </h1>
        
        <form method="post">
            <div class="form-group">
                <textarea name="text" placeholder="Enter your here..."></textarea>
            </div>
            <button type="submit" class="submit-btn">Analyze Sentiment</button>
        </form>

        {% if result is not none %}
        <div class="result-container {{ 'positive' if result == 'Positive' else 'negative' }}">
            <i class="result-icon fas {{ 'fa-smile-beam' if result == 'Positive' else 'fa-frown' }}"></i>
            <h3>Prediction: {{ result }}</h3>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            result = "Positive" if pred == 1 else "Negative"
    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == '__main__':
    app.run(debug=True)