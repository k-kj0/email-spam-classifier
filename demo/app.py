!pip install gradio scikit-learn pandas numpy matplotlib -q

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

print("🎯 Email Spam Classifier - Research Demo")
print("Based on: 'Email Spam Classification: A Comparative Analysis'")
print("=" * 60)

# === 1. EXTENDED TRAINING DATA ===
print("\n📚 Creating comprehensive dataset...")

spam_emails = [
    "Win free money now! Click here to claim your $1000 prize",
    "Congratulations! You've won an iPhone 15 Pro Max",
    "URGENT: Your account has been compromised. Verify now",
    "Earn $5000 weekly working from home. No experience needed",
    "Limited time offer: Get rich quick with our secret method",
    "Your payment is pending. Confirm your bank details immediately",
    "Exclusive offer: Free gift card for loyal customers",
    "Instant loan approval with 0% interest. Apply now",
    "You've been selected for a free cruise vacation",
    "Double your bitcoin investment in 7 days. Guaranteed",
    "Secret weight loss method doctors don't want you to know",
    "Your computer has viruses. Download our cleaner now",
    "Act now! This exclusive offer expires in 24 hours",
    "Miracle cure for all diseases. Discover the truth",
    "Nigerian prince needs your help to transfer $10 million"
]

ham_emails = [
    "Meeting scheduled for tomorrow at 10 AM in conference room",
    "Project report attached for your review and feedback",
    "Team lunch this Friday at 12:30 PM at the usual restaurant",
    "Please find the updated document attached for reference",
    "Reminder: Deadline for project submission is tomorrow EOD",
    "Can we reschedule our weekly call to next Tuesday?",
    "Thanks for sending the files. I've received them all",
    "Weekly status update from the marketing team attached",
    "Please review the proposal and share your thoughts by Friday",
    "HR announcement: Office will be closed next Monday for holiday",
    "Follow-up on our discussion regarding the quarterly budget",
    "Attached are the meeting minutes from yesterday's session",
    "Please approve the expenditure for the new equipment",
    "Reminder about the cybersecurity training session next week",
    "Your annual performance review is scheduled for next month"
]

all_emails = spam_emails + ham_emails
labels = [1]*len(spam_emails) + [0]*len(ham_emails)

print(f"✅ Dataset created: {len(all_emails)} emails")
print(f"   - Spam emails: {len(spam_emails)}")
print(f"   - Ham emails: {len(ham_emails)}")

# === 2. VECTORIZE TEXT ===
print("\n🔢 Vectorizing email text...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(all_emails)
print("✅ Text vectorization complete")

# === 3. TRAIN ALL 4 ALGORITHMS FROM PAPER ===
print("\n🤖 Training machine learning models...")

models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

for name, model in models.items():
    model.fit(X, labels)
    print(f"   ✓ {name} trained")

# === 4. CREATE ACCURACY CHART (FROM PAPER RESULTS) ===
def create_accuracy_chart():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data from your paper's Table II
    algorithms = ['Naive Bayes', 'SVM', 'Random Forest', 'Neural Network']
    accuracy = [94.2, 96.8, 98.3, 97.5]
    colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853']

    bars = ax.bar(algorithms, accuracy, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title('Email Spam Classifier - Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Machine Learning Algorithm', fontsize=12)
    ax.set_ylim(90, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, val in zip(bars, accuracy):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                f'{val}%', ha='center', va='bottom', fontweight='bold')

    # Add reference line for comparison
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='Baseline (95%)')
    ax.legend()

    plt.tight_layout()
    return fig

# === 5. CREATE PERFORMANCE TABLE (FROM PAPER) ===
def create_performance_table():
    # Data from Table II in your paper
    data = {
        'Algorithm': ['Naive Bayes', 'SVM', 'Random Forest', 'Neural Network'],
        'Accuracy': ['94.2%', '96.8%', '98.3%', '97.5%'],
        'Precision': ['93.0%', '95.0%', '97.0%', '96.0%'],
        'Recall': ['92.0%', '94.0%', '96.0%', '95.0%'],
        'F1-Score': ['92.5%', '94.5%', '96.5%', '95.5%']
    }

    df = pd.DataFrame(data)

    # Create HTML table with styling
    html = """
    <style>
    .performance-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-family: Arial, sans-serif;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .performance-table th {
        background-color: #2c3e50;
        color: white;
        padding: 12px;
        text-align: center;
        font-weight: bold;
    }
    .performance-table td {
        padding: 10px;
        text-align: center;
        border-bottom: 1px solid #ddd;
    }
    .performance-table tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    .performance-table tr:hover {
        background-color: #e8f4f8;
    }
    .best-metric {
        font-weight: bold;
        color: #27ae60;
    }
    </style>

    <table class="performance-table">
        <tr>
            <th>Algorithm</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
        </tr>
    """

    for _, row in df.iterrows():
        html += f"""
        <tr>
            <td><b>{row['Algorithm']}</b></td>
            <td class="best-metric">{row['Accuracy']}</td>
            <td>{row['Precision']}</td>
            <td>{row['Recall']}</td>
            <td>{row['F1-Score']}</td>
        </tr>
        """

    html += "</table>"
    return html

# === 6. MAIN CLASSIFICATION FUNCTION ===
def classify_email(email_text):
    """Classify email using all 4 algorithms"""

    if not email_text.strip():
        return "Please enter email text to classify", None, None

    # Vectorize input
    email_vector = vectorizer.transform([email_text])

    # Get predictions from all models
    results = []
    for name, model in models.items():
        pred = model.predict(email_vector)[0]
        proba = model.predict_proba(email_vector)[0]

        results.append({
            "Algorithm": name,
            "Prediction": "SPAM" if pred == 1 else "HAM",
            "Spam_Prob": f"{proba[1]*100:.1f}%",
            "Ham_Prob": f"{proba[0]*100:.1f}%",
            "Spam_Num": proba[1]*100,
            "Ham_Num": proba[0]*100
        })

    # Create results table
    html = """
    <style>
    .results-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-family: Arial, sans-serif;
    }
    .results-table th {
        background-color: #34495e;
        color: white;
        padding: 12px;
        text-align: left;
    }
    .results-table td {
        padding: 10px;
        border-bottom: 1px solid #ecf0f1;
    }
    .spam-row {
        background-color: #ffebee;
        color: #c62828;
    }
    .ham-row {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .probability-bar {
        display: inline-block;
        height: 20px;
        border-radius: 3px;
        margin-right: 5px;
    }
    </style>

    <table class="results-table">
        <tr>
            <th>Algorithm</th>
            <th>Prediction</th>
            <th>Spam Probability</th>
            <th>Ham Probability</th>
        </tr>
    """

    for result in results:
        row_class = "spam-row" if result["Prediction"] == "SPAM" else "ham-row"

        # Create probability bars
        spam_bar = f'<div class="probability-bar" style="width:{result["Spam_Num"]}px; background-color:#ef5350;"></div>'
        ham_bar = f'<div class="probability-bar" style="width:{result["Ham_Num"]}px; background-color:#66bb6a;"></div>'

        html += f"""
        <tr class="{row_class}">
            <td><b>{result['Algorithm']}</b></td>
            <td><strong>{result['Prediction']}</strong></td>
            <td>{spam_bar} {result['Spam_Prob']}</td>
            <td>{ham_bar} {result['Ham_Prob']}</td>
        </tr>
        """

    html += "</table>"

    # Calculate final verdict (majority vote)
    spam_count = sum(1 for r in results if r["Prediction"] == "SPAM")

    if spam_count == 4:
        verdict = "🚨 **HIGH CONFIDENCE SPAM** - All 4 algorithms agree this is spam"
        verdict_class = "high-spam"
        color = "#d32f2f"
    elif spam_count == 3:
        verdict = "⚠️ **LIKELY SPAM** - 3 out of 4 algorithms indicate spam"
        verdict_class = "likely-spam"
        color = "#f57c00"
    elif spam_count == 2:
        verdict = "🔍 **UNCLEAR / SUSPICIOUS** - Algorithms are split 50/50"
        verdict_class = "suspicious"
        color = "#ffb300"
    elif spam_count == 1:
        verdict = "📧 **PROBABLY HAM** - Only 1 algorithm detected spam"
        verdict_class = "likely-ham"
        color = "#388e3c"
    else:
        verdict = "✅ **CLEARLY LEGITIMATE** - All algorithms agree this is not spam"
        verdict_class = "clear-ham"
        color = "#2e7d32"

    verdict_html = f"""
    <div style="background-color: {color};
                padding: 20px;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin: 20px 0;
                font-size: 18px;
                font-weight: bold;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        {verdict}
        <br>
        <small style="font-weight: normal; opacity: 0.9;">
        Consensus: {spam_count}/4 algorithms detected spam
        </small>
    </div>
    """

    return html, verdict_html, create_accuracy_chart()

# === 7. BUILD GRADIO INTERFACE ===
print("\n🎨 Building professional interface...")

with gr.Blocks(title="Email Spam Classifier - Research Demo", theme=gr.themes.Soft()) as demo:

    # Header
    gr.Markdown("""
    # 📧 Email Spam Classifier
    ### A Comparative Analysis of Machine Learning Approaches
    *Research Paper Demo - IEEE Conference Format*

    **Author:** Kavya Jaiswal
    **Affiliation:** Department of Computer Science and Engineering, Amity University, Lucknow
    **Paper:** *Email Spam Classification: A Comparative Analysis*
    """)

    # Main layout
    with gr.Row():
        # Left column - Input and controls
        with gr.Column(scale=2):
            gr.Markdown("### ✍️ Email Input")

            email_input = gr.Textbox(
                label="Enter email content to classify:",
                placeholder="Paste email text here...",
                lines=6,
                elem_id="email-input"
            )

            # Example emails from research context
            examples = gr.Examples(
                examples=[
                    ["Congratulations! You've won $1,000,000! Click here to claim your prize now!"],
                    ["Meeting reminder: Project review tomorrow at 2 PM in Conference Room A"],
                    ["URGENT: Your bank account will be suspended unless you verify your identity immediately"],
                    ["Hi team, please find attached the quarterly financial report for review"],
                    ["You've been selected for a free vacation package! Reply NOW to claim"],
                    ["Please review the attached document and provide feedback by EOD Friday"]
                ],
                inputs=email_input,
                label="Research Test Cases:"
            )

            classify_btn = gr.Button(
                "🔍 Classify Email",
                variant="primary",
                size="lg",
                elem_id="classify-btn"
            )

            # Performance metrics from paper
            gr.Markdown("### 📊 Research Results Summary")
            performance_table = gr.HTML(create_performance_table())

        # Right column - Output and visualizations
        with gr.Column(scale=2):
            gr.Markdown("### 📈 Algorithm Performance")

            # Accuracy chart
            accuracy_plot = gr.Plot(
                label="Accuracy Comparison (From Paper)",
                value=create_accuracy_chart()
            )

            gr.Markdown("### 🔍 Classification Results")
            results_display = gr.HTML(label="Algorithm Predictions")

            gr.Markdown("### 🎯 Final Verdict")
            verdict_display = gr.HTML(label="Consensus Analysis")

    # Footer with research info
    gr.Markdown("---")
    gr.Markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
    <h3>Research Methodology</h3>
    <p>
    <strong>Algorithms Compared:</strong> Naive Bayes, SVM, Random Forest, Neural Network<br>
    <strong>Dataset:</strong> Synthetic email corpus (30 samples) for demonstration<br>
    <strong>Feature Extraction:</strong> TF-IDF vectorization with 1000 features<br>
    <strong>Evaluation:</strong> Accuracy, Precision, Recall, F1-Score metrics<br>
    <strong>Research Contribution:</strong> Comparative analysis demonstrating Random Forest superiority (98.3% accuracy)
    </p>
    <p><em>This demo simulates the research findings presented in the IEEE conference paper.</em></p>
    </div>
    """)

    # Connect button to function
    classify_btn.click(
        fn=classify_email,
        inputs=email_input,
        outputs=[results_display, verdict_display, accuracy_plot]
    )

# === 8. LAUNCH THE APP ===
print("\n🚀 Launching research demo application...")
print("=" * 60)
print("📢 IMPORTANT: When the link appears below, CLICK IT!")
print("⏳ Please wait 15-20 seconds for the app to start...")
print("=" * 60)

demo.launch(share=True, debug=False)
