from transformers import pipeline
from PIL import Image
import gradio as gr

# Load models
text_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
image_emotion = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

def analyze_sentiment(image, text):
    results = []
    if text:
        text_result = text_sentiment(text)[0]
        results.append(f"📝 Text Sentiment: {text_result['label']} (confidence: {round(text_result['score']*100, 1)}%)")
    if image is not None:
        image_result = image_emotion(image)[0]
        results.append(f"🖼️ Image Emotion: {image_result['label']} (confidence: {round(image_result['score']*100, 1)}%)")
    if not results:
        return "Please provide an image or text!"
    return "\n\n".join(results)

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=[
        gr.Image(type="pil", label="Upload a face image"),
        gr.Textbox(label="Enter some text", placeholder="I am feeling great today!")
    ],
    outputs=gr.Textbox(label="Sentiment Analysis Results"),
    title="🧠 Ashwitha's Multimodal Sentiment Analyzer",
    description="Upload an image + enter text → AI analyzes both emotions!"
)

demo.launch(share=True)
