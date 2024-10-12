from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Initialize the QA model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

app = Flask(__name__)

# Main route to serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

# API route for question answering
@app.route('/get_answer', methods=['POST'])
def get_answer():
    data = request.json
    question = data['question']
    context = data['context']

    # Get answer from the model
    result = qa_pipeline({'question': question, 'context': context})
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
