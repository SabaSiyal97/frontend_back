from flask import Flask, request, render_template, jsonify
import pickle
import json
import os

app = Flask(__name__)

# Load ML models
model_anxiety = pickle.load(open('model_anxiety.pkl', 'rb'))
model_depression = pickle.load(open('model_depression.pkl', 'rb'))
model_stress = pickle.load(open('model_stress.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/meditation')
def meditation():
    return render_template('meditation.html')


@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json['message'].lower()

    if 'hello' in user_message:
        return jsonify({'reply': 'Hi there! How are you feeling today?'})
    elif 'sad' in user_message:
        return jsonify({"reply": "I'm here for you. Want to talk about it?"})
    elif 'bye' in user_message:
        return jsonify({"reply": "Take care! I'm always here if you need support."})
    else:
        return jsonify({"reply": "Can you please tell me more about how you're feeling?"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['answers']
    result = {
        'anxiety': model_anxiety.predict([data])[0],
        'depression': model_depression.predict([data])[0],
        'stress': model_stress.predict([data])[0]
    }
    return jsonify(result)

# ✅ New journal saving route

@app.route('/save_journal', methods=['POST'])
def save_journal():
    journal_data = request.json.get('journal', [])

    if not journal_data:
        return jsonify({'message': '⚠️ No journal data received'}), 400

    try:
        os.makedirs("journals", exist_ok=True)

        with open('journals/journal_entries.txt', 'a', encoding='utf-8') as f:
            f.write(json.dumps(journal_data) + '\n')

        return jsonify({'message': '✅ Journal saved successfully!'})
    except Exception as e:
        print(f"Save error: {e}")
        return jsonify({'message': '❌ Failed to save journal entry.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
