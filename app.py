from flask import Flask, request, render_template, jsonify
import pickle
import json
from datetime import datetime
import os

app = Flask(__name__)

mood_history = []  # For demo; use DB in real app

@app.route('/get_mood_history', methods=['POST'])
def save_mood_history_entry():  # ✅ New unique name
    data = request.get_json()
    entry = {
        'mood': int(data['mood']),
        'note': data.get('note', ''),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    mood_history.append(entry)
    return jsonify({"message": "✅ Mood saved!"})

@app.route('/get_mood_history', methods=['GET'])
def get_mood_history():
    return jsonify(mood_history[::-1])  # Return newest first

MOOD_FILE = "data/mood_entries.json"

@app.route('/assessment')
def assessment():
    return render_template('assessment.html')


@app.route('/quranic_support')
def quranic_support():
    return render_template('quranic_support.html')

@app.route('/save_mood', methods=['POST'])
def save_mood():
    data = request.get_json()
    mood = data.get('mood')
    note = data.get('note', '')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    entry = {"mood": mood, "note": note, "timestamp": timestamp}

    file_path = "data/mood_entries.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        # Load existing entries
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                entries = json.load(f)
        else:
            entries = []

        # Add the new entry to the beginning
        entries.insert(0, entry)

        # Keep only the 3 most recent
        entries = entries[:3]

        # Save back
        with open(file_path, 'w') as f:
            json.dump(entries, f, indent=2)

        return jsonify({"message": "✅ Mood entry saved."})
    except Exception as e:
        return jsonify({"message": "❌ Failed to save mood entry.", "error": str(e)}), 500

@app.route('/get_mood_entries')
def get_mood_entries():
    file_path = "data/mood_entries.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            entries = json.load(f)
        return jsonify(entries)
    else:
        return jsonify([])

# Load ML models
model_anxiety = pickle.load(open('model/anxiety_model.pkl', 'rb'))
model_depression = pickle.load(open('model/depression_model.pkl', 'rb'))
model_stress = pickle.load(open('model/stress_model.pkl', 'rb'))

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

# Load trained models
stress_model = pickle.load(open('model/stress_model.pkl', 'rb'))
anxiety_model = pickle.load(open('model/anxiety_model.pkl', 'rb'))
depression_model = pickle.load(open('model/depression_model.pkl', 'rb'))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    answers = data.get("answers", [])

    if len(answers) != 10:
        return jsonify({"error": "Expected 10 answers"}), 400

    prediction_input = [answers]  # Must be 2D

    stress = stress_model.predict(prediction_input)[0]
    anxiety = anxiety_model.predict(prediction_input)[0]
    depression = depression_model.predict(prediction_input)[0]

    return jsonify({
        "stress": int(stress),
        "anxiety": int(anxiety),
        "depression": int(depression)
    })

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
