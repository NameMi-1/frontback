from flask import Flask, request, jsonify, render_template, send_file
import pronouncing
import os
import json
import re
from difflib import SequenceMatcher
from Levenshtein import ratio
from gtts import gTTS
import tempfile
from rapidfuzz import fuzz
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from google.cloud import speech
from google.oauth2 import service_account
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import openai
import joblib



app = Flask(__name__)

# ----------------- UI Pages -----------------
@app.route('/')
def home(): return render_template('index.html')

@app.route('/user')
def user(): return render_template('user.html')

@app.route('/difficulty')
def difficulty(): return render_template('difficulty.html')

@app.route('/settings')
def settings(): return render_template('settings.html')

@app.route('/test')
def speech_test(): return render_template('test.html')

@app.route('/learn')
def learn(): return render_template('learn.html', current_page='learn')

@app.route('/lesson1')
def lesson1(): return render_template('lesson1.html')

@app.route('/lesson2')
def lesson2(): return render_template('lesson2.html')

@app.route('/lesson3')
def lesson3(): return render_template('lesson3.html')

@app.route('/lesson4')
def lesson4(): return render_template('lesson4.html')

@app.route('/lesson5')
def lesson5(): return render_template('lesson5.html')

@app.route('/admin')
def admin(): return render_template('admin.html')

@app.route('/forget-password')
def forget_password(): return render_template('forget-password.html')

# ----------------- Audio Generation -----------------
@app.route('/get_native_audio')
def get_native_audio():
    text = request.args.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    tts = gTTS(text=text, lang='en')
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(temp_path)
    return send_file(temp_path, mimetype='audio/mpeg')

# ----------------- Load Sentences -----------------
@app.route('/get-test-sentences', methods=['GET'])
def get_test_sentences():
    try:
        json_path = os.path.join(os.getcwd(), "static", "sentences.json")
        if not os.path.exists(json_path):
            return jsonify({"error": "sentences.json file not found!"}), 404
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

# ----------------- Pronunciation Analysis -----------------
@app.route('/analyze', methods=['POST'])
def analyze_pronunciation():
    try:
        data = request.json
        target_sentence = data.get('target_sentence', '').lower().strip()
        user_speech = data.get('user_speech', '').lower().strip()
        strictness = data.get('strictness', 'medium')

        def preprocess_sentence(sentence):
            return re.sub(r'[^\w\s]', '', sentence).strip()

        target_sentence = preprocess_sentence(target_sentence)
        user_speech = preprocess_sentence(user_speech)

        def sentence_to_phonemes(sentence):
            return [pronouncing.phones_for_word(word)[0] if pronouncing.phones_for_word(word) else "" for word in sentence.split()]

        target_phonemes = sentence_to_phonemes(target_sentence)
        user_phonemes = sentence_to_phonemes(user_speech)

        if not target_phonemes or not user_phonemes:
            return jsonify({"error": "Phoneme extraction failed. Check input sentences."}), 400

        def phoneme_similarity(p1, p2):
            key_phonemes = {"TH", "S", "NG", "STH"}
            penalty_factor = 1.5
            seq_score = SequenceMatcher(None, p1, p2).ratio()
            lev_score = ratio(p1, p2)
            base_score = (lev_score * 0.7) + (seq_score * 0.3)
            if p1 in key_phonemes and p1 != p2:
                return base_score / penalty_factor
            return base_score

        correct = sum(phoneme_similarity(t, u) for t, u in zip(target_phonemes, user_phonemes))
        total = len(target_phonemes)
        accuracy = (correct / total) * 100 if total > 0 else 0

        def check_key_phonemes(t_ph, u_ph):
            key_phonemes = {"TH", "S", "NG", "STH"}
            for p in key_phonemes:
                if p in t_ph and p not in u_ph:
                    return False
            return True

        if not check_key_phonemes(target_phonemes, user_phonemes):
            accuracy *= 0.7
        if accuracy > 95:
            accuracy *= 0.98
        if strictness == "high" and accuracy < 95:
            accuracy *= 0.85
        elif strictness == "very_high" and accuracy < 98:
            accuracy *= 0.75

        accuracy = max(0, min(accuracy, 100))
        similarity_score = fuzz.ratio(target_sentence, user_speech)

        def word_match_ratio(ref, hypo):
            ref_words = ref.split()
            hypo_words = hypo.split()
            matches = sum(1 for r, h in zip(ref_words, hypo_words) if r == h)
            return (matches / len(ref_words)) * 100 if ref_words else 0

        word_match = word_match_ratio(target_sentence, user_speech)

        # ✅ Accent error detection
        def flag_accent_mistakes(target_phonemes, user_phonemes):
            accent_issues = []
            common_confusions = {
                "TH": ["D", "T"],
                "V": ["B", "W", "F"],
                "L": ["R"],
                "R": ["L"],
                "S": ["SH"],
                "Z": ["S"],
                "NG": ["N"]
            }
            for t, u in zip(target_phonemes, user_phonemes):
                for key, confusions in common_confusions.items():
                    if key in t and any(conf in u for conf in confusions):
                        accent_issues.append(f"{key} → {u}")
            return accent_issues

        accent_mistakes = flag_accent_mistakes(target_phonemes, user_phonemes)

        return jsonify({
            "target_phonemes": target_phonemes,
            "user_phonemes": user_phonemes,
            "transcription": user_speech,
            "similarity": round(similarity_score, 2),
            "word_match": round(word_match, 2),
            "accuracy": round(accuracy, 2),
            "strictness": strictness,
            "accent_issues": accent_mistakes
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    # === ตั้งค่าโมเดลสำเนียง ===
score_model = joblib.load("score_classifier_model.pkl")

openai.api_key = "YOUR_OPENAI_API_KEY"
GCP_CREDENTIALS = "PATH_TO_YOUR_GOOGLE_JSON_CREDENTIAL"  # ใส่ path ของคุณ

REFERENCE_AUDIO_FILES = {
    "male": {
        "alloy": "male_reference_alloy.wav",
        "echo": "male_reference_echo.wav",
        "onyx": "male_reference_onyx.wav",
        "fable": "male_reference_fable.wav"
    },
    "female": {
        "nova": "female_reference_nova.wav",
        "alloy": "female_reference_alloy.wav",
        "shimmer": "female_reference_shimmer.wav"
    }
}


# === ฟังก์ชันจาก realtime.py ===
def preprocess_audio(input_file, output_file, target_sr=16000):
    y, sr = librosa.load(input_file, sr=target_sr, mono=True)
    sf.write(output_file, y, target_sr, format='WAV')
    return output_file

def remove_noise(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
    denoised_path = "denoised_" + os.path.basename(audio_path)
    sf.write(denoised_path, y_denoised, sr)
    return denoised_path

def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
    return mfcc

def generate_speech_if_not_exists(text, voices):
    reference_mfccs = {}
    for voice, file_path in voices.items():
        if not os.path.exists(file_path):
            print(f"🔹 Creating reference voice {file_path}...")
            response = openai.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )
            with open(file_path, "wb") as f:
                f.write(response.content)
        reference_mfccs[voice] = extract_mfcc(file_path)
    return reference_mfccs

def speech_to_text(file_path, expected_text):
    credentials = service_account.Credentials.from_service_account_file(GCP_CREDENTIALS)
    client = speech.SpeechClient(credentials=credentials)

    with open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"
    )

    response = client.recognize(config=config, audio=audio)

    if not response.results:
        return False, None

    transcript = response.results[0].alternatives[0].transcript
    confidence = response.results[0].alternatives[0].confidence
    similarity_score = fuzz.ratio(expected_text.lower(), transcript.lower())

    return similarity_score >= 85, transcript

def compute_similarity(distance, num_frames):
    base = 15000
    max_distance = base + (num_frames * 500)
    min_distance = 0
    normalized_distance = max(min_distance, min(distance, max_distance))
    similarity = 100 * (1 - (normalized_distance - min_distance) / (max_distance - min_distance))
    return round(similarity, 2)

def compare_accent_with_reference(user_mfcc, reference_mfcc):
    distance, _ = fastdtw(user_mfcc, reference_mfcc, dist=lambda x, y: euclidean(x, y))
    return compute_similarity(distance, len(user_mfcc))

def compare_with_multiple_references(user_mfcc, reference_mfccs):
    best_similarity = 0
    best_voice = None
    for voice, ref_mfcc in reference_mfccs.items():
        similarity = compare_accent_with_reference(user_mfcc, ref_mfcc)
        if similarity > best_similarity:
            best_similarity = similarity
            best_voice = voice
    return best_voice, best_similarity

def predict_score_from_similarity(similarity_percent):
    input_feature = np.array([[similarity_percent]])
    score = score_model.predict(input_feature)[0]
    return score


# === Route ประเมินสำเนียง ===
@app.route('/evaluate_accent', methods=['POST'])
def evaluate_accent():
    try:
        audio = request.files['audio']
        gender = request.form.get('gender', 'male')
        expected_text = request.form.get('text', 'good morning')

        filename = "uploaded_user.wav"
        audio.save(filename)

        processed = preprocess_audio(filename, "processed_input.wav")
        denoised = remove_noise("processed_input.wav")

        is_correct, transcript = speech_to_text(denoised, expected_text)
        if not is_correct:
            return jsonify({'success': False, 'message': 'Speech mismatch.'})

        reference_mfccs = generate_speech_if_not_exists(expected_text, REFERENCE_AUDIO_FILES[gender])
        user_mfcc = extract_mfcc(denoised)
        best_voice, similarity = compare_with_multiple_references(user_mfcc, reference_mfccs)
        predicted_score = predict_score_from_similarity(similarity)

        return jsonify({
            'success': True,
            'recognized_text': transcript,
            'similarity': similarity,
            'predicted_score': predicted_score
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
 

# ----------------- Run -----------------
if __name__ == '__main__':
    app.run(debug=True)
