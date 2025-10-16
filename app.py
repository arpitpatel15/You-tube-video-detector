import json, re, math, os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import warnings
from flask import Flask, render_template, request, jsonify
import joblib
from datetime import datetime
import unicodedata

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import only necessary components from sklearn/scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from scipy.stats import entropy

# Install textstat if needed: pip install textstat
try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
except ImportError:
    # Fallback functions if textstat is not installed
    def flesch_reading_ease(text):
        return 0
    def flesch_kincaid_grade(text):
        return 0

# ========================= USER CONFIG (Copied from original) =========================
# NOTE: Only use parameters needed for prediction/feature engineering
MODEL_PATH = "enhanced_fake_video_detector.joblib"
REAL_THRESHOLD = 0.5
ENABLE_ADVANCED_FEATURES = True
# Text feature sizes must match the ones used in training the saved model
TEXT_MAX_WORD_FEATURES = 20000
TEXT_MAX_CHAR_FEATURES = 12000
RANDOM_STATE = 42
# ======================================================================================

# Enhanced negative marker seeds with more comprehensive patterns
NEG_WORDS_SEED = [
    # English fake/clickbait indicators
    "fake", "hoax", "clickbait", "waste", "report", "scam", "buffering", "not working",
    "time waste", "boring", "stupid", "useless", "misleading", "lie", "lies", "fraud",
    "exposed", "truth revealed", "shocking", "unbelievable", "gone wrong", "prank",
    "you won't believe", "doctors hate", "secret revealed", "leaked", "scandal",

    # Hindi/Hinglish fake indicators
    "bakwaas", "bekar", "jhooth", "time kharab", "report karo", "report kar do",
    "explain nahi", "explain nhi", "nahi mila", "nhi mila", "movie nahi", "movie nhi",
    "ullu", "bewakoof", "faltu", "ghatiya", "kharab", "dhokha", "jhootha",

    # Engagement manipulation patterns
    "like subscribe", "smash like", "hit the bell", "notification squad", "first comment",
    "pin this comment", "heart this", "make this viral", "share if you", "repost if",

    # Common spam/fake patterns
    "100% working", "guaranteed", "instant", "unlimited", "free money", "easy money",
    "work from home", "make money online", "get rich quick", "secret method"
]
NEG_WORDS_SEED = list(dict.fromkeys([w.lower() for w in NEG_WORDS_SEED]))

# Positive indicators for real content
POS_WORDS_SEED = [
    "tutorial", "guide", "how to", "step by step", "explanation", "analysis", "review",
    "educational", "informative", "documentary", "news", "official", "verified",
    "research", "study", "facts", "data", "statistics", "evidence", "source",
    "interview", "expert", "professional", "academic", "scientific", "journal"
]
POS_WORDS_SEED = list(dict.fromkeys([w.lower() for w in POS_WORDS_SEED]))

# ---------- Enhanced Utility Functions (Copied from original) ----------

def clean_int(x):
    """Enhanced number cleaning with better handling of formatted numbers"""
    if x is None or x == '':
        return 0
    s = str(x).lower().replace(',', '').replace(' ', '').strip()

    multiplier = 1
    if 'k' in s:
        multiplier = 1000
        s = s.replace('k', '')
    elif 'm' in s:
        multiplier = 1000000
        s = s.replace('m', '')
    elif 'b' in s:
        multiplier = 1000000000
        s = s.replace('b', '')

    digits = re.findall(r'\d+\.?\d*', s)
    if digits:
        try:
            return int(float(digits[0]) * multiplier)
        except:
            return 0
    return 0

def normalize_text(text: str) -> str:
    """Enhanced text normalization"""
    if not text:
        return ""

    try:
        text = unicodedata.normalize('NFKD', text)
    except:
        pass

    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def extract_advanced_text_features(text: str) -> Dict[str, float]:
    """Extract advanced linguistic features from text"""
    # NOTE: Simplified for deployment to avoid unnecessary textstat errors
    if not text or len(text.strip()) < 10:
        return {
            'caps_ratio': 0, 'exclamation_ratio': 0, 'question_ratio': 0,
            'reading_ease': 0, 'reading_grade': 0, 'word_diversity': 0,
            'avg_word_length': 0, 'sentence_count': 0, 'emoji_count': 0
        }

    caps_ratio = len([c for c in text if c.isupper()]) / len(text) if text else 0
    exclamation_ratio = text.count('!') / len(text) if text else 0
    question_ratio = text.count('?') / len(text) if text else 0

    try:
        reading_ease = flesch_reading_ease(text)
        reading_grade = flesch_kincaid_grade(text)
    except:
        reading_ease = reading_grade = 0

    words = text.lower().split()
    word_diversity = len(set(words)) / len(words) if words else 0
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    sentence_count = len(re.findall(r'[.!?]+', text))
    emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text))

    return {
        'caps_ratio': caps_ratio,
        'exclamation_ratio': exclamation_ratio,
        'question_ratio': question_ratio,
        'reading_ease': reading_ease,
        'reading_grade': reading_grade,
        'word_diversity': word_diversity,
        'avg_word_length': avg_word_length,
        'sentence_count': sentence_count,
        'emoji_count': emoji_count
    }

def extract_engagement_features(record: Dict[str, Any]) -> Dict[str, float]:
    """Extract advanced engagement and metadata features"""
    likes = clean_int(record.get("likes", 0))
    views = clean_int(record.get("views", 0))
    subscribers = clean_int(record.get("subscribers", 0))
    duration = int(record.get("video_duration_seconds", 0))

    engagement_rate = (likes) / (views + 1)
    subscriber_view_ratio = subscribers / (views + 1)
    view_per_sub = views / (subscribers + 1)

    is_short_video = int(duration < 60)
    is_long_video = int(duration > 1800)
    duration_category = 0 if duration < 300 else 1 if duration < 1200 else 2
    channel_maturity = 0 if subscribers < 1000 else 1 if subscribers < 100000 else 2

    return {
        'engagement_rate': engagement_rate,
        'subscriber_view_ratio': subscriber_view_ratio,
        'view_per_sub': view_per_sub,
        'is_short_video': is_short_video,
        'is_long_video': is_long_video,
        'duration_category': duration_category,
        'channel_maturity': channel_maturity,
        'log_likes': np.log1p(likes),
        'log_views': np.log1p(views),
        'log_subscribers': np.log1p(subscribers),
        'log_duration': np.log1p(duration)
    }

def concat_texts(record: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Enhanced text concatenation with normalization"""
    title = normalize_text(record.get("title", "") or "")
    description = normalize_text(record.get("description", "") or "")

    comments = []
    # Simplified comment extraction for the web form, assumes comments are passed as a single string/list
    for k in ["top_comments", "top_replies"]:
        for c in (record.get(k) or []):
            if isinstance(c, dict):
                text = normalize_text(c.get("text", ""))
            else:
                text = normalize_text(str(c))
            if text:
                comments.append(text)

    comments_text = " \n ".join(comments)
    combined = " \n ".join([title, description, comments_text])

    return title, description, comments_text, combined

def has_pattern(s: str, pat: str) -> int:
    """Enhanced pattern matching with normalization"""
    if not s:
        return 0
    s = normalize_text(s)
    return int(bool(re.search(pat, s, flags=re.IGNORECASE)))

def count_occurrences(text: str, phrases: List[str]) -> int:
    """Enhanced phrase counting with better matching"""
    if not text:
        return 0
    text = normalize_text(text.lower())
    count = 0
    for phrase in phrases:
        pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
        count += len(re.findall(pattern, text))
    return count

def extract_spam_features(text: str) -> Dict[str, int]:
    """Extract spam indicators from text"""
    if not text:
        return {'repeated_chars': 0, 'excessive_caps': 0, 'spam_urls': 0, 'phone_numbers': 0}

    repeated_chars = len(re.findall(r'(.)\1{3,}', text))
    words = text.split()
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
    excessive_caps = int(caps_words > len(words) * 0.3)
    spam_urls = len(re.findall(r'bit\.ly|tinyurl|short\.link|free\.com', text, re.I))
    phone_numbers = len(re.findall(r'\b\d{10,}\b', text))

    return {
        'repeated_chars': repeated_chars,
        'excessive_caps': excessive_caps,
        'spam_urls': spam_urls,
        'phone_numbers': phone_numbers
    }

def build_enhanced_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build enhanced dataframe with advanced features"""
    rows = []

    for r in records:
        title, description, comments, combined = concat_texts(r)

        likes = clean_int(r.get("likes"))
        views = clean_int(r.get("views"))
        subscribers = clean_int(r.get("subscribers"))
        duration = int(r.get("video_duration_seconds") or 0)
        channel_id = r.get("channel_id", "") # Keep channel_id for grouping logic if needed

        total_comments = len((r.get("top_comments") or [])) + len((r.get("top_replies") or []))

        neg_count = count_occurrences(comments, NEG_WORDS_SEED)
        pos_count = count_occurrences(comments, POS_WORDS_SEED)

        title_features = extract_advanced_text_features(title)
        desc_features = extract_advanced_text_features(description)
        comment_features = extract_advanced_text_features(comments)
        engagement_features = extract_engagement_features(r)
        spam_features = extract_spam_features(combined)

        content_patterns = {
            'has_tutorial': has_pattern(title + " " + description, r'\b(tutorial|how\s*to|guide|step\s*by\s*step)\b'),
            'has_review': has_pattern(title + " " + description, r'\b(review|rating|opinion|thoughts)\b'),
            'has_news': has_pattern(title + " " + description, r'\b(news|breaking|latest|update|report)\b'),
            'has_entertainment': has_pattern(title + " " + description, r'\b(funny|comedy|meme|viral|trending)\b'),
            'has_educational': has_pattern(title + " " + description, r'\b(learn|education|explain|facts|science)\b'),
            'has_clickbait': has_pattern(title, r'\b(shocking|amazing|unbelievable|you\s*won\'?t\s*believe|secret|exposed)\b'),
            'has_full_movie': has_pattern(title, r'\b(full\s*movie|complete\s*film|entire\s*movie|फुल\s*मूवी)\b'),
            'has_fair_use': has_pattern(description, r'(fair\s*use|copyright\s*disclaimer|section\s*107|educational\s*purpose)'),
            'has_urgency': has_pattern(title, r'\b(urgent|hurry|limited\s*time|act\s*now|don\'?t\s*miss)\b')
        }

        row = {
            'channel_id': channel_id,
            'title': title,
            'description': description,
            'comments_text': comments,
            'combined_text': combined,
            'title_len': len(title),
            'desc_len': len(description),
            'comments_len': len(comments),
            'combined_len': len(combined),
            'likes': likes,
            'views': views,
            'subscribers': subscribers,
            'duration': duration,
            'total_comments': total_comments,
            'like_ratio': likes / (views + 1),
            'comment_ratio': total_comments / (views + 1),
            'neg_comment_count': neg_count,
            'pos_comment_count': pos_count,
            'neg_comment_ratio': neg_count / (total_comments + 1),
            'pos_comment_ratio': pos_count / (total_comments + 1),
            'sentiment_balance': (pos_count - neg_count) / (pos_count + neg_count + 1),
            'label': int(bool(r.get("is_real"))) # Placeholder for prediction, will be dropped
        }

        if ENABLE_ADVANCED_FEATURES:
            for prefix, features in [('title', title_features), ('desc', desc_features), ('comment', comment_features)]:
                for key, value in features.items():
                    row[f'{prefix}_{key}'] = value

            row.update(engagement_features)
            row.update(spam_features)
            row.update(content_patterns)

        rows.append(row)

    df = pd.DataFrame(rows)
    # Ensure all expected columns are present, fill missing with 0 for safety (important for a loaded model)
    # This step is often complex with vectorizers, but for prediction, we only need the numeric/text columns
    
    # Drop columns that are not features (identifiers, label)
    if 'video_id' in df.columns: df = df.drop(columns=['video_id'])
    if 'label' in df.columns: df = df.drop(columns=['label'])
    
    return df

# ---------- Prediction Function (Adapted for Flask) ----------

def predict_video_from_dict(pipe: Pipeline, new_video_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict on new video data using the loaded model pipeline.
    
    Args:
        pipe: The loaded scikit-learn Pipeline object.
        new_video_dict: Dictionary containing video metadata.
        
    Returns:
        Dictionary with prediction results and explanation.
    """
    try:
        # 1. Feature Engineering
        df = build_enhanced_dataframe([new_video_dict])

        # 2. Prediction
        prob_real = pipe.predict_proba(df)[0, 1]
        pred_label = int(prob_real >= REAL_THRESHOLD)
        
        # 3. Explanation Generation
        # enhanced_explain_prediction relies on the sample being passed *after* feature creation 
        # but the original logic requires a specific format which we must replicate
        sample_df_row = build_enhanced_dataframe([new_video_dict]).iloc[0] # Re-run for safety

        explanation_result = enhanced_explain_prediction(pipe, sample_df_row, REAL_THRESHOLD)
        
        return explanation_result

    except Exception as e:
        return {
            "verdict": "ERROR",
            "confidence_score": 0.0,
            "probability_real": 0.0,
            "reasoning": f"Prediction failed due to model or feature processing error: {str(e)}",
            "key_evidence": [],
            "raw_features": {}
        }

# ---------- Enhanced Explain Prediction (Copied from original) ----------

def enhanced_explain_prediction(pipe: Pipeline, sample_df_row: pd.Series,
                               real_threshold: float = REAL_THRESHOLD, k: int = 8) -> Dict[str, Any]:
    """Enhanced prediction explanation with better interpretability"""
    try:
        # Need to ensure the sample is a DataFrame for prediction
        sample_df = pd.DataFrame([sample_df_row])
        X_sample = sample_df.drop(columns=["label", "channel_id"], errors='ignore') # Drop ID columns too

        prob_real = pipe.predict_proba(X_sample)[0, 1]
        pred_label = int(prob_real >= real_threshold)
        confidence = prob_real if pred_label == 1 else 1 - prob_real

        # Extract key information for explanation
        title = sample_df_row.get('title', '')
        views = sample_df_row.get('views', 0)
        likes = sample_df_row.get('likes', 0)
        subscribers = sample_df_row.get('subscribers', 0)
        neg_comments = sample_df_row.get('neg_comment_count', 0)
        pos_comments = sample_df_row.get('pos_comment_count', 0)

        # Build explanation
        if pred_label == 1:
            verdict = "REAL"
            explanation = "This video appears to be authentic based on"
            if ENABLE_ADVANCED_FEATURES:
                factors = []
                if sample_df_row.get('has_educational', 0): factors.append("educational content patterns")
                if sample_df_row.get('has_tutorial', 0): factors.append("tutorial/instructional format")
                if sample_df_row.get('pos_comment_ratio', 0) > 0.1: factors.append("positive audience engagement")
                if sample_df_row.get('engagement_rate', 0) > 0.01: factors.append("healthy engagement metrics")
                if not factors: factors = ["content quality indicators", "engagement patterns"]
                explanation += " " + ", ".join(factors[:3])
        else:
            verdict = "FAKE/MISLEADING"
            explanation = "This video shows signs of being fake/misleading due to"
            if ENABLE_ADVANCED_FEATURES:
                factors = []
                if sample_df_row.get('has_clickbait', 0): factors.append("clickbait title patterns")
                if sample_df_row.get('has_full_movie', 0): factors.append("full movie claim (likely copyright violation)")
                if sample_df_row.get('neg_comment_ratio', 0) > 0.1: factors.append("negative audience feedback")
                if sample_df_row.get('spam_urls', 0) > 0: factors.append("spam links in content")
                if not factors: factors = ["suspicious content patterns", "engagement anomalies"]
                explanation += " " + ", ".join(factors[:3])

        # Key metrics for evidence
        engagement_rate = likes / (views + 1)
        subscriber_view_ratio = subscribers / (views + 1)
        
        evidence = [
            f"Confidence Score: {confidence:.3f}",
            f"Engagement Rate (Likes/Views): {engagement_rate:.4f}",
            f"Subscriber-View Ratio: {subscriber_view_ratio:.4f}"
        ]

        if neg_comments > 0 or pos_comments > 0:
            evidence.append(f"Comment Sentiment: {pos_comments} positive, {neg_comments} negative")

        return {
            "verdict": verdict,
            "confidence_score": round(confidence, 4),
            "probability_real": round(prob_real, 4),
            "reasoning": explanation,
            "key_evidence": evidence[:k],
            "raw_features": {
                "title": title[:50] + "...",
                "views": int(views),
                "likes": int(likes),
                "subscribers": int(subscribers),
                "engagement_rate": round(engagement_rate, 6)
            }
        }

    except Exception as e:
        return {
            "verdict": "ERROR",
            "confidence_score": 0.0,
            "probability_real": 0.0,
            "reasoning": f"Prediction explanation failed due to error: {str(e)}",
            "key_evidence": [],
            "raw_features": {}
        }


# ==============================================================================
# FLASK APPLICATION SETUP
# ==============================================================================

app = Flask(__name__)
# Global variable to hold the model pipeline
MODEL_PIPELINE = None

def load_model():
    """Load the trained model pipeline on startup."""
    global MODEL_PIPELINE
    if MODEL_PIPELINE is None:
        try:
            print(f"Attempting to load model from {MODEL_PATH}...")
            MODEL_PIPELINE = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
        except FileNotFoundError:
            # THIS IS CRITICAL FOR DEPLOYMENT: The file must exist!
            print(f"CRITICAL ERROR: Model file {MODEL_PATH} not found. Ensure it is in the root directory.")
            # Optionally, you can create a dummy pipeline if you absolutely can't train/save it locally first
            MODEL_PIPELINE = None
        except Exception as e:
            print(f"Error loading model: {e}")
            MODEL_PIPELINE = None

# Load the model once when the application starts
with app.app_context():
    load_model()


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if MODEL_PIPELINE is None:
            return render_template('index.html', error="Model not loaded. Please check deployment files.")

        try:
            # 1. Collect form data
            video_dict = {
                "title": request.form.get('title', ''),
                "description": request.form.get('description', ''),
                "likes": request.form.get('likes', 0),
                "views": request.form.get('views', 0),
                "subscribers": request.form.get('subscribers', 0),
                "video_duration_seconds": request.form.get('duration', 0),
                "channel_id": "DUMMY_CHANNEL",
                # Parse comments field - splitting by newline or semicolon
                "top_comments": re.split(r'[\n;]+', request.form.get('comments', '')),
                "top_replies": []
            }
            
            # 2. Convert numerical inputs
            for key in ["likes", "views", "subscribers", "video_duration_seconds"]:
                 video_dict[key] = clean_int(video_dict[key])


            # 3. Predict
            result = predict_video_from_dict(MODEL_PIPELINE, video_dict)
            
            # 4. Prepare result for display
            if result['verdict'] == 'REAL':
                result_class = 'alert-success'
            elif result['verdict'] == 'FAKE/MISLEADING':
                result_class = 'alert-danger'
            else:
                result_class = 'alert-warning'
            
            result['result_class'] = result_class

        except Exception as e:
            result = {'verdict': 'ERROR', 'reasoning': f"An unexpected error occurred: {str(e)}", 'result_class': 'alert-warning'}

    return render_template('index.html', result=result)

if __name__ == '__main__':
    # When running locally, it's best to train the model first.
    # We assume the user has run the original code's main function to create MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        print("\n" + "="*70)
        print(f"WARNING: Model file '{MODEL_PATH}' not found.")
        print("Please run the original ML script's main function to train and save the model before running the web app.")
        print("Starting Flask app anyway, but prediction will fail if model is not trained/saved.")
        print("="*70 + "\n")
        
    app.run(debug=True, port=5000)