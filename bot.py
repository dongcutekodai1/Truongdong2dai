# -*- coding: utf-8 -*-
import telebot
from telebot import types
import json
import time
import os
import hashlib
import logging
from telebot.types import Message
from collections import defaultdict
import random
import string
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from keep_alive import keep_alive
keep_alive()

# ================== CONFIG ==================
TOKEN = "8076967422:AAFswjKNP8A5QSIA0RwwJ1AoW9z5ezIfcmU"
ADMIN_ID = "6381480476"  # Replace with numeric chat ID if needed, e.g., "123456789"
DATA_FILE = "user_data.json"
KEYS_FILE = "keys.json"
GAME_DATA_FILE = "game_data.json"
HISTORY_FILE = "prediction_history.json"

# Set up logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "bot.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

bot = telebot.TeleBot(TOKEN)
BASE_DIR = os.path.join(os.path.dirname(__file__), "bot_data")
os.makedirs(BASE_DIR, exist_ok=True)
DATA_FILE = os.path.join(BASE_DIR, DATA_FILE)
KEYS_FILE = os.path.join(BASE_DIR, KEYS_FILE)
GAME_DATA_FILE = os.path.join(BASE_DIR, GAME_DATA_FILE)
HISTORY_FILE = os.path.join(BASE_DIR, HISTORY_FILE)

# ================== INITIALIZATION ==================
def initialize_files():
    """Initialize all required files"""
    default_files = {
        KEYS_FILE: {'thomas27031': float('inf'), '14112003': float('inf')},
        DATA_FILE: {},
        GAME_DATA_FILE: {},
        HISTORY_FILE: {"history": {}, "patterns": {}, "predictions": [], "factor_accuracy": {}}
    }
    
    for file_path, default_content in default_files.items():
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(default_content, f, indent=4, ensure_ascii=False)
            logger.info(f"Created {file_path}")

initialize_files()

# ================== DATA MANAGEMENT ==================
def load_data(file_path: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return {}

def save_data(data: dict, file_path: str) -> bool:
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {e}")
        return False

user_data = load_data(DATA_FILE)
game_data = load_data(GAME_DATA_FILE)
history_data = load_data(HISTORY_FILE)

# ================== CORE FUNCTIONS ==================
def is_valid_hash(hash_str: str, hash_type: str = "md5") -> bool:
    valid_lengths = {
        "md5": 32,
        "sha1": 40,
        "sha256": 64,
        "sha512": 128,
        "sha3_256": 64
    }
    if hash_type not in valid_lengths:
        return False
    return len(hash_str) == valid_lengths[hash_type] and all(c in "0123456789abcdef" for c in hash_str.lower())

def generate_hash(input_string: str, hash_type: str = "md5") -> str:
    hash_funcs = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512,
        "sha3_256": hashlib.sha3_256
    }
    if hash_type not in hash_funcs:
        return None
    return hash_funcs[hash_type](input_string.encode('utf-8')).hexdigest()

def is_prime(n: int) -> bool:
    if n < 2: return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0: return False
    return True

def is_fibonacci(n: int) -> bool:
    x, y = 0, 1
    while y < n:
        x, y = y, x + y
    return y == n

def is_key_valid(cid: str) -> bool:
    """Check if user has a valid, non-expired key"""
    if cid not in user_data or "key" not in user_data[cid] or "key_expiry" not in user_data[cid]:
        return False
    expiry = user_data[cid]["key_expiry"]
    if expiry == float('inf'):
        return True
    if time.time() > expiry:
        user_data[cid].pop("key", None)
        user_data[cid].pop("key_expiry", None)
        save_data(user_data, DATA_FILE)
        logger.info(f"Key for user {cid} has expired and was removed")
        return False
    return True

def clean_game_data():
    """Clean invalid entries in game_data"""
    cleaned_data = {}
    for key, result in game_data.items():
        try:
            hash_type, hash_str = key.split(":", 1)
            if is_valid_hash(hash_str, hash_type) and result in ["Tài", "Xỉu"]:
                cleaned_data[key] = result
        except:
            continue
    save_data(cleaned_data, GAME_DATA_FILE)
    return cleaned_data

# ================== PREDICTION MODELS ==================
class PredictionModel:
    def __init__(self):
        self.pattern_history = history_data.get("patterns", {})
        self.global_history = history_data.get("history", {})
        self.factor_accuracy = history_data.get("factor_accuracy", {})
        self.model = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier(random_state=42))
        ], voting='soft')
        self.is_trained = False
        self.load_and_train_model()
        
    def save_history(self):
        history_data["patterns"] = self.pattern_history
        history_data["history"] = self.global_history
        history_data["factor_accuracy"] = self.factor_accuracy
        save_data(history_data, HISTORY_FILE)
    
    def get_hash_pattern(self, hash_str: str, hash_type: str) -> str:
        """Extract pattern features from any hash"""
        if hash_type == "md5":
            return f"{hash_str[:4]}-{hash_str[8:12]}-{hash_str[-4:]}"
        elif hash_type == "sha1":
            return f"{hash_str[:5]}-{hash_str[10:15]}-{hash_str[-5:]}"
        elif hash_type in ["sha256", "sha3_256"]:
            return f"{hash_str[:6]}-{hash_str[16:22]}-{hash_str[-6:]}"
        elif hash_type == "sha512":
            return f"{hash_str[:8]}-{hash_str[32:40]}-{hash_str[-8:]}"
        return hash_str[:6]

    def basic_analysis(self, hash_str: str) -> str:
        """Basic odd/even analysis"""
        digits = [int(c, 16) for c in hash_str]
        even = sum(1 for d in digits if d % 2 == 0)
        return "Tài" if even > len(digits) / 2 else "Xỉu"
    
    def get_ngrams(self, hash_str: str, n: int = 2) -> list:
        """Extract n-grams from hash"""
        return [hash_str[i:i+n] for i in range(len(hash_str) - n + 1)]
    
    def hex_transition_matrix(self, hash_str: str) -> float:
        """Calculate transition entropy of hex digits."""
        transitions = defaultdict(int)
        total_transitions = 0
        for i in range(len(hash_str) - 1):
            pair = hash_str[i:i+2]
            transitions[pair] += 1
            total_transitions += 1
        
        # Calculate transition probabilities and entropy
        entropy = 0
        for count in transitions.values():
            prob = count / total_transitions if total_transitions > 0 else 0
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy

    def positional_entropy(self, hash_str: str) -> dict:
        """Calculate entropy for different segments of the hash."""
        length = len(hash_str)
        segments = {
            "first": hash_str[:4] if length >= 4 else hash_str,
            "middle": hash_str[length//4:length//2] if length >= 8 else "",
            "last": hash_str[-4:] if length >= 4 else hash_str
        }
        entropy_dict = {}
        for segment_name, segment in segments.items():
            if not segment:
                entropy_dict[segment_name] = 0
                continue
            freq = [segment.count(c) for c in "0123456789abcdef"]
            total = len(segment)
            entropy = 0
            for count in freq:
                prob = count / total if total > 0 and count > 0 else 0
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            entropy_dict[segment_name] = entropy
        return entropy_dict

    def hex_clustering(self, hash_str: str) -> float:
        """Calculate ratio of high (8-f) to low (0-7) hex digits."""
        high_digits = sum(1 for c in hash_str if c in "89abcdef")
        low_digits = sum(1 for c in hash_str if c in "01234567")
        total = len(hash_str)
        return high_digits / total if total > 0 else 0

    def statistical_analysis(self, hash_str: str, hash_type: str) -> dict:
        """Advanced statistical analysis for any hash."""
        digits = [int(c, 16) for c in hash_str]
        mean = sum(digits) / len(digits)
        variance = sum((x - mean) ** 2 for x in digits) / len(digits)
        
        # Hex-specific features
        transition_entropy = self.hex_transition_matrix(hash_str)
        positional_entropies = self.positional_entropy(hash_str)
        high_low_ratio = self.hex_clustering(hash_str)
        
        # MD5-specific features
        if hash_type == "md5":
            ngrams = self.get_ngrams(hash_str, 2)
            ngram_freq = {ng: ngrams.count(ng) for ng in set(ngrams)}
            ngram_entropy = -sum((freq/len(ngrams)) * math.fix2(freq/len(ngrams)) for freq in ngram_freq.values() if freq > 0)
            first_eight = sum(int(c, 16) for c in hash_str[:8])
            last_eight = sum(int(c, 16) for c in hash_str[-8:])
            binary = ''.join(format(int(c, 16), '04b') for c in hash_str)
            bit_ratio = sum(1 for b in binary if b == '1') / len(binary)
        else:
            ngram_entropy = first_eight = last_eight = bit_ratio = 0
        
        return {
            "mean": mean,
            "std_dev": math.sqrt(variance),
            "even_ratio": sum(1 for d in digits if d % 2 == 0) / len(digits),
            "prime_digits": sum(1 for d in digits if is_prime(d)),
            "fib_digits": sum(1 for d in digits if is_fibonacci(d)),
            "ngram_entropy": ngram_entropy,
            "first_eight_sum": first_eight,
            "last_eight_sum": last_eight,
            "bit_ratio": bit_ratio,
            "transition_entropy": transition_entropy,
            "first_entropy": positional_entropies.get("first", 0),
            "middle_entropy": positional_entropies.get("middle", 0),
            "last_entropy": positional_entropies.get("last", 0),
            "high_low_ratio": high_low_ratio,
            "hash_type": hash_type
        }
    
    def extract_features(self, hash_str: str, hash_type: str) -> np.array:
        """Extract features from hash for ML model."""
        stats = self.statistical_analysis(hash_str, hash_type)
        digits = [int(c, 16) for c in hash_str]
        hex_freq = [digits.count(i) / len(digits) for i in range(16)]
        features = [
            stats["mean"],
            stats["std_dev"],
            stats["even_ratio"],
            stats["prime_digits"],
            stats["fib_digits"],
            int(hash_str[-1], 16),
            stats["transition_entropy"],
            stats["first_entropy"],
            stats["middle_entropy"],
            stats["last_entropy"],
            stats["high_low_ratio"]
        ] + hex_freq
        if hash_type == "md5":
            features.extend([
                stats["ngram_entropy"],
                stats["first_eight_sum"],
                stats["last_eight_sum"],
                stats["bit_ratio"]
            ])
        else:
            features.extend([0, 0, 0, 0])  # Pad for other hash types
        return np.array(features)

    def load_and_train_model(self):
        """Load history and train ML model."""
        global game_data
        game_data = clean_game_data()  # Clean data before training
        if not game_data:
            return
        
        X, y = [], []
        for key, result in game_data.items():
            try:
                hash_type, hash_str = key.split(":", 1)
                X.append(self.extract_features(hash_str, hash_type))
                y.append(1 if result == "Tài" else 0)
            except:
                continue
        
        if len(X) < 20:  # Increased minimum data requirement
            logger.info(f"Not enough data to train model: {len(X)} samples")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameter tuning for RandomForest
        param_grid = {
            'rf__n_estimators': [50, 100],
            'rf__max_depth': [None, 10],
            'xgb__n_estimators': [50, 100],
            'xgb__max_depth': [3, 5]
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        self.is_trained = True
        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        logger.info(f"Model trained with accuracy: {accuracy:.2f}")

    def ml_prediction(self, hash_str: str, hash_type: str) -> tuple:
        """Predict using ML model."""
        if not self.is_trained:
            return "Tài", 50.0
        
        features = self.extract_features(hash_str, hash_type).reshape(1, -1)
        pred = self.model.predict(features)[0]
        prob = self.model.predict_proba(features)[0][pred] * 100
        return ("Tài" if pred == 1 else "Xỉu", prob)
    
    def enhanced_prediction(self, hash_str: str, hash_type: str) -> tuple:
        """Combined prediction with ML and hex features."""
        stats = self.statistical_analysis(hash_str, hash_type)
        pattern = self.get_hash_pattern(hash_str, hash_type)
        ml_pred, ml_conf = self.ml_prediction(hash_str, hash_type)
        
        # Dynamic weights based on factor accuracy
        default_weights = {
            "even_ratio": 0.2,
            "mean": 0.15,
            "prime_digits": 0.15,
            "fib_digits": 0.1,
            "last_digit": 0.1,
            "pattern": 0.3,
            "ml": 0.4,
            "transition_entropy": 0.15,  # Weight for transition entropy
            "first_entropy": 0.1,        # Weight for first segment entropy
            "high_low_ratio": 0.1        # Weight for high/low hex digit ratio
        }
        weights = self.factor_accuracy.get(hash_type, default_weights)
        
        # Decision factors
        factors = [
            (stats["even_ratio"] > 0.5, weights["even_ratio"]),
            (stats["mean"] > 7.5, weights["mean"]),
            (stats["prime_digits"] > 4, weights["prime_digits"]),
            (stats["fib_digits"] > 3, weights["fib_digits"]),
            (hash_str[-1] in "01234567", weights["last_digit"]),
            (stats["transition_entropy"] > 3.5, weights["transition_entropy"]),  # Threshold for transition entropy
            (stats["first_entropy"] > 2.0, weights["first_entropy"]),           # Threshold for first segment entropy
            (stats["high_low_ratio"] > 0.5, weights["high_low_ratio"])          # Threshold for high/low ratio
        ]
        
        # Pattern history
        pattern_key = f"{hash_type}:{pattern}"
        pattern_factor = 0
        if pattern_key in self.pattern_history:
            history = self.pattern_history[pattern_key]
            tai_count = history.count("Tài")
            xiu_count = len(history) - tai_count
            pattern_factor = weights["pattern"] if tai_count > xiu_count else -weights["pattern"]
        
        # Combine with ML prediction
        ml_factor = weights["ml"] if ml_pred == "Tài" else -weights["ml"]
        score = pattern_factor + ml_factor + sum(f * w for f, w in factors)
        final_pred = "Tài" if score >= 0 else "Xỉu"
        accuracy = ml_conf if self.is_trained else 75.0
        
        # Apply confidence threshold
        if accuracy < 60.0:
            return None, accuracy  # Indicate low confidence
        
        return final_pred, accuracy
    
    def update_model(self, hash_str: str, hash_type: str, actual_result: str):
        """Update model with actual results and track hex feature accuracy."""
        pattern = self.get_hash_pattern(hash_str, hash_type)
        pattern_key = f"{hash_type}:{pattern}"
        
        # Update pattern history
        if pattern_key not in self.pattern_history:
            self.pattern_history[pattern_key] = []
        self.pattern_history[pattern_key].append(actual_result)
        
        # Update global history
        self.global_history.setdefault(hash_type, []).append(actual_result)
        if len(self.global_history[hash_type]) > 1000:
            self.global_history[hash_type].pop(0)
        
        # Update factor accuracy
        stats = self.statistical_analysis(hash_str, hash_type)
        factors = {
            "even_ratio": stats["even_ratio"] > 0.5,
            "mean": stats["mean"] > 7.5,
            "prime_digits": stats["prime_digits"] > 4,
            "fib_digits": stats["fib_digits"] > 3,
            "last_digit": hash_str[-1] in "01234567",
            "transition_entropy": stats["transition_entropy"] > 3.5,
            "first_entropy": stats["first_entropy"] > 2.0,
            "high_low_ratio": stats["high_low_ratio"] > 0.5
        }
        for factor, value in factors.items():
            if factor not in self.factor_accuracy.setdefault(hash_type, {}):
                self.factor_accuracy[hash_type][factor] = {"correct": 0, "total": 0}
            self.factor_accuracy[hash_type][factor]["total"] += 1
            if (value and actual_result == "Tài") or (not value and actual_result == "Xỉu"):
                self.factor_accuracy[hash_type][factor]["correct"] += 1
        
        self.save_history()
        self.load_and_train_model()

    def log_prediction(self, hash_str: str, hash_type: str, prediction: str, confidence: float):
        """Log prediction for tracking."""
        history_data["predictions"].append({
            "hash": hash_str,
            "hash_type": hash_type,
            "prediction": prediction,
            "confidence": confidence,
            "timestamp": time.time()
        })
        save_data(history_data, HISTORY_FILE)

    def calculate_md5_accuracy(self):
        """Calculate accuracy for MD5 predictions."""
        md5_data = {k: v for k, v in game_data.items() if k.startswith("md5:")}
        if not md5_data:
            return 0
        correct = sum(1 for k, v in md5_data.items() if self.ml_prediction(k.split(":")[1], "md5")[0] == v)
        return correct / len(md5_data) * 100

prediction_model = PredictionModel()

# ================== KEY MANAGEMENT ==================
def generate_random_key(length: int = 12) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def check_key(key: str, chat_id: str) -> bool:
    keys = load_data(KEYS_FILE)
    if key in keys:
        expiry = time.time() + (keys[key] * 86400) if keys[key] != float('inf') else float('inf')
        
        for uid, data in user_data.items():
            if data.get("key") == key and uid != chat_id:
                return False
        
        user_data.setdefault(chat_id, {"username": "Unknown"})
        user_data[chat_id].update({
            "key": key,
            "key_expiry": expiry,
            "join_date": time.time(),
            "username": user_data[chat_id].get("username", "Unknown")
        })
        
        save_data(user_data, DATA_FILE)
        return True
    return False

# ================== BOT HANDLERS ==================
@bot.message_handler(commands=['start'])
def handle_start(msg: Message):
    cid = str(msg.chat.id)
    user_data.setdefault(cid, {"username": msg.from_user.username or "Unknown"})
    
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    buttons = [
        types.KeyboardButton("Phân Tích Hash"),
        types.KeyboardButton("Hướng Dẫn"),
        types.KeyboardButton("Bảng Giá"),
        types.KeyboardButton("Liên Hệ Admin")
    ]
    kb.add(*buttons)
    total_users = len(user_data)
    message = (
        f"⭐ CHÀO MỪNG ĐẾN VỚI BOT TÀI XỈU MD5 ⭐\n\n"
        f"🆔 ID của bạn: {cid}\n"
        f"──────\n"
        f"🎮 Chọn một tùy chọn bên dưới để bắt đầu!"
    )
    bot.send_message(cid, message, reply_markup=kb, parse_mode=None)
    if cid == ADMIN_ID:
        bot.send_message(
            cid,
            f"👑 BẢNG ĐIỀU KHIỂN ADMIN 👑\n"
            f"Tổng số người dùng: {total_users}\n"
            f"Dùng /generatekey, /listkeys, /revokey để quản lý key.\n"
            f"MD5 Accuracy: {prediction_model.calculate_md5_accuracy():.2f}%",
            parse_mode=None
        )
    logger.info(f"User {cid} started the bot")

@bot.message_handler(commands=['key'])
def handle_key(msg: Message):
    cid = str(msg.chat.id)
    try:
        key = msg.text.split()[1]
        if check_key(key, cid):
            expiry = user_data[cid]["key_expiry"]
            expiry_msg = "vĩnh viễn" if expiry == float('inf') else f"{(expiry - time.time()) / 86400:.1f} ngày"
            bot.send_message(cid, f"✅ **Key hợp lệ!**\nThời hạn: {expiry_msg}\nBạn có thể sử dụng bot!", parse_mode="Markdown")
        else:
            bot.send_message(cid, "❌ Key không hợp lệ hoặc đã được sử dụng!", parse_mode="Markdown")
    except IndexError:
        bot.send_message(cid, "⚠️ Vui lòng nhập: `/key <mã_key>`", parse_mode="Markdown")

@bot.message_handler(commands=['predict'])
def handle_predict(msg: Message):
    cid = str(msg.chat.id)
    if not is_key_valid(cid):
        bot.send_message(cid, "🔐 Vui lòng nhập key trước khi sử dụng!", parse_mode="Markdown")
        return
    
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    buttons = [
        types.KeyboardButton("MD5"),
        types.KeyboardButton("SHA1"),
        types.KeyboardButton("SHA256"),
        types.KeyboardButton("SHA512"),
        types.KeyboardButton("SHA3_256")
    ]
    kb.add(*buttons)
    bot.send_message(cid, "🔍 Chọn loại mã băm để phân tích:", reply_markup=kb)

@bot.message_handler(func=lambda m: m.text in ["MD5", "SHA1", "SHA256", "SHA512", "SHA3_256"])
def handle_hash_type(msg: Message):
    cid = str(msg.chat.id)
    if not is_key_valid(cid):
        bot.send_message(cid, "🔐 Vui lòng nhập key trước khi sử dụng!", parse_mode="Markdown")
        return
    
    hash_type = msg.text.lower()
    user_data[cid]["selected_hash_type"] = hash_type
    save_data(user_data, DATA_FILE)
    bot.send_message(cid, f"🔍 Vui lòng gửi mã {msg.text} cần phân tích...")

@bot.message_handler(func=lambda m: is_valid_hash(m.text.strip(), user_data.get(str(m.chat.id), {}).get("selected_hash_type", "md5")))
def handle_hash(msg: Message):
    cid = str(msg.chat.id)
    if not is_key_valid(cid):
        bot.send_message(cid, "🔐 Vui lòng nhập key trước khi sử dụng!", parse_mode="Markdown")
        return
    
    hash_str = msg.text.strip().lower()
    hash_type = user_data.get(cid, {}).get("selected_hash_type", "md5")
    
    bot.send_chat_action(cid, 'typing')
    
    try:
        # Run predictions
        basic_pred = prediction_model.basic_analysis(hash_str)
        enhanced_pred, accuracy = prediction_model.enhanced_prediction(hash_str, hash_type)
        
        if enhanced_pred is None:
            bot.send_message(cid, f"⚠️ Độ tin cậy thấp ({accuracy:.1f}%). Vui lòng chờ thêm dữ liệu!", parse_mode="Markdown")
            return
        
        stats = prediction_model.statistical_analysis(hash_str, hash_type)
        
        # Prepare response
        response = (
            f"🔮 **Kết Quả Phân Tích**\n"
            f"📌 {hash_type.upper()}: `{hash_str}`\n\n"
            f"📊 Thống kê:\n"
            f"- Tỉ lệ chẵn: {stats['even_ratio']*100:.1f}%\n"
            f"- Số nguyên tố: {stats['prime_digits']}\n"
            f"- Số Fibonacci: {stats['fib_digits']}\n"
            f"- Entropy chuyển tiếp hex: {stats['transition_entropy']:.2f}\n"
            f"- Entropy 4 ký tự đầu: {stats['first_entropy']:.2f}\n"
            f"- Tỉ lệ hex cao/thấp: {stats['high_low_ratio']*100:.1f}%\n"
        )
        if hash_type == "md5":
            response += (
                f"- Entropy N-gram: {stats['ngram_entropy']:.2f}\n"
                f"- Tổng 8 ký tự đầu: {stats['first_eight_sum']}\n"
                f"- Tổng 8 ký tự cuối: {stats['last_eight_sum']}\n"
                f"- Tỉ lệ bit 1: {stats['bit_ratio']*100:.1f}%\n"
            )
        response += (
            f"\n🎯 Dự đoán:\n"
            f"- Cơ bản: {basic_pred}\n"
            f"- Nâng cao: {enhanced_pred} (Độ tin cậy: {accuracy:.1f}%)\n\n"
            f"📌 Kết luận: **{enhanced_pred}**\n"
            f"📈 Độ chính xác: {accuracy:.1f}%\n\n"
            f"📌 Gửi phản hồi bằng lệnh: `/feedback {hash_str} {hash_type} Tài|Xỉu`"
        )
        
        bot.send_message(cid, response, parse_mode="Markdown")
        prediction_model.log_prediction(hash_str, hash_type, enhanced_pred, accuracy)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        bot.send_message(cid, "⚠️ Có lỗi xảy ra khi phân tích!", parse_mode="Markdown")

@bot.message_handler(commands=['feedback'])
def handle_feedback(msg: Message):
    cid = str(msg.chat.id)
    if not is_key_valid(cid):
        bot.send_message(cid, "🔐 Vui lòng nhập key trước khi sử dụng!", parse_mode="Markdown")
        return
    
    try:
        _, hash_str, hash_type, result = msg.text.split()
        hash_type = hash_type.lower()
        if not is_valid_hash(hash_str, hash_type) or result not in ["Tài", "Xỉu"]:
            raise ValueError
        
        game_data[f"{hash_type}:{hash_str}"] = result
        save_data(game_data, GAME_DATA_FILE)
        
        prediction_model.update_model(hash_str, hash_type, result)
        
        bot.send_message(cid, "✅ Cảm ơn phản hồi của bạn! Hệ thống đã cập nhật.", parse_mode="Markdown")
        
    except Exception as e:
        bot.send_message(cid, "⚠️ Định dạng: `/feedback <hash> <hash_type> <Tài|Xỉu>`", parse_mode="Markdown")

@bot.message_handler(func=lambda m: m.text.strip() == "Phân Tích Hash")
def handle_analyze_button(msg: Message):
    handle_predict(msg)

@bot.message_handler(func=lambda m: m.text.strip() == "Hướng Dẫn")
def handle_guide(msg: Message):
    logger.info(f"User {msg.chat.id} requested guide")
    guide = (
        "📖 **Hướng Dẫn Sử Dụng**\n\n"
        "1. Nhập key kích hoạt bằng lệnh `/key <mã_key>`\n"
        "2. Chọn loại mã băm (MD5, SHA1, v.v.)\n"
        "3. Gửi mã hash để phân tích\n"
        "4. Xem kết quả dự đoán Tài/Xỉu\n"
        "5. Gửi phản hồi sau khi có kết quả thực tế\n\n"
        "📌 Mỗi key có thời hạn sử dụng riêng\n"
        "📊 Bot sẽ học từ phản hồi để cải thiện độ chính xác"
    )
    bot.send_message(msg.chat.id, guide, parse_mode="Markdown")
    logger.info(f"Guide sent to user {msg.chat.id}")

@bot.message_handler(func=lambda m: m.text.strip() == "Bảng Giá")
def handle_pricing(msg: Message):
    cid = str(msg.chat.id)
    logger.info(f"User {cid} requested pricing")
    pricing = (
        f"💰 **Bảng Giá Dịch Vụ**\n\n"
        f"┌──────────────┬──────────────┐\n"
        f"│ Thời Hạn     │ Giá          │\n"
        f"├──────────────┼──────────────┤\n"
        f"│ 1 Ngày       │ 20,000đ      │\n"
        f"│ 1 Tháng      │ 150,000đ     │\n"
        f"│ 3 Tháng      │ 400,000đ     │\n"
        f"│ Code bot     │ 200,000đ     │\n"
        f"└──────────────┴──────────────┘\n\n"
        f"💳 Liên hệ @Truongdong1920 để mua key\n"
        f"📈 Độ chính xác MD5: {prediction_model.calculate_md5_accuracy():.1f}%"
    )
    try:
        bot.send_message(cid, pricing, parse_mode="Markdown")
        logger.info(f"Pricing sent to user {cid}")
    except Exception as e:
        logger.error(f"Error sending pricing to {cid}: {e}")
        bot.send_message(cid, "⚠️ Có lỗi xảy ra! Vui lòng thử lại sau.", parse_mode="Markdown")

@bot.message_handler(func=lambda m: m.text.strip() == "Liên Hệ Admin")
def handle_contact(msg: Message):
    cid = str(msg.chat.id)
    logger.info(f"User {cid} requested contact")
    try:
        bot.send_message(cid, "📩 Liên hệ admin: @Truongdong1920\n☎️", parse_mode="Markdown")
        logger.info(f"Contact info sent to {cid}")
    except Exception as e:
        logger.error(f"Error sending contact info to {cid}: {e}")
        bot.send_message(cid, "⚠️ Có lỗi xảy ra! Vui lòng thử lại sau.", parse_mode="Markdown")

# ================== ADMIN COMMANDS ==================
@bot.message_handler(commands=['generatekey'])
def handle_generate_key(msg: Message):
    cid = str(msg.chat.id)
    if cid != ADMIN_ID:
        bot.send_message(cid, "🚫 Chỉ admin được sử dụng lệnh này!", parse_mode="Markdown")
        return
    
    try:
        _, duration = msg.text.split()
        duration = float(duration) if duration.lower() != "inf" else float('inf')
        new_key = generate_random_key()
        
        keys = load_data(KEYS_FILE)
        keys[new_key] = duration
        save_data(keys, KEYS_FILE)
        
        expiry = "vĩnh viễn" if duration == float('inf') else f"{duration} ngày"
        bot.send_message(cid, f"🔑 **Key mới!**:\n`{new_key}`\n⏳ Thời hạn: {expiry}", parse_mode="Markdown")
        
    except Exception as e:
        bot.send_message(cid, "⚠️ Định dạng: `/generatekey <số_ngày|inf>`", parse_mode="Markdown")

@bot.message_handler(commands=['listkeys'])
def handle_list_keys(msg: Message):
    cid = str(msg.chat.id)
    if cid != ADMIN_ID:
        bot.send_message(cid, "🚫 Chỉ admin được sử dụng lệnh này!", parse_mode="Markdown")
        return
    
    keys = load_data(KEYS_FILE)
    if not keys:
        bot.send_message(cid, "🔑 Không có key nào trong hệ thống!", parse_mode="Markdown")
        return
    
    response = "🔑 **Danh Sách Key**\n\n"
    for key, duration in sorted(keys.items()):
        expiry = "vĩnh viễn" if duration == float('inf') else f"{duration} ngày"
        user_id = None
        for uid, data in user_data.items():
            if data.get("key") == key:
                user_id = uid
                break
        response += f"Key: `{key}`\nThời hạn: {expiry}\n" + \
                    (f"Được sử dụng bởi: {user_id}\n\n" if user_id else "Trạng thái: Chưa sử dụng\n\n")
    
    bot.send_message(cid, response, parse_mode="Markdown")
    logger.info(f"Admin {cid} listed all keys")

@bot.message_handler(commands=['revokey'])
def handle_revoke_key(msg: Message):
    cid = str(msg.chat.id)
    if cid != ADMIN_ID:
        bot.send_message(cid, "🚫 Chỉ admin được sử dụng lệnh này!", parse_mode="Markdown")
        return
    
    try:
        _, key = msg.text.split()
        keys = load_data(KEYS_FILE)
        if key not in keys:
            bot.send_message(cid, "❌ Key không tồn tại!", parse_mode="Markdown")
            return
        
        del keys[key]
        save_data(keys, KEYS_FILE)
        
        for uid, data in user_data.items():
            if data.get("key") == key:
                data.pop("key", None)
                data.pop("key_expiry", None)
                save_data(user_data, DATA_FILE)
                break
        
        bot.send_message(cid, f"✅ Key `{key}` đã bị thu hồi thành công!", parse_mode="Markdown")
        logger.info(f"Admin {cid} revoked key {key}")
        
    except Exception as e:
        bot.send_message(cid, "⚠️ Định dạng: `/revokey <mã_key>`", parse_mode="Markdown")

@bot.message_handler(commands=['stats'])
def handle_stats(msg: Message):
    cid = str(msg.chat.id)
    if cid != ADMIN_ID:
        bot.send_message(cid, "🚫 Chỉ admin được sử dụng lệnh này!", parse_mode="Markdown")
        return
    
    tai_ratio = {}
    for hash_type, history in prediction_model.global_history.items():
        if history:
            tai_ratio[hash_type] = 100 * sum(1 for h in history if h == 'Tài') / len(history)
    
    stats_msg = (
        f"📊 **Thống Kê Hệ Thống**\n\n"
        f"👤 Tổng users: {len(user_data)}\n"
        f"🔑 Tổng keys: {len(load_data(KEYS_FILE))}\n"
        f"🎯 Lịch sử dự đoán:\n"
    )
    for hash_type, ratio in tai_ratio.items():
        stats_msg += f"- {hash_type.upper()}: {len(prediction_model.global_history[hash_type])} (Tài: {ratio:.1f}%)\n"
    stats_msg += f"📈 MD5 Accuracy: {prediction_model.calculate_md5_accuracy():.2f}%"
    
    bot.send_message(cid, stats_msg, parse_mode="Markdown")

# ================== RUN BOT ==================
if __name__ == "__main__":
    logger.info("Starting bot...")
    print("🤖 Bot đang chạy...")
    try:
        bot.infinity_polling()
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        print(f"❌ Lỗi: {e}")
