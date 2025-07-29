import logging
import json
from datetime import datetime
from aiogram import Bot, Dispatcher, types, executor
from aiogram.types import ParseMode
from datetime import timedelta
import pytz
import hashlib
import math

# === CẤU HÌNH ===
TOKEN = "8432695700:AAF1_Q30qhMpOAL2ZhsgnVsQF0_l9wZwP2E"
ADMIN_ID =[6902698316,6381480476] # ID admin chính

activated_users = {}

try:
    with open("activated_users.json", "r", encoding="utf-8") as f:
        activated_users = json.load(f)
except FileNotFoundError:
    activated_users = {}

# Gán quyền vĩnh viễn cho ADMIN_ID
activated_users[str(ADMIN_ID)] = {"expires": "vĩnh viễn"}

def save_activated_users():
    with open("activated_users.json", "w", encoding="utf-8") as f:
        json.dump(activated_users, f, ensure_ascii=False, indent=2)

def is_admin(user_id):
    return user_id == ADMIN_ID

def check_user(user_id):
    try:
        with open("activated_users.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return False, None

    if str(user_id) in data:
        expire = data[str(user_id)]["expires"]
        if expire == "vĩnh viễn":
            return True, "vĩnh viễn"
        else:
            exp_date = datetime.strptime(expire, "%Y-%m-%d %H:%M:%S")

            # Đảm bảo exp_date có thông tin múi giờ
            timezone = pytz.timezone("Asia/Ho_Chi_Minh")  # Sử dụng múi giờ phù hợp
            exp_date = timezone.localize(exp_date)  # Thêm thông tin múi giờ cho exp_date

            # Lấy thời gian hiện tại với múi giờ
            now = datetime.now(timezone)

            # Kiểm tra thời gian hết hạn
            if now < exp_date:
                return True, expire
            else:
                return False, expire
    return False, None
    
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# ======== Hủy kích hoạt theo hẹn giờ ========
def schedule_deactivation(user_id: int, hours: int):
    run_time = datetime.now(pytz.utc) + timedelta(hours=hours)
    job_id = f"deactivate_{user_id}"
    if scheduler.get_job(job_id):
        scheduler.remove_job(job_id)
    scheduler.add_job(
        lambda: asyncio.create_task(deactivate_user(user_id)),
        trigger="date",
        run_date=run_time,
        id=job_id,
        timezone=pytz.utc
    )

async def deactivate_user(user_id: int):
    active_users.pop(user_id, None)
    save_activated_users()
    try:
        await bot.send_message(user_id, "⏰ Thời hạn sử dụng đã hết. Bot của bạn đã bị hủy kích hoạt.")
    except Exception as e:
        logging.error(f"Lỗi khi gửi tin nhắn hủy kích hoạt: {e}")
           
def generate_sha224(md5_hash):
    return hashlib.sha224(md5_hash.encode('utf-8')).hexdigest()

def calculate_entropy(md5_hash):
    freq = Counter(md5_hash)
    prob = [freq[char] / len(md5_hash) for char in freq]
    entropy = -sum(p * math.log2(p) for p in prob)
    return round(entropy, 4)

def geometric_mean(values):
    product = 1
    for value in values:
        product *= value
    return product ** (1 / len(values))

def bit_1_ratio(md5_hash):
    binary_rep = bin(int(md5_hash, 16))[2:].zfill(128)
    return binary_rep.count("1") / len(binary_rep)

def hex_greater_than_8_ratio(md5_hash):
    return sum(1 for char in md5_hash if int(char, 16) >= 8) / len(md5_hash)

def standard_deviation(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))

def fibonacci_mod(x, mod):
    fib = [0, 1]
    while len(fib) <= x:
        fib.append(fib[-1] + fib[-2])
    return fib[x] % mod

def analyze_md5_advanced(md5_hash):
    hex_pairs = [int(md5_hash[i:i+2], 16) for i in range(0, len(md5_hash), 2)]
    md5_int = int(md5_hash, 16)

    digits_sum = sum(int(char, 16) for char in md5_hash)
    hex_sum = sum(hex_pairs)
    binary_ones = bin(md5_int).count("1")
    bit_1_percentage = bit_1_ratio(md5_hash)
    hex_greater_than_8 = hex_greater_than_8_ratio(md5_hash)
    
    xor_value = 0
    for value in hex_pairs:
        xor_value ^= value

    lucas = [2, 1]
    for _ in range(14):
        lucas.append(lucas[-1] + lucas[-2])
    lucas_weighted_sum = sum(a * b for a, b in zip(hex_pairs[:15], lucas[:15]))

    hex_std_dev = standard_deviation(hex_pairs)
    complexity = len(set(md5_hash))
    fourier_energy = sum(abs(hex_pairs[i] - hex_pairs[i - 1]) for i in range(1, len(hex_pairs)))

    sha224_hash = generate_sha224(md5_hash)
    sha224_sum = sum(int(sha224_hash[i:i+2], 16) for i in range(0, len(sha224_hash), 2))

    first_half, second_half = md5_hash[:16], md5_hash[16:]
    symmetry_score = sum(1 for i in range(16) if first_half[i] == second_half[i])
    geometric_mean_value = geometric_mean(hex_pairs)
    combined_xor = xor_value ^ int(sha224_hash[:2], 16)
    fibonacci_score = fibonacci_mod(digits_sum, 100)

    blake2b_xor = 0
    for i in range(0, len(sha224_hash), 2):
        blake2b_xor ^= int(sha224_hash[i:i+2], 16)

    weighted_edge = (hex_pairs[0] * 3 + hex_pairs[-1] * 2) % 100
    prime_mods = [43, 47, 53, 59, 61, 67]
    mod_values = [hex_sum % prime for prime in prime_mods]
    max_repeating_char = max(md5_hash.count(char) for char in set(md5_hash))
    odd_chars = sum(1 for char in md5_hash if int(char, 16) % 2 == 1)
    middle_bytes = sum(hex_pairs[len(hex_pairs)//4: 3*len(hex_pairs)//4])
    fibo_in_md5 = sum(1 for char in md5_hash if char in '12358')
    sha1_symmetry = sum(1 for i in range(16) if sha224_hash[i] == sha224_hash[39-i])
    entropy = calculate_entropy(md5_hash)
    total_xor = xor_value ^ blake2b_xor ^ combined_xor
    last_digit = int(md5_hash[-1], 16)

    total_score = (
        digits_sum * 0.05 + hex_sum * 0.05 + binary_ones * 0.05 +
        bit_1_percentage * 0.1 + hex_greater_than_8 * 0.1 + lucas_weighted_sum * 0.05 +
        hex_std_dev * 0.05 + complexity * 0.05 + fourier_energy * 0.05 +
        sha224_sum * 0.05 + symmetry_score * 0.05 + geometric_mean_value * 0.05 +
        combined_xor * 0.05 + fibonacci_score * 0.05 + blake2b_xor * 0.05 +
        weighted_edge * 0.05 + sum(mod_values) * 0.05 + max_repeating_char * 0.05 +
        odd_chars * 0.05 + middle_bytes * 0.05 + fibo_in_md5 * 0.05 +
        sha1_symmetry * 0.05 + entropy * 0.05 + total_xor * 0.05 +
        last_digit * 0.05
    ) % 100

    return {
        "tai": round(total_score, 2),
        "xiu": round(100 - total_score, 2),
        "last_digit": last_digit,
        "bit_1_percentage": round(bit_1_percentage * 100, 2),
        "hex_8_percentage": round(hex_greater_than_8 * 100, 2),
        "hex_std_dev": round(hex_std_dev, 2),
        "entropy": round(entropy, 4),
    }

@dp.message_handler(commands=["start"])
async def start_cmd(message: types.Message):
    ok, exp = check_user(message.from_user.id)
    if not ok:
        await message.reply("❌ Bạn chưa được cấp quyền sử dụng bot!")
        return
    await message.reply("👋 Chào mừng bạn! Gửi một chuỗi MD5 để tôi phân tích giúp bạn.\nVí dụ: c54954fc1fcaa22a372b618eea9cb9bd")

@dp.message_handler(commands=["help"])
async def help_cmd(message: types.Message):
    is_ad = is_admin(message.from_user.id)
    text = "🌟 TRỢ GIÚP BOT ZEALAND PREMIUM 🌟\n"
    text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    text += "📋 Danh sách lệnh cơ bản:\n"
    text += "🔹 /start - Khởi động bot và bắt đầu phân tích\n"
    text += "🔹 /id - Xem thông tin ID của bạn\n"
    text += "🔹 /help -  Hiển thị menu trợ giúp này\n"
    text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    if is_ad:
        text += "👑QUẢN TRỊ VIÊN ĐẶC QUYỀN👑\n"
        text += "🔧 Các Lệnh Quản Lý:\n"
        text += "✅ /adduser <id> <ngày hoặc vĩnh>\n"
        text += "❌ /removeuser <id>\n"
        text += "📢 /broadcast <nội dung>\n"
        text += "🗓 /danhsach - Danh sách người dùng\n"
        text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    text += "ℹ️ Gửi chuỗi MD5 (32 ký tự) để phân tích ngay!\n"
    text += "📞 Liên hệ hỗ trợ: https://t.me/Cstooldudoan11"
    await message.reply(text)

@dp.message_handler(commands=["id"])
async def id_cmd(message: types.Message):
    uid = message.from_user.id
    name = message.from_user.full_name
    is_ad = is_admin(uid)
    ok, exp = check_user(uid)
    status = "👑 Admin" if is_ad else ("✅ Đã kích hoạt" if ok else "❌ Chưa kích hoạt")
    text = [
        "🆔 THÔNG TIN NGƯỜI DÙNG 🆔",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"👤 Tên: {name}",
        f"🔢 ID: {uid}",
        f"📊 Trạng Thái: {status}",
        f"⏰ Hạn Dùng: {exp}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📞 Liên hệ:https://t.me/Cstooldudoan11"
    ]
    await message.reply("\n".join(text))

# === ADMIN: ADD USER ===
@dp.message_handler(commands=["adduser"])
async def add_user(message: types.Message):
    if not is_admin(message.from_user.id):
        return await message.reply("⛔ Bạn không có quyền.")

    parts = message.text.split()
    if len(parts) != 3:
        return await message.reply("❗ Dùng: /adduser <id> <số ngày|vĩnh>")

    user_id = parts[1]
    days = parts[2]

    if days == "vĩnh":
        activated_users[user_id] = {"expires": "vĩnh viễn"}
    else:
        try:
            days = int(days)
            expire_time = datetime.now() + timedelta(days=days)
            activated_users[user_id] = {
                "expires": expire_time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except ValueError:
            return await message.reply("❗ Số ngày không hợp lệ.")

    save_activated_users()
    await message.reply(f"✅ Đã cấp quyền cho ID {user_id} ({'vĩnh viễn' if days == 'vĩnh' else f'{days} ngày'}).")

# === ADMIN: REMOVE USER ===
@dp.message_handler(commands=["removeuser"])
async def remove_user(message: types.Message):
    if not is_admin(message.from_user.id):
        return await message.reply("⛔ Bạn không có quyền.")
    parts = message.text.split()
    if len(parts) != 2:
        return await message.reply("❗ Dùng: /removeuser <id>")

    user_id = parts[1]
    if user_id in activated_users:
        del activated_users[user_id]
        save_activated_users()
        await message.reply(f"❌ Đã xóa quyền của ID {user_id}.")
    else:
        await message.reply("⚠️ ID không tồn tại.")

# === ADMIN: BROADCAST ===
@dp.message_handler(commands=["broadcast"])
async def broadcast(message: types.Message):
    if not is_admin(message.from_user.id):
        return await message.reply("⛔ Bạn không có quyền.")
    content = message.text.replace("/broadcast", "").strip()
    if not content:
        return await message.reply("❗ Dùng: /broadcast <nội dung>")

    success, fail = 0, 0
    for uid in activated_users:
        try:
            await bot.send_message(uid, f"📢 THÔNG BÁO:\n\n{content}")
            success += 1
        except:
            fail += 1
    await message.reply(f"✅ Gửi thành công: {success}\n❌ Thất bại: {fail}")


@dp.message_handler(commands=["danhsach"])
async def danhsach_cmd(message: types.Message):
    if not is_admin(message.from_user.id):
        return
    lines = ["📋 Danh sách người dùng đã kích hoạt:"]
    for uid, info in activated_users.items():
        if uid == str(ADMIN_ID):
            lines.append(f"👑 Admin ({uid}) - Hạn: Vĩnh viễn")
        else:
            lines.append(f"👤 {uid} - Hạn: {info['expires']}")
    await message.reply("\n".join(lines))

@dp.message_handler(lambda msg: len(msg.text) == 32 and all(c in '0123456789abcdefABCDEF' for c in msg.text))
async def md5_handler(message: types.Message):
    ok, _ = check_user(message.from_user.id)
    if not ok:
        await message.reply("🚫 Bạn chưa được cấp quyền sử dụng bot này")
        return
    result = analyze_md5_advanced(message.text.lower())

    reply_text = (
    f"<b>🎰 PHÂN TÍCH MD5 SIÊU CHUẨN 🔮✨🌌🎰</b>\n\n"
    f"🔮 <code>{message.text.lower()}</code>🔮\n"
    f"🔢 Số cuối: <b>{result['last_digit']}</b> | Entropy: <b>{result['entropy']}</b>\n"
    f"⚙️ Tỷ lệ bit 1:  <b>{result['bit_1_percentage']}%</b>\n"
    f"🔢 Tỷ lệ Hex ≥8: <b>{result['hex_8_percentage']}%</b>\n"
    f"📉 Độ lệch chuẩn Hex: <b>{result['hex_std_dev']}</b>\n"
    f"🌌 <b>Kết quả:🔥 </b> {'TÀI' if result['tai'] >= 50 else 'XỈU'}🌌\n"
    f"💥 Tài: <b>{result['tai']}%</b>\n"
    f"💦 Xỉu: <b>{result['xiu']}%</b>\n\n"
    f"👤<b>{message.from_user.full_name}</b>\n"
    )
    await message.reply(reply_text, parse_mode="HTML")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
  
