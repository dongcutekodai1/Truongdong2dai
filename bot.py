import logging
import json
from datetime import datetime
from aiogram import Bot, Dispatcher, types, executor
from collections import Counter
from aiogram.types import ParseMode
from datetime import timedelta
import pytz
import hashlib
import math
from collections import Counter
from math import log2

def entropy(s):
    prob = [v / len(s) for v in Counter(s).values()]
    return -sum(p * log2(p) for p in prob)


# === CẤU HÌNH ===
TOKEN = "8432695700:AAF1_Q30qhMpOAL2ZhsgnVsQF0_l9wZwP2E"
ADMIN_ID = 6902698316,6381480476  # ID admin chính

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
           
def calc_result(md5):
    num = int(md5, 16)
    xiu = num % 100
    tai = 100 - xiu
    return xiu, tai

def get_dice(md5: str, num_dice=3):
    # Lấy các ký tự hex đầu tiên theo số lượng cần dùng
    digits = [int(c, 16) % 6 + 1 for c in md5[:num_dice]]
    return digits, sum(digits)

def ai_score(xiu, tai, ent):
    diff = abs(xiu - tai)
    balance = 100 - abs(50 - xiu) * 2
    rand_boost = max(0, (ent - 3.5) * 15)
    return min(100.0, diff * 1.3 + balance * 0.5 + rand_boost)

def deep_score(md5):
    sha = hashlib.sha256(md5.encode()).hexdigest()
    ent_md5, ent_sha = entropy(md5), entropy(sha)
    diff = abs(ent_md5 - ent_sha)
    rep_pen = sum(md5.count(c) > 2 for c in set(md5)) / len(set(md5))
    half = len(md5)//2
    sym = sum(1 for a,b in zip(md5[:half], md5[-half:]) if a==b) / half
    score = (ent_sha * 10) + (50 - diff * 20) + (sym * 30) - (rep_pen * 15)
    return max(0.0, min(100.0, score))

def draw_bar(label, val, color):
    bar = "█" * int(30 * val / 100) + "-" * (30 - int(30 * val / 100))
    print(f"{color}{BOLD}{label:<5} |{bar}| {val:6.2f}%{RESET}")

def analyze_and_predict(history: list) -> tuple:
    if len(history) != 5:
        return "Không đủ dữ liệu", 50
 
    patterns = {
        "bệt": lambda h: h.count(h[-1]) >= 3,
        "đảo_1_1": lambda h: all(h[i] != h[i + 1] for i in range(4)),
        "kép_2_2": lambda h: h[:2] == h[2:] and h[:2] in [["Tài", "Tài"], ["Xỉu", "Xỉu"]],
        "1_2_3": lambda h: h[:1] == h[1:3] and h[3:] == [h[1]] * 2,
        "3_3": lambda h: h[:3] == [h[0]] * 3 and h[3:] == [h[3]] * 3,
        "inverse_pattern": lambda h: h.count("Tài") >= 4 and h[-1] == "Xỉu",
        "synchronized_pattern": lambda h: len(set(h)) == 1,
        "u_shaped": lambda h: h[0] == "Tài" and h[-1] == "Tài" and h[1] == "Xỉu" and h[2] == "Xỉu",
        "parallel_pattern": lambda h: h[:3] == ["Tài"] * 3 and h[3:] == ["Xỉu"] * 2,
        "wavy_pattern": lambda h: all(h[i] != h[i + 1] for i in range(len(h) - 1)),
        "tai_tai_xiu_xiu": lambda h: h[:2] == ["Tài", "Tài"] and h[2:4] == ["Xỉu", "Xỉu"],
        "cầu_công": lambda h: h.count("Tài") == 3 and h.count("Xỉu") == 2,
        "đảo_chieu": lambda h: h == ["Xỉu", "Tài", "Xỉu", "Tài", "Xỉu"],
    }

    predictions = {"Tài": 0, "Xỉu": 0}

    for name, rule in patterns.items():
        if rule(history):
            if name in ["bệt", "synchronized_pattern"]:
                predictions[history[-1]] += 3
            elif name == "đảo_1_1":
                predictions["Tài" if history[-1] == "Xỉu" else "Xỉu"] += 3
            elif name == "kép_2_2":
                predictions["Tài" if history[-1] == "Xỉu" else "Xỉu"] += 2
            elif name == "1_2_3":
                predictions[history[-1]] += 2
            elif name == "3_3":
                predictions["Tài" if history[-1] == "Xỉu" else "Xỉu"] += 2
            elif name == "inverse_pattern":
                predictions["Tài"] += 4
            elif name == "u_shaped":
                predictions["Xỉu"] += 3
            elif name == "parallel_pattern":
                predictions["Xỉu"] += 2
            elif name == "wavy_pattern":
                predictions["Tài" if history[-1] == "Xỉu" else "Xỉu"] += 2
            elif name == "tai_tai_xiu_xiu":
                predictions["Xỉu"] += 3
            elif name == "cầu_công":
                predictions["Tài"] += 2
            elif name == "đảo_chieu":
                predictions["Tài"] += 3

    total = predictions["Tài"] + predictions["Xỉu"]
    if total == 0:
        counts = {"Tài": history.count("Tài"), "Xỉu": history.count("Xỉu")}
        if counts["Tài"] > counts["Xỉu"]:
            return "Tài", 60
        elif counts["Tài"] < counts["Xỉu"]:
            return "Xỉu", 60
        else:
            return "Tài", 50

    tai_percentage = (predictions["Tài"] / total) * 100
    xiu_percentage = (predictions["Xỉu"] / total) * 100

    if tai_percentage > xiu_percentage:
        return "Tài", round(tai_percentage, 2)
    else:
        return "Xỉu", round(xiu_percentage, 2)


def convert_history(input_str: str) -> list:
    mapping = {"T": "Tài", "X": "Xỉu"}
    try:
        return [mapping[char] for char in input_str.upper()]
    except KeyError:
        return []

@dp.message_handler(commands=["start"])
async def start_cmd(message: types.Message):
    ok, exp = check_user(message.from_user.id)
    if not ok:
        await message.reply("❌ Bạn chưa được cấp quyền sử dụng bot!")
        return
    await message.reply("👋 Chào mừng bạn! Gửi một chuỗi MD5 hoặc một chuỗi lịch sử cầu để tôi phân tích giúp bạn.\nVí dụ: c54954fc1fcaa22a372b618eea9cb9bd\nHoặc: TXTXX")

@dp.message_handler(commands=["help"])
async def help_cmd(message: types.Message):
    is_ad = is_admin(message.from_user.id)
    text = "🌟 TRỢ GIÚP BOT NEURIX PREMIUM 🌟\n"
    text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    text += "📋 Danh sách lệnh cơ bản:\n"
    text += "🔹 /start - Khởi động bot và bắt đầu phân tích\n"
    text += "🔹 /pricelist - Bảng giá bot\n"
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

@dp.message_handler(commands=["pricelist"])
async def pricelist_cmd(message: types.Message):
    await message.reply(
        "💵 <b>BẢNG GIÁ SỬ DỤNG BOT:</b>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "🔹 <b>1 Tuần</b>   : 100K\n"
        "🔹 <b>1 Tháng</b>  : 250K\n"
        "🔹 <b>2 Tháng</b>  : 400K\n"
        "🔹 <b>Vĩnh viễn</b>: 600K\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "💬 Liên hệ admin để kích hoạt: https://t.me/Cstooldudoan11",
        parse_mode="HTML"
    )

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

import html  # Đảm bảo đã import để escape MD5

@dp.message_handler(lambda msg: len(msg.text) == 32 and all(c in '0123456789abcdefABCDEF' for c in msg.text))
async def md5_handler(message: types.Message):
    ok, _ = check_user(message.from_user.id)
    if not ok:
        await message.reply("🚫 Bạn chưa được cấp quyền sử dụng bot này")
        return

    md5 = message.text.lower()
    xiu, tai = calc_result(md5)
    ent = entropy(md5)
    ai = ai_score(xiu, tai, ent)
    deep = deep_score(md5)
    dice, total = get_dice(md5, num_dice=3)

    md5_clean = html.escape(md5)  # Escape MD5 để tránh lỗi HTML

    result = (
        f"<b>🎰 PHÂN TÍCH MD5 SIÊU CHUẨN 🔮✨🌌🎰</b>\n\n"
        f"<b>🔮 MD5:</b> <code>{md5_clean}</code> 🔮\n\n"
        f"🧮 <b>Entropy:</b> {ent:.4f}\n"
        f"⚙️ <b>AI Score:</b> {ai:.2f}%\n"
        f"📉 <b>Deep Score:</b> {deep:.2f}%\n"
        f"🌌 <b>Kết quả dự đoán:</b> {'<b>❄️ TÀI</b>' if tai > xiu else '<b>🔥 XỈU</b>'}🌌\n"
        f"💥 <b>Xỉu:</b> {xiu}%[🟥🟩🟦⬜️⬜️⬜️]\n"
        f"💦 <b>Tài:</b> {tai}%[🟥🟩🟦⬜️⬜️⬜️]\n"

        f"👤 <i>Yêu cầu bởi: {message.from_user.full_name}</i>"
        
    )

    await message.reply(result, parse_mode="HTML")

     
# Xử lý chuỗi cầu như TXTXX
@dp.message_handler(lambda msg: len(msg.text) == 5 and all(c in "TXtx" for c in msg.text))
async def history_handler(message: types.Message):
    ok, _ = check_user(message.from_user.id)
    if not ok:
        await message.reply("🚫 Bạn chưa được cấp quyền sử dụng bot này")
        return

    history = convert_history(message.text)
    if len(history) != 5:
        await message.reply("❗ Chuỗi không hợp lệ. Chỉ nhập 5 ký tự gồm T và X.")
        return

    prediction, percentage = analyze_and_predict(history)

    # Format kết quả
    history_str = " → ".join(history)
    icon = "🔥" if prediction == "Tài" else "❄️"
    await message.reply(
        f"📊 <b>PHÂN TÍCH LỊCH SỬ CẦU</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🧮 Lịch sử: <code>{history_str}</code>\n"
        f"🔮 Dự đoán tiếp theo: <b>{icon} {prediction}</b>\n"
        f"📈 Độ tin cậy: <b>{percentage:.2f}%</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"👤 <i>Yêu cầu bởi: {message.from_user.full_name}</i>",
        parse_mode="HTML"
    )

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
