from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from aiogram.utils import executor
import hashlib, json, os, re
from functools import wraps
from keep_alive import keep_alive
keep_alive()

TOKEN = "8076967422:AAHAacz8x1NJSLd6ONuLx3Hz_jYJZui6VVw"
ADMIN_USERNAME = "Truongdong1920"
XU_COST = 1

HISTORY_FILE = "lich_su.json"
ADMINS_FILE = "admins.json"
USER_XU_FILE = "user_xu.json"

# Initialize data structures
history = {}
admins = []
user_xu = {}

# Load data from files
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

def load_admins():
    if os.path.exists(ADMINS_FILE):
        with open(ADMINS_FILE, "r") as f:
            return json.load(f)
    else:
        default_admins = [6381480476]  # Default admin ID
        with open(ADMINS_FILE, "w") as f:
            json.dump(default_admins, f, indent=2)
        return default_admins

def save_admins(admin_list):
    with open(ADMINS_FILE, "w") as f:
        json.dump(admin_list, f, indent=2)

def load_user_xu():
    if os.path.exists(USER_XU_FILE):
        with open(USER_XU_FILE, "r") as f:
            return json.load(f)
    return {}

def save_user_xu():
    with open(USER_XU_FILE, "w") as f:
        json.dump(user_xu, f, indent=2)

# Load initial data
admins = load_admins()
user_xu = load_user_xu()

# Admin decorator
def admin_only(handler):
    @wraps(handler)
    async def wrapper(message: types.Message, *args, **kwargs):
        if message.from_user.id not in admins:
            await message.reply("❌ Bạn không có quyền sử dụng lệnh này.")
            return
        return await handler(message, *args, **kwargs)
    return wrapper

# Keyboard generator
def get_user_keyboard(user_id):
    if user_id in admins:
        return ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="Gửi MD5 để phân tích"), KeyboardButton(text="Bảng giá xu")],
                [KeyboardButton(text="Cấp xu"), KeyboardButton(text="Thêm admin"), KeyboardButton(text="Xóa admin")],
                [KeyboardButton(text="Liên hệ admin"), KeyboardButton(text="Danh sách người dùng")]
            ],
            resize_keyboard=True
        )
    else:
        return ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="Gửi MD5 để phân tích"), KeyboardButton(text="Bảng giá xu")],
                [KeyboardButton(text="Liên hệ admin")],
            ],
            resize_keyboard=True
        )

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# MD5 Analysis Function
def analyze_md5(md5_hash):
    if md5_hash in history:
        result = history[md5_hash]
        return (result, 100, 0) if result == "Tài" else (result, 0, 100)

    digit_values = [int(char, 16) for char in md5_hash if char.isdigit()]
    letter_values = [ord(char) for char in md5_hash if char.isalpha()]

    if not digit_values or not letter_values:
        return "Xỉu", 0, 100

    digit_sum = sum(digit_values)
    letter_sum = sum(letter_values)

    xor_digit = 0
    for num in digit_values:
        xor_digit ^= num

    xor_letter = 0
    for char in letter_values:
        xor_letter ^= char

    squared_digit_sum = sum(x**2 for x in digit_values) % 100
    squared_letter_sum = sum(x**2 for x in letter_values) % 100

    hex_blocks = [int(md5_hash[i:i+4], 16) for i in range(0, len(md5_hash), 4)]
    hex_weighted_sum = sum((i + 1) * hex_blocks[i % len(hex_blocks)] for i in range(len(hex_blocks))) % 100

    even_count = sum(1 for x in digit_values if x % 2 == 0)
    odd_count = sum(1 for x in digit_values if x % 2 != 0)

    final_score = (
        digit_sum * 2 + letter_sum +
        xor_digit * 3 + xor_letter +
        squared_digit_sum * 2 + squared_letter_sum +
        hex_weighted_sum +
        even_count * 5 - odd_count * 3
    ) % 100

    result = "Tài" if final_score % 2 == 0 else "Xỉu"
    history[md5_hash] = result
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    
    return result, final_score, 100 - final_score

# Command Handlers
@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    user_id = message.from_user.id
    username = message.from_user.username or "Không có"
    current_xu = user_xu.get(str(user_id), 0)
    
    welcome_msg = (
        "✨ CHÀO MỪNG ĐẾN VỚI TOOL PHÂN TÍCH MD5 ✨\n\n"
        f"👤 Tài khoản: @{username}\n"
        f"🆔 ID: {user_id}\n"
        f"💰 Xu hiện có: {current_xu}\n\n"
        "Chọn chức năng bên dưới để bắt đầu trải nghiệm nhé!"
    )
    
    await message.reply(welcome_msg, reply_markup=get_user_keyboard(user_id))

@dp.message_handler(lambda message: message.text == "Thoát")
async def exit_handler(message: types.Message):
    await message.reply("👋 Hẹn gặp lại bạn lần sau!", reply_markup=ReplyKeyboardRemove())

@dp.message_handler(lambda message: message.text == "Gửi MD5 để phân tích")
async def prompt_md5_handler(message: types.Message):
    await message.reply("Vui lòng gửi mã MD5 32 ký tự để tôi phân tích.")

@dp.message_handler(lambda message: message.text == "Bảng giá xu")
async def show_xu_price_table(message: types.Message):
    xu_table = (
        "💰 *Bảng Giá Xu*\n\n"
        "1 Xu = 1,000 VNĐ\n"
        "10 Xu = 9,500 VNĐ (-5% giảm giá)\n"
        "50 Xu = 45,000 VNĐ (-10% giảm giá)\n"
        "100 Xu = 85,000 VNĐ (-15% giảm giá)\n\n"
        "Liên hệ admin để mua số lượng lớn với giá tốt hơn!"
    )
    await message.reply(xu_table, parse_mode="Markdown", reply_markup=get_user_keyboard(message.from_user.id))

@dp.message_handler(lambda message: message.text == "Cấp xu")
@admin_only
async def cap_xu_handler(message: types.Message):
    await message.reply(
        "Nhập theo định dạng: `ID_XU`\nVí dụ: `123456789_100` (cấp 100 Xu cho user 123456789)",
        parse_mode="Markdown",
        reply_markup=get_user_keyboard(message.from_user.id)
    )

@dp.message_handler(lambda message: re.fullmatch(r"\d+_\d+", message.text or ""))
@admin_only
async def cap_xu_process(message: types.Message):
    try:
        user_id_str, xu_str = message.text.split("_")
        user_id = int(user_id_str)
        xu_cung_cap = int(xu_str)

        current_xu = user_xu.get(str(user_id), 0)
        user_xu[str(user_id)] = current_xu + xu_cung_cap
        save_user_xu()
        
        await message.reply(
            f"✅ Đã cấp {xu_cung_cap} xu cho user ID {user_id}.",
            reply_markup=get_user_keyboard(message.from_user.id)
        )
    except Exception as e:
        await message.reply(f"❌ Lỗi: {str(e)}")

@dp.message_handler(lambda message: message.text == "Thêm admin")
@admin_only
async def add_admin_handler(message: types.Message):
    await message.reply(
        "Nhập ID người dùng muốn thêm làm Admin:",
        reply_markup=get_user_keyboard(message.from_user.id)
    )

@dp.message_handler(lambda message: message.text == "Xóa admin")
@admin_only
async def remove_admin_handler(message: types.Message):
    await message.reply(
        "Nhập ID người dùng muốn xóa khỏi Admin:",
        reply_markup=get_user_keyboard(message.from_user.id)
    )

@dp.message_handler(lambda message: message.text.isdigit() and message.reply_to_message and 
                   (message.reply_to_message.text == "Nhập ID người dùng muốn thêm làm Admin:" or 
                    message.reply_to_message.text == "Nhập ID người dùng muốn xóa khỏi Admin:"))
@admin_only
async def admin_management_handler(message: types.Message):
    admin_id = int(message.text)
    if message.reply_to_message.text.startswith("Nhập ID người dùng muốn thêm"):
        if admin_id in admins:
            await message.reply(f"⚠️ ID {admin_id} đã là admin rồi.")
        else:
            admins.append(admin_id)
            save_admins(admins)
            await message.reply(f"✅ Đã thêm ID {admin_id} vào danh sách admin.")
    else:
        if admin_id not in admins:
            await message.reply(f"⚠️ ID {admin_id} không phải admin.")
        elif admin_id == 6381480476:  # Default admin ID
            await message.reply("❌ Không thể xóa admin mặc định.")
        else:
            admins.remove(admin_id)
            save_admins(admins)
            await message.reply(f"✅ Đã xóa ID {admin_id} khỏi danh sách admin.")

@dp.message_handler(lambda message: message.text == "Liên hệ admin")
async def contact_admin_handler(message: types.Message):
    await message.reply(
        f"👉 Liên hệ admin: https://t.me/{ADMIN_USERNAME}",
        reply_markup=get_user_keyboard(message.from_user.id)
    )

@dp.message_handler(lambda message: message.text == "Gửi thông báo")
@admin_only
async def broadcast_prompt(message: types.Message):
    await message.reply(
        "Nhập nội dung thông báo bạn muốn gửi đến tất cả người dùng:",
        reply_markup=ReplyKeyboardRemove()
    )

@dp.message_handler(lambda message: message.reply_to_message and 
                   message.reply_to_message.text == "Nhập nội dung thông báo bạn muốn gửi đến tất cả người dùng:")
@dp.message_handler(commands=["broadcast"])
@admin_only
async def broadcast_message(message: types.Message):
    content = message.get_args()
    if not content:
        await message.reply("Vui lòng nhập nội dung thông báo ngay sau lệnh, vd: /broadcast Hello mọi người!")
        return

    success = 0
    failed = 0
    
    for user_id in user_xu.keys():
        try:
            await bot.send_message(int(user_id), f"📢 Thông báo từ Admin:\n\n{content}")
            success += 1
            await asyncio.sleep(0.1)  # tránh rate limit
        except Exception as e:
            failed += 1
            print(f"Lỗi gửi tin nhắn user {user_id}: {e}")
    
    await message.reply(
        f"✅ Đã gửi thông báo đến {success} người dùng.\n❌ Không gửi được đến {failed} người.",
        reply_markup=get_user_keyboard(message.from_user.id)
    )

@dp.message_handler(lambda message: message.text == "Danh sách người dùng")
@admin_only
async def user_list_handler(message: types.Message):
    if not user_xu:
        await message.reply("Chưa có người dùng nào sử dụng bot.")
        return
    
    user_list = "📋 Danh sách người dùng:\n\n"
    for user_id, xu in user_xu.items():
        user_list += f"• ID: {user_id} - Xu: {xu}\n"
    
    await message.reply(
        user_list,
        reply_markup=get_user_keyboard(message.from_user.id)
    )

@dp.message_handler(regexp=r"^[a-f0-9]{32}$")
async def md5_analyze_handler(message: types.Message):
    md5 = message.text.lower()
    user_id = str(message.from_user.id)
    
    if user_id not in user_xu:
        user_xu[user_id] = 0
    
    if user_xu[user_id] < XU_COST:
        await message.reply(
            f"❌ Bạn không đủ {XU_COST} xu để phân tích MD5.\nVui lòng liên hệ admin để mua thêm xu.",
            reply_markup=get_user_keyboard(message.from_user.id)
        )
        return
    
    # Deduct Xu
    user_xu[user_id] -= XU_COST
    save_user_xu()
    
    # Analyze MD5
    result, tai_percent, xiu_percent = analyze_md5(md5)
    
    await message.reply(
        f"🎰 PHÂN TÍCH MD5 SIÊU CHUẨN 🔮✨🌌🎰\n"
        f"📌 MD5: {md5}\n"
        f"💥 Tài: {tai_percent}%[🟥🟩🟦⬜️⬜️⬜️]\n"
        f"💦 Xỉu: {xiu_percent}%[🟥🟩🟦⬜️⬜️⬜️]\n"
        f"💰 XU CÒN LẠI: {user_xu[user_id]}",
        parse_mode="Markdown",
        reply_markup=get_user_keyboard(message.from_user.id)
    )

if __name__ == "__main__":
    print("Bot đang chạy...")
    executor.start_polling(dp)
