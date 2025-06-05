import asyncio # Added asyncio
import json
import os
from functools import wraps

from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from aiogram.utils import executor

# --- CONFIGURATION ---
TOKEN = "8076967422:AAFfQfggn_PdDQx5uBtUnAZO-PlsNo9eNnI"  # IMPORTANT: Replace with YOUR REAL BOT TOKEN
ADMIN_USERNAME = "Truongdong1920"
XU_COST = 1

HISTORY_FILE = "lich_su.json"
ADMINS_FILE = "admins.json"
USER_XU_FILE = "user_xu.json"

# --- Initialize data structures ---
history = {}
admins = []
user_xu = {}

# --- Data Loading and Saving ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode {HISTORY_FILE}. Starting with empty history.")
                return {}
    return {}

def save_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def load_admins():
    if os.path.exists(ADMINS_FILE):
        with open(ADMINS_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode {ADMINS_FILE}. Initializing with default admin.")
    # Fallback to default if file is missing or corrupt
    default_admins = [6381480476]  # Default admin ID
    with open(ADMINS_FILE, "w", encoding="utf-8") as f:
        json.dump(default_admins, f, indent=2, ensure_ascii=False)
    return default_admins

def save_admins(admin_list):
    with open(ADMINS_FILE, "w", encoding="utf-8") as f:
        json.dump(admin_list, f, indent=2, ensure_ascii=False)

def load_user_xu():
    if os.path.exists(USER_XU_FILE):
        with open(USER_XU_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode {USER_XU_FILE}. Starting with empty user xu data.")
                return {}
    return {}

def save_user_xu():
    with open(USER_XU_FILE, "w", encoding="utf-8") as f:
        json.dump(user_xu, f, indent=2, ensure_ascii=False)

# Load initial data
history = load_history()
admins = load_admins()
user_xu = load_user_xu()

# --- Admin Decorator ---
def admin_only(handler):
    @wraps(handler)
    async def wrapper(message: types.Message, *args, **kwargs):
        if message.from_user.id not in admins:
            await message.reply("❌ Bạn không có quyền sử dụng lệnh này.")
            return
        return await handler(message, *args, **kwargs)
    return wrapper

# --- Keyboard Generator ---
def get_user_keyboard():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Gửi MD5 để phân tích"), KeyboardButton(text="Bảng giá xu")],
            [KeyboardButton(text="Liên hệ admin")],
            [KeyboardButton(text="Thoát")] # Added Thoát to main keyboard for consistency
        ],
        resize_keyboard=True
    )

# --- Bot Initialization ---
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# --- MD5 Analysis Logic ---
def analyze_md5(md5_hash_str):
    md5_hash_str = md5_hash_str.lower() # Ensure consistency
    if md5_hash_str in history:
        entry = history[md5_hash_str]
        # Handle new format (dict) and potential old format (str)
        if isinstance(entry, dict) and 'result' in entry and 'ratio_tai' in entry and 'ratio_xiu' in entry:
            return entry['result'], entry['ratio_tai'], entry['ratio_xiu']
        elif isinstance(entry, str): # Old format: entry is "Tài" or "Xỉu"
            result = entry
            # For old format, return fixed ratios as per original logic
            if result == "Tài":
                return result, 100.0, 0.0
            else:
                return result, 0.0, 100.0
        # If entry is in an unexpected format, recalculate
        print(f"MD5 {md5_hash_str} found in history with unexpected format. Recalculating.")


    digit_values = [int(char, 16) for char in md5_hash_str if char.isdigit()]
    letter_values = [ord(char) for char in md5_hash_str if char.isalpha()]

    if not digit_values or not letter_values:
        # Default to Xỉu if either list is empty, and store this specific result
        result, ratio_tai, ratio_xiu = "Xỉu", 0.0, 100.0
        history[md5_hash_str] = {"result": result, "ratio_tai": ratio_tai, "ratio_xiu": ratio_xiu}
        save_history()
        return result, ratio_tai, ratio_xiu

    digit_sum = sum(digit_values)
    letter_sum = sum(letter_values)

    xor_digit = 0
    for num in digit_values:
        xor_digit ^= num

    xor_letter = 0
    for char_val in letter_values: # Iterate over numeric ord values
        xor_letter ^= char_val

    squared_digit_sum = sum(x**2 for x in digit_values) % 100
    squared_letter_sum = sum(x**2 for x in letter_values) % 100

    hex_blocks = [int(md5_hash_str[i:i+4], 16) for i in range(0, len(md5_hash_str), 4)]
    hex_weighted_sum = sum((idx + 1) * hex_blocks[idx % len(hex_blocks)] for idx in range(len(hex_blocks))) % 100 # fixed i to idx

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
    ratio_tai = float(final_score)
    ratio_xiu = 100.0 - float(final_score)

    # Store the new analysis result
    history[md5_hash_str] = {"result": result, "ratio_tai": ratio_tai, "ratio_xiu": ratio_xiu}
    save_history()

    return result, ratio_tai, ratio_xiu

# --- Command Handlers ---
@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    user_id = message.from_user.id
    username = message.from_user.username or "Không có"
    # Ensure user is in xu system
    if str(user_id) not in user_xu:
        user_xu[str(user_id)] = 0 # Give 0 xu initially
        save_user_xu()
    current_xu = user_xu.get(str(user_id), 0)

    welcome_msg = (
        "✨ CHÀO MỪNG ĐẾN VỚI TOOL PHÂN TÍCH MD5 ✨\n\n"
        f"👤 Tài khoản: @{username}\n"
        f"🆔 ID: {user_id}\n"
        f"💰 Xu hiện có: {current_xu}\n\n"
        "Chọn chức năng bên dưới để bắt đầu trải nghiệm nhé!"
    )
    await message.reply(welcome_msg, reply_markup=get_user_keyboard())

@dp.message_handler(lambda message: message.text == "Thoát")
async def exit_handler(message: types.Message):
    await message.reply("👋 Hẹn gặp lại bạn lần sau!", reply_markup=ReplyKeyboardRemove())

@dp.message_handler(lambda message: message.text == "Gửi MD5 để phân tích")
async def prompt_md5_handler(message: types.Message):
    await message.reply("Vui lòng gửi mã MD5 (32 ký tự) để tôi phân tích.")

@dp.message_handler(lambda message: message.text == "Bảng giá xu")
async def show_xu_price_table(message: types.Message):
    xu_table = (
        "💰 *Bảng Giá Xu*\n\n"
        "1 Xu = 1,000 VNĐ\n"
        "10 Xu = 9,500 VNĐ (-5% giảm giá)\n"
        "50 Xu = 45,000 VNĐ (-10% giảm giá)\n"
        "100 Xu = 85,000 VNĐ (-15% giảm giá)\n\n"
        f"Liên hệ admin @{ADMIN_USERNAME} để mua số lượng lớn với giá tốt hơn!"
    )
    await message.reply(xu_table, parse_mode="Markdown", reply_markup=get_user_keyboard())

@dp.message_handler(commands=['capxu'])
@admin_only
async def cap_xu_handler(message: types.Message):
    args = message.get_args()
    if not args:
        await message.reply(
            "Nhập theo định dạng: `/capxu USER_ID SỐ_XU`\n"
            "Ví dụ: `/capxu 123456789 100` (cấp 100 Xu cho user 123456789)",
            parse_mode="Markdown"
        )
        return

    try:
        parts = args.split()
        if len(parts) != 2:
            raise ValueError("Sai định dạng. Cần USER_ID và SỐ_XU.")

        user_id_to_cap = int(parts[0])
        xu_cung_cap = int(parts[1])

        if xu_cung_cap < 0:
            await message.reply("❌ Số xu cấp không thể âm.")
            return

        current_xu = user_xu.get(str(user_id_to_cap), 0)
        user_xu[str(user_id_to_cap)] = current_xu + xu_cung_cap
        save_user_xu()

        await message.reply(
            f"✅ Đã cấp {xu_cung_cap} xu cho user ID {user_id_to_cap}.\n"
            f"💰 Xu mới của user {user_id_to_cap}: {user_xu[str(user_id_to_cap)]}"
        )
    except ValueError as e:
        await message.reply(f"❌ Lỗi định dạng: {e}\nĐúng định dạng: /capxu USER_ID SỐ_XU (cả hai phải là số).")
    except Exception as e:
        await message.reply(f"❌ Lỗi không xác định: {str(e)}")


@dp.message_handler(commands=['addadmin'])
@admin_only
async def add_admin_handler(message: types.Message):
    args = message.get_args()
    if not args:
        await message.reply("Nhập ID người dùng muốn thêm làm Admin: /addadmin USER_ID")
        return

    try:
        admin_id = int(args)
        if admin_id in admins:
            await message.reply(f"⚠️ ID {admin_id} đã là admin rồi.")
        else:
            admins.append(admin_id)
            save_admins(admins)
            await message.reply(f"✅ Đã thêm ID {admin_id} vào danh sách admin.")
    except ValueError:
        await message.reply("❌ ID người dùng phải là một số nguyên.")

@dp.message_handler(commands=['deladmin'])
@admin_only
async def remove_admin_handler(message: types.Message):
    args = message.get_args()
    if not args:
        await message.reply("Nhập ID người dùng muốn xóa khỏi Admin: /deladmin USER_ID")
        return

    try:
        admin_id_to_remove = int(args)
        if admin_id_to_remove not in admins:
            await message.reply(f"⚠️ ID {admin_id_to_remove} không có trong danh sách admin.")
        elif admin_id_to_remove == 6906617636 and message.from_user.id != 6906617636 : # Prevent removing default admin unless it's the default admin itself
             await message.reply("❌ Không thể xóa admin mặc định trừ khi bạn là admin đó.")
        elif len(admins) == 1 and admin_id_to_remove in admins:
            await message.reply("❌ Không thể xóa admin cuối cùng. Phải có ít nhất một admin.")
        else:
            admins.remove(admin_id_to_remove)
            save_admins(admins)
            await message.reply(f"✅ Đã xóa ID {admin_id_to_remove} khỏi danh sách admin.")
    except ValueError:
        await message.reply("❌ ID người dùng phải là một số nguyên.")


@dp.message_handler(lambda message: message.text == "Liên hệ admin")
async def contact_admin_handler(message: types.Message):
    await message.reply(
        f"👉 Liên hệ admin: https://t.me/{ADMIN_USERNAME}",
        reply_markup=get_user_keyboard()
    )

@dp.message_handler(commands=['broadcast'])
@admin_only
async def broadcast_message_handler(message: types.Message): # Renamed for clarity
    content = message.get_args()
    if not content:
        await message.reply("Vui lòng nhập nội dung thông báo ngay sau lệnh, vd: /broadcast Hello mọi người!")
        return

    success_count = 0
    failed_count = 0
    
    # Create a copy of keys to iterate over, in case user_xu changes during broadcast (unlikely here)
    user_ids_to_message = list(user_xu.keys())

    for user_id_str in user_ids_to_message:
        try:
            await bot.send_message(int(user_id_str), f"📢 Thông báo từ Admin:\n\n{content}")
            success_count += 1
            await asyncio.sleep(0.1)  # To avoid hitting Telegram rate limits
        except Exception as e:
            failed_count += 1
            print(f"Lỗi gửi tin nhắn đến user {user_id_str}: {e}")
    
    await message.reply(
        f"📣 Thông báo đã được gửi.\n"
        f"✅ Thành công: {success_count} người dùng.\n"
        f"❌ Thất bại: {failed_count} người dùng."
    )

@dp.message_handler(commands=['users'])
@admin_only
async def user_list_handler(message: types.Message):
    if not user_xu:
        await message.reply("Chưa có người dùng nào trong hệ thống.")
        return
    
    user_list_msg = "📋 Danh sách người dùng và số xu:\n\n"
    for user_id, xu_amount in user_xu.items():
        user_list_msg += f"• ID: `{user_id}` - Xu: {xu_amount}\n"
    
    # Handle cases where the message might be too long for Telegram
    if len(user_list_msg) > 4096:
        await message.reply("Danh sách người dùng quá dài để hiển thị. Cân nhắc xuất ra file hoặc chia nhỏ.")
        # Optionally, you could send it as a file:
        # with open("user_list.txt", "w", encoding="utf-8") as f:
        #     f.write(user_list_msg)
        # await message.reply_document(open("user_list.txt", "rb"))
        # os.remove("user_list.txt")
    else:
        await message.reply(user_list_msg, parse_mode="Markdown")


@dp.message_handler(regexp=r"^[a-fA-F0-9]{32}$") # Allow uppercase hex characters too
async def md5_analyze_handler(message: types.Message):
    md5_hash = message.text.lower() # Convert to lowercase for consistent processing & history keys
    user_id_str = str(message.from_user.id)
    
    # Ensure user is in xu system, add if not (e.g., if they interact first time via MD5)
    if user_id_str not in user_xu:
        user_xu[user_id_str] = 0 # Or some initial amount if you give free xu
        save_user_xu()
            
    if user_xu.get(user_id_str, 0) < XU_COST:
        await message.reply(
            f"❌ Bạn không đủ {XU_COST} xu để phân tích MD5.\n"
            f"💰 Xu hiện tại: {user_xu.get(user_id_str, 0)}.\n"
            f"Vui lòng liên hệ admin @{ADMIN_USERNAME} để mua thêm xu.",
            reply_markup=get_user_keyboard()
        )
        return

    # Deduct xu BEFORE analysis
    user_xu[user_id_str] -= XU_COST
    save_user_xu()
    
    await message.reply("🔍 Đang phân tích mã MD5, vui lòng chờ một lát...")
    
    try:
        result, ratio_tai, ratio_xiu = analyze_md5(md5_hash)
        
        reply_text = (
            f"🎰 *PHÂN TÍCH MD5 SIÊU CHUẨN* 🔮✨🌌🎰\n"
            f"📌 *MD5*: `{md5_hash}`\n"
            f"💥 *Kết quả dự đoán*: **{result}**\n"
            f"📈 *Tỷ lệ Tài*: {ratio_tai:.2f}% {'🟥'*int(ratio_tai/10)}{'⬜️'*int((100-ratio_tai)/10)}\n"
            f"📉 *Tỷ lệ Xỉu*: {ratio_xiu:.2f}% {'🟦'*int(ratio_xiu/10)}{'⬜️'*int((100-ratio_xiu)/10)}\n"
            f"💰 *Xu còn lại*: {user_xu[user_id_str]}"
        )
        
        await message.reply(
            reply_text,
            parse_mode="Markdown",
            reply_markup=get_user_keyboard()
        )
    except Exception as e:
        # Refund xu if analysis fails unexpectedly
        user_xu[user_id_str] += XU_COST
        save_user_xu()
        await message.reply(
            f"❌ Đã xảy ra lỗi trong quá trình phân tích: {e}\n"
            f"Xu của bạn đã được hoàn lại. Vui lòng thử lại hoặc liên hệ admin nếu sự cố tiếp diễn.",
            reply_markup=get_user_keyboard()
        )
        print(f"Error during MD5 analysis for {md5_hash}: {e}")


# Fallback handler for any other text not caught by specific handlers
@dp.message_handler(content_types=types.ContentType.TEXT)
async def handle_unknown_text(message: types.Message):
    # You can choose to ignore, or reply with a "command not understood" message
    await message.reply(
        "🤔 Tôi không hiểu yêu cầu của bạn. Vui lòng sử dụng các nút bấm bên dưới hoặc gửi mã MD5 hợp lệ.",
        reply_markup=get_user_keyboard()
    )


if __name__ == "__main__":
    print("Bot đang chạy...")
    try:
        executor.start_polling(dp, skip_updates=True)
    except Exception as e:
        print(f"Bot dừng do lỗi: {e}")
