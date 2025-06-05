import asyncio
import json
import os
import re
from functools import wraps
from aiogram import Bot, Dispatcher, types, executor  # Đã sửa import executor ở đây
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from keep_alive import keep_alive
keep_alive()

# !!! QUAN TRỌNG: Hãy thay thế TOKEN và ADMIN_USERNAME bằng thông tin thực tế của bạn !!!
TOKEN = "8076967422:AAFfQfggn_PdDQx5uBtUnAZO-PlsNo9eNnI" # ĐÂY LÀ TOKEN VÍ DỤ, HÃY THAY THẾ
ADMIN_USERNAME = "truongdong1920" # Thay thế bằng username admin của bạn
XU_COST = 1

HISTORY_FILE = "lich_su.json"
ADMINS_FILE = "admins.json"
USER_XU_FILE = "user_xu.json"

# Initialize data structures
history = {}
admins = []
user_xu = {}

# --- Data Loading and Saving Functions ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                loaded_history = json.load(f)
                if isinstance(loaded_history, dict):
                    return loaded_history
                print(f"Warning: {HISTORY_FILE} content is not a dictionary. Starting with empty history.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {HISTORY_FILE}. Starting with empty history.")
        except Exception as e:
            print(f"Warning: Error loading {HISTORY_FILE}: {e}. Starting with empty history.")
    return {}

def save_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving history: {e}")

def load_admins():
    if os.path.exists(ADMINS_FILE):
        try:
            with open(ADMINS_FILE, "r", encoding="utf-8") as f:
                loaded_admins = json.load(f)
                if isinstance(loaded_admins, list) and all(isinstance(item, int) for item in loaded_admins):
                    return loaded_admins
                print(f"Warning: {ADMINS_FILE} content is not a list of integers. Reverting to default.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {ADMINS_FILE}. Reverting to default.")
        except Exception as e:
            print(f"Warning: Error loading {ADMINS_FILE}: {e}. Reverting to default.")
    
    default_admins = [6381480476]  # Default admin ID
    try:
        with open(ADMINS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_admins, f, indent=2)
    except Exception as e:
        print(f"Error creating default admins file: {e}")
    return default_admins

def save_admins(admin_list):
    try:
        with open(ADMINS_FILE, "w", encoding="utf-8") as f:
            json.dump(admin_list, f, indent=2)
    except Exception as e:
        print(f"Error saving admins: {e}")

def load_user_xu():
    if os.path.exists(USER_XU_FILE):
        try:
            with open(USER_XU_FILE, "r", encoding="utf-8") as f:
                loaded_xu = json.load(f)
                if isinstance(loaded_xu, dict) and all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in loaded_xu.items()):
                    return loaded_xu
                print(f"Warning: {USER_XU_FILE} content is not in the expected format. Starting with empty user_xu.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {USER_XU_FILE}. Starting with empty user_xu.")
        except Exception as e:
            print(f"Warning: Error loading {USER_XU_FILE}: {e}. Starting with empty user_xu.")
    return {}

def save_user_xu():
    try:
        with open(USER_XU_FILE, "w", encoding="utf-8") as f:
            json.dump(user_xu, f, indent=2)
    except Exception as e:
        print(f"Error saving user_xu: {e}")

# Load initial data
history = load_history()
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
                [KeyboardButton(text="Gửi thông báo"), KeyboardButton(text="Danh sách người dùng")], # Thêm nút Gửi thông báo
                [KeyboardButton(text="Liên hệ admin")]
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

def analyze_md5(md5_hash):
    md5_hash = md5_hash.lower()
    if md5_hash in history:
        cached_data = history[md5_hash]
        # Expecting history to store [result, ratio_tai, ratio_xiu]
        if isinstance(cached_data, list) and len(cached_data) == 3:
            try:
                return cached_data[0], float(cached_data[1]), float(cached_data[2])
            except ValueError:
                # Invalid data in cache, proceed to recalculate
                pass

    digit_values = [int(char, 16) for char in md5_hash if char.isdigit()]
    letter_values = [ord(char) for char in md5_hash if char.isalpha()]

    if not digit_values or not letter_values:
        result, ratio_tai, ratio_xiu = "Xỉu", 0.0, 100.0
        history[md5_hash] = [result, ratio_tai, ratio_xiu]
        save_history()
        return result, ratio_tai, ratio_xiu

    digit_sum = sum(digit_values)
    letter_sum = sum(letter_values)

    xor_digit = 0
    for num in digit_values:
        xor_digit ^= num

    xor_letter = 0
    for char_val in letter_values:
        xor_letter ^= char_val

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
    ratio_tai = float(final_score)
    ratio_xiu = 100.0 - ratio_tai
    
    history[md5_hash] = [result, ratio_tai, ratio_xiu]
    save_history()
    return result, ratio_tai, ratio_xiu

# Command Handlers
@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    user_id = message.from_user.id
    username = message.from_user.username or "Không có"
    current_xu = user_xu.get(str(user_id), 0)
    
    # Add user to user_xu if not exists, for /broadcast and user list purposes
    if str(user_id) not in user_xu:
        user_xu[str(user_id)] = 0
        save_user_xu()

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
    await message.reply("Vui lòng gửi mã MD5 (32 ký tự) để tôi phân tích.")

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
        "Nhập theo định dạng: `ID_XU`\nVí dụ: `123456789_100` (cấp 100 Xu cho user ID 123456789)",
        parse_mode="Markdown",
        reply_markup=get_user_keyboard(message.from_user.id)
    )

# This handler processes replies to "Cấp xu" prompt
@dp.message_handler(lambda message: message.reply_to_message and 
                                      message.reply_to_message.text and 
                                      message.reply_to_message.text.startswith("Nhập theo định dạng: `ID_XU`") and
                                      re.fullmatch(r"\d+_\d+", message.text or ""),
                   content_types=types.ContentType.TEXT)
@admin_only
async def cap_xu_process(message: types.Message):
    try:
        user_id_str, xu_str = message.text.split("_")
        target_user_id = int(user_id_str)
        xu_cung_cap = int(xu_str)

        current_xu = user_xu.get(str(target_user_id), 0)
        user_xu[str(target_user_id)] = current_xu + xu_cung_cap
        save_user_xu()
        
        await message.reply(
            f"✅ Đã cấp {xu_cung_cap} xu cho user ID {target_user_id}. Số dư mới: {user_xu[str(target_user_id)]}",
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

# This handler processes replies for adding/removing admins
@dp.message_handler(lambda message: message.text and message.text.isdigit() and message.reply_to_message and 
                   message.reply_to_message.text and
                   (message.reply_to_message.text.startswith("Nhập ID người dùng muốn thêm làm Admin:") or 
                    message.reply_to_message.text.startswith("Nhập ID người dùng muốn xóa khỏi Admin:"))
                   ,content_types=types.ContentType.TEXT)
@admin_only
async def admin_management_handler(message: types.Message):
    try:
        admin_id_to_manage = int(message.text)
        action_prompt = message.reply_to_message.text

        if action_prompt.startswith("Nhập ID người dùng muốn thêm"):
            if admin_id_to_manage in admins:
                await message.reply(f"⚠️ ID {admin_id_to_manage} đã là admin rồi.", reply_markup=get_user_keyboard(message.from_user.id))
            else:
                admins.append(admin_id_to_manage)
                save_admins(admins)
                await message.reply(f"✅ Đã thêm ID {admin_id_to_manage} vào danh sách admin.", reply_markup=get_user_keyboard(message.from_user.id))
        elif action_prompt.startswith("Nhập ID người dùng muốn xóa"):
            if admin_id_to_manage not in admins:
                await message.reply(f"⚠️ ID {admin_id_to_manage} không phải admin.", reply_markup=get_user_keyboard(message.from_user.id))
            elif admin_id_to_manage == 6906617636:  # Default admin ID, can be made more dynamic if needed
                await message.reply("❌ Không thể xóa admin mặc định này.", reply_markup=get_user_keyboard(message.from_user.id))
            else:
                admins.remove(admin_id_to_manage)
                save_admins(admins)
                await message.reply(f"✅ Đã xóa ID {admin_id_to_manage} khỏi danh sách admin.", reply_markup=get_user_keyboard(message.from_user.id))
    except ValueError:
        await message.reply("ID không hợp lệ. Vui lòng nhập một số.", reply_markup=get_user_keyboard(message.from_user.id))
    except Exception as e:
        await message.reply(f"Đã xảy ra lỗi: {e}", reply_markup=get_user_keyboard(message.from_user.id))


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
        "Nhập nội dung thông báo bạn muốn gửi đến tất cả người dùng (trả lời tin nhắn này):",
        # reply_markup=ReplyKeyboardRemove() # Giữ lại keyboard để admin dễ thao tác
        reply_markup=get_user_keyboard(message.from_user.id)
    )

# Handles both command /broadcast and reply to prompt
@dp.message_handler(
    lambda message: message.reply_to_message and
                   message.reply_to_message.from_user.is_bot and # Check if replying to bot
                   message.reply_to_message.text and
                   message.reply_to_message.text.startswith("Nhập nội dung thông báo bạn muốn gửi"),
    content_types=types.ContentType.TEXT
)
@dp.message_handler(commands=["broadcast"])
@admin_only
async def broadcast_message_handler(message: types.Message):
    content = ""
    if message.get_command() == "/broadcast":
        content = message.get_args()
    elif message.reply_to_message: # Must be a reply to the prompt
        content = message.text

    if not content:
        await message.reply(
            "Vui lòng nhập nội dung thông báo.\n"
            "Cách 1: Dùng lệnh `/broadcast Nội dung tin nhắn`\n"
            "Cách 2: Bấm nút 'Gửi thông báo' và trả lời tin nhắn của bot với nội dung.",
            parse_mode="Markdown",
            reply_markup=get_user_keyboard(message.from_user.id)
        )
        return

    success = 0
    failed = 0
    
    all_user_ids = list(user_xu.keys()) # Create a list to iterate over, as user_xu might change if new users interact
    
    for user_id_str in all_user_ids:
        try:
            await bot.send_message(int(user_id_str), f"📢 Thông báo từ Admin:\n\n{content}")
            success += 1
            await asyncio.sleep(0.1)  # Tránh rate limit (0.1s là khá nhanh, có thể tăng nếu cần)
        except Exception as e:
            failed += 1
            print(f"Lỗi gửi tin nhắn user {user_id_str}: {e}")
    
    await message.reply(
        f"✅ Đã gửi thông báo đến {success} người dùng.\n❌ Không gửi được đến {failed} người.",
        reply_markup=get_user_keyboard(message.from_user.id)
    )

@dp.message_handler(lambda message: message.text == "Danh sách người dùng")
@admin_only
async def user_list_handler(message: types.Message):
    if not user_xu:
        await message.reply("Chưa có người dùng nào sử dụng bot.", reply_markup=get_user_keyboard(message.from_user.id))
        return
    
    user_list_parts = ["📋 Danh sách người dùng và số xu:\n"]
    for user_id, xu_count in user_xu.items():
        user_list_parts.append(f"• ID: {user_id} - Xu: {xu_count}")
    
    full_user_list = "\n".join(user_list_parts)

    # Telegram has a message length limit (4096 chars)
    if len(full_user_list) > 4000:
        await message.reply("Danh sách người dùng quá dài để hiển thị. Cân nhắc xuất ra file hoặc chia nhỏ.", reply_markup=get_user_keyboard(message.from_user.id))
        # For very long lists, you might want to send it as a file
        # with open("user_list.txt", "w", encoding="utf-8") as f:
        #     f.write(full_user_list)
        # await message.reply_document(open("user_list.txt", "rb"))
        # os.remove("user_list.txt")
    else:
        await message.reply(
            full_user_list,
            reply_markup=get_user_keyboard(message.from_user.id)
        )

# This handler processes MD5 strings
@dp.message_handler(regexp=r"^[a-fA-F0-9]{32}$") # Allow uppercase hex
async def md5_analyze_handler(message: types.Message):
    md5_hash_input = message.text.lower() # Convert to lowercase for consistency
    user_id_str = str(message.from_user.id)
    
    if user_id_str not in user_xu: # Should be added by /start, but as a fallback
        user_xu[user_id_str] = 0
        # No save_user_xu() here, will be saved after deduction or if they top up
    
    if user_xu.get(user_id_str, 0) < XU_COST:
        await message.reply(
            f"❌ Bạn không đủ {XU_COST} xu để phân tích MD5.\n"
            f"Số xu hiện tại: {user_xu.get(user_id_str, 0)}.\n"
            "Vui lòng liên hệ admin để mua thêm xu.",
            reply_markup=get_user_keyboard(message.from_user.id)
        )
        return
    
    # Deduct xu before analysis
    user_xu[user_id_str] -= XU_COST
    save_user_xu()
    
    await message.reply("🔍 Đang phân tích mã MD5, vui lòng chờ giây lát...", reply_markup=get_user_keyboard(message.from_user.id)) # Show keyboard immediately

    try:
        result, ratio_tai, ratio_xiu = analyze_md5(md5_hash_input)
        
        reply_text = (
            f"🎰 *PHÂN TÍCH MD5 SIÊU CHUẨN* 🔮✨🌌🎰\n"
            f"📌 *MD5*: `{md5_hash_input}`\n"
            f"💥 *Tài*: {ratio_tai:.2f}% [ {'🟥' * int(ratio_tai/10)}{'⬜️' * (10 - int(ratio_tai/10))} ]\n"
            f"💦 *Xỉu*: {ratio_xiu:.2f}% [ {'🟦' * int(ratio_xiu/10)}{'⬜️' * (10 - int(ratio_xiu/10))} ]\n"
            f"💰 *XU CÒN LẠI*: {user_xu[user_id_str]}"
        )
        
        await message.reply(
            reply_text,
            parse_mode="Markdown",
            reply_markup=get_user_keyboard(message.from_user.id)
        )
    except Exception as e:
        # Refund xu if analysis fails unexpectedly
        user_xu[user_id_str] += XU_COST
        save_user_xu()
        await message.reply(
            f"Rất tiếc, đã có lỗi xảy ra trong quá trình phân tích: {e}\nXu của bạn đã được hoàn lại.",
            reply_markup=get_user_keyboard(message.from_user.id)
        )
        print(f"Error during MD5 analysis for {md5_hash_input}: {e}")


# Fallback handler for text messages not caught by other handlers
@dp.message_handler(content_types=types.ContentType.TEXT)
async def unknown_text_handler(message: types.Message):
    # This can happen if admin replies to "Cấp xu" or "Thêm admin" prompts with non-matching text
    if message.reply_to_message and message.reply_to_message.from_user.is_bot:
        if message.reply_to_message.text:
            if message.reply_to_message.text.startswith("Nhập theo định dạng: `ID_XU`") or \
               message.reply_to_message.text.startswith("Nhập ID người dùng muốn thêm làm Admin:") or \
               message.reply_to_message.text.startswith("Nhập ID người dùng muốn xóa khỏi Admin:"):
                await message.reply("Định dạng không hợp lệ. Vui lòng thử lại theo hướng dẫn.", reply_markup=get_user_keyboard(message.from_user.id))
                return
    
    await message.reply(
        "Lệnh không xác định hoặc định dạng không đúng. Vui lòng sử dụng các nút trên bàn phím.",
        reply_markup=get_user_keyboard(message.from_user.id)
    )


if __name__ == "__main__":
    print("Bot đang chạy...")
    executor.start_polling(dp, skip_updates=True) # skip_updates=True is good for development
