import time
import pyautogui

print("Script bắt đầu sau 3 giây. Bấm Ctrl+C trong terminal này để dừng...")
time.sleep(3)

try:
    while True:
        pyautogui.press('enter')
        time.sleep(2)  # Nhấn Enter mỗi 2 giây
except KeyboardInterrupt:
    print("\nĐã dừng script.")