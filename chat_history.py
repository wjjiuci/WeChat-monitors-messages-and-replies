from wxauto import WeChat
from _ctypes import COMError
import time
import os

try:
    from config import CHAT_NAME
except ImportError:
    raise SystemExit("缺少配置文件：请先复制 config.example.py 为 config.py 并填写后重试")

def msg_line(message):
    """把一条消息转成保存用的文本行"""
    if hasattr(message, 'sender') and hasattr(message, 'content'):
        return f"{message.sender}:{message.content}"
    if hasattr(message, 'time') and not hasattr(message, 'sender'):
        return f"[Time]:{message.time}"
    return f"[Other]:{getattr(message, 'content', str(message))}"


def msg_key(message):
    """消息去重键：优先用消息id，否则用类型+属性组合"""
    mid = getattr(message, 'id', None)
    if mid is not None:
        return ('id', mid)
    return ('tuple', type(message).__name__, msg_line(message))


def get_chat_history(contact_name):
    wx = WeChat()
    wx.ChatWith(contact_name)
    time.sleep(3)

    # 微信聊天窗口只保留已加载的消息，用 LoadMoreMessage 反复加载更早的历史，
    # 边加载边收集新出现的消息，直到连续几轮都没有新消息（到顶）
    seen = set()
    collected = []  # 全量聊天记录行，按时间正序
    stable_rounds = 0
    for i in range(300):
        try:
            wx.LoadMoreMessage()
        except COMError:
            time.sleep(1)
        time.sleep(0.5)
        try:
            messages = wx.GetAllMessage()
        except COMError:
            continue
        new_lines = []
        for m in messages:
            k = msg_key(m)
            if k not in seen:
                seen.add(k)
                new_lines.append(msg_line(m))
        if new_lines:
            collected = new_lines + collected  # 新加载的都是更早的消息，插到前面
            stable_rounds = 0
        else:
            stable_rounds += 1
            if stable_rounds >= 3:  # 连续3轮无新消息，认为已到顶
                break
        print(f"\r滚动加载中：已收集 {len(collected)} 条消息", end="", flush=True)
    print()

    path = os.getcwd()
    file_path = os.path.join(path, f"{contact_name}所有聊天记录.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in collected:
            f.write(item + '\n')

    return file_path, len(collected)

if __name__ == "__main__":
    contact_name = CHAT_NAME
    file_path, count = get_chat_history(contact_name)
    print(f"共收集 {count} 条消息，已保存至:", file_path)


















