from wxauto import WeChat
import time
import os

def get_chat_history(contact_name):
    wx = WeChat()
    wx.ChatWith(contact_name)
    time.sleep(2)

    all_msg = []
    messages = wx.GetAllMessage()

    for message in messages:
        # 判断是否是文本消息（有 sender 和 content 属性）
        if hasattr(message, 'sender') and hasattr(message, 'content'):
            all_msg.append(f"{message.sender}:{message.content}")
        # 判断是否是时间消息（有 time 属性，且没有 sender）
        elif hasattr(message, 'time') and not hasattr(message, 'sender'):
            all_msg.append(f"[Time]:{message.time}")
        # 其他类型：尝试提取可用信息
        else:
            # 尝试获取 content 或转为字符串
            content = getattr(message, 'content', str(message))
            all_msg.append(f"[Other]:{content}")

    path = os.getcwd()
    file_path = os.path.join(path, f"{contact_name}所有聊天记录.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in all_msg:
            f.write(item + '\n')

    return file_path

if __name__ == "__main__":
    contact_name = ""
    file_path = get_chat_history(contact_name)
    print("聊天记录已保存至:", file_path)


















