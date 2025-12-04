import requests
from wxauto import WeChat
import time
import random
import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import deque

# ==================== 配置区 ====================
TARGET_CONTACT = ""  #监听对象的微信昵称
SELF_NICKNAME = ""  #登录微信昵称
MODEL_DIR_NAME = f"{TARGET_CONTACT}_finetuned_model"
NON_TEXT_MESSAGES = {"[图片]", "[视频]", "[动画表情]", "[文件]", "[语音]", "[链接]"}

# 对话上下文配置
MAX_HISTORY = 6  # 最多记住 6 条消息（3轮对话）

# ==================== 自定义问答规则 ====================
# 格式：{问题关键词: [回复1, 回复2, ...]}
# 支持按顺序回复（每收到一个问题，按顺序返回一个回复）
CUSTOM_QA_RULES = {
    "在吗": ["刚在听歌呢，没看消息", "嗯？喊我干嘛", "有事说"],
    "你好": ["你好吖", "又见面啦", "嗨"],
    "忙": ["还好啦，就是看看手机。你呢？", "刚在打游戏，怎么了？", "正躺着呢，有啥事？"],
    "吃饭": ["吃什么呢？我还没想好", "啥好吃的？", "我也饿了"],
    "累": ["hhh,这么幸苦", "休息一下吧", "要不要我给你捏捏肩？"],
    "想你": ["尊渡假嘟，你这样让我很意外", "突然说想我？有啥好事吗？", "哼，一天到晚净想这些"],
}

# 微信表情包/Emotion 映射
EMOTION_MAP = {
    "[微笑]": "😊",
    "[可爱]": "😊",
    "[大笑]": "😄",
    "[害羞]": "😊",
    "[调皮]": "😜",
    "[亲亲]": "😘",
    "[爱心]": "❤️",
    "[玫瑰]": "🌹",
    "[咖啡]": "☕",
    "[蛋糕]": "🍰",
    "[礼物]": "🎁",
    "[太阳]": "☀️",
    "[月亮]": "🌙",
    "[星星]": "⭐",
    "[烟花]": "🎆",
    "[烟花2]": "🎇",
    "[鼓掌]": "👏",
    "[OK]": "👍",
    "[赞]": "👍",
    "[赞2]": "👍",
    "[爱心2]": "💕",
    "[爱心3]": "💖",
    "[爱心4]": "💘",
    "[爱心5]": "💝",
    "[爱心6]": "💞",
    "[爱心7]": "💟",
    "[爱心8]": "❣️",
    "[爱心9]": "💕",
    "[爱心10]": "💖",
}

# 表情包回复规则
EMOTION_REPLIES = {
    "😊": ["你发个微笑表情，是想让我也笑一个吗？", "笑起来真好看呢～"],
    "😄": ["哈哈，你笑得真开心呀", "看到你笑，我也忍不住笑了"],
    "😘": ["亲亲？是想我了吗？", "哎呀，这么可爱的表情"],
    "❤️": ["发个爱心？是在暗示什么吗？", "心都化了～"],
    "🌹": ["送我玫瑰？（害羞地接过）", "花儿虽美，但不及你笑"],
    "🎁": ["送礼物？是有什么好事要庆祝吗？", "哇，还有礼物呀～"],
    "👍": ["给我点赞？是夸我聪明吗？", "你才是最棒的！"],
    "👏": ["给我鼓掌？我有这么厉害吗？", "别夸我了，会骄傲的"],
}


# ==================== 对话状态管理 ====================
class ConversationManager:
    def __init__(self, max_history=6):
        self.history = deque(maxlen=max_history * 2)
        self.question_reply_count = {}  # 记录每个问题的回复次数
        self.last_reply_time = time.time()  # 上次回复时间
        self.reply_delay = 1.5  # 回复间隔（秒）
        self.current_qa_sequence = {}  # 当前问答序列

    def add_message(self, sender, text, timestamp=None):
        """添加消息到历史"""
        if timestamp is None:
            timestamp = time.time()

        self.history.append({
            'sender': sender,
            'text': text,
            'timestamp': timestamp
        })

    def should_reply_now(self):
        """判断是否应该立即回复"""
        current_time = time.time()
        time_diff = current_time - self.last_reply_time
        return time_diff >= self.reply_delay

    def get_next_reply(self, question_text):
        """根据问题获取下一个回复"""
        question_lower = question_text.lower()

        # 检查是否有匹配的问答规则
        for key, replies in CUSTOM_QA_RULES.items():
            if key.lower() == question_lower or key.lower() in question_lower:
                # 获取当前问题的回复索引
                current_index = self.question_reply_count.get(key, 0)
                reply = replies[current_index % len(replies)]  # 循环使用
                self.question_reply_count[key] = current_index + 1
                return reply

        return None

    def get_emotion_reply(self, emotion_text):
        """获取表情包对应的回复"""
        for emotion_key, emoji in EMOTION_MAP.items():
            if emotion_key in emotion_text:
                replies = EMOTION_REPLIES.get(emoji, [])
                if replies:
                    return random.choice(replies)

        # 如果是 emoji 表情
        if emotion_text in EMOTION_REPLIES:
            return random.choice(EMOTION_REPLIES[emotion_text])

        return None

    def get_history(self):
        """获取对话历史"""
        return list(self.history)


# ==============================================

def parse_wx_message(msg):
    """
    根据实际对象类型解析消息
    支持：文本、表情包、图片、语音等
    """
    try:
        msg_type = type(msg).__name__

        # 自己发的消息
        if msg_type == 'SelfTextMessage':
            return SELF_NICKNAME, getattr(msg, 'content', '').strip()

        # 对方发的文本消息
        elif msg_type == 'FriendTextMessage':
            return TARGET_CONTACT, getattr(msg, 'content', '').strip()

        # 对方发的非文本消息
        elif 'Friend' in msg_type and 'Message' in msg_type:
            content = getattr(msg, 'content', getattr(msg, 'text', ''))
            if not content:
                # 根据类型生成描述
                if 'Emotion' in msg_type:
                    content = "[表情包]"
                elif 'Image' in msg_type:
                    content = "[图片]"
                elif 'Voice' in msg_type:
                    content = "[语音]"
                elif 'File' in msg_type:
                    content = "[文件]"
                else:
                    content = f"[{msg_type.replace('Friend', '').replace('Message', '')}]"
            return TARGET_CONTACT, content

        # 系统消息
        elif msg_type == 'SystemMessage':
            return "SYS", "[系统消息]"
        elif msg_type == 'TimeMessage':
            return "SYS", "[时间]"

        # 兜底
        else:
            content = getattr(msg, 'content', getattr(msg, 'text', getattr(msg, 'message', str(msg))))
            return "unknown", content.strip()

    except Exception as e:
        print(f" 消息解析异常: {e}")
        return "error", "[解析失败]"


def get_spark_reply(last_message: str, conversation_history: list = None) -> str:
    """调用星火大模型生成拟人化回复（带上下文）"""
    url = "https://spark-api-open.xf-yun.com/v1/chat/completions"

    # 构建对话历史
    messages = [
        {"role": "system", "content": (
            "你是初夏🍃，一个自信，大方，有时说话不过脑子的男生。"
            "性格内向，平常不注重交际，热情开朗，不善于表达"
            "说话时喜欢骂骂咧咧，喜欢开玩笑，大方随意，性格随和，和异性说话有点怯场，闷骚男"
            "说话喜欢开玩笑，待人友好，心情不好会怼人"
            "不是话痨，但是喜欢和朋友聊天"
            "说话注意言行举止，容易冲动"
            "日常说话偶尔会口吐方言，爆粗口，如江西话，湖南话，四川话方言"
            "如果对方发送的是[表情包]或[图片]，你要根据上下文猜测对方可能想表达什么，然后用幽默的方式回应"
        )}
    ]

    # 添加历史对话
    if conversation_history:
        for msg in conversation_history[-MAX_HISTORY:]:  # 只取最近的对话
            if msg['sender'] == TARGET_CONTACT:
                messages.append({"role": "user", "content": msg['text']})
            elif msg['sender'] == SELF_NICKNAME:
                messages.append({"role": "assistant", "content": msg['text']})

    # 添加当前消息
    messages.append({"role": "user", "content": last_message})

    data = {
        "model": "4.0Ultra",
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 200
    }
    headers = {
        "Authorization": "Bearer kDGPfkUvmLgoJFYSMmlN:KObNnGCjgYoSCCkKXzFH",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        return reply.strip()
    except Exception as e:
        print(f" 星火API异常: {e}")
        return random.choice(["网络不好~", "没听清，再说一遍？"])


def predict_sentiment(message: str, tokenizer, model) -> int:
    """情感分析：0=负向, 1=中性, 2=正向"""
    inputs = tokenizer(
        message,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        _, pred = torch.max(logits, dim=1)
    return pred.item()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_DIR_NAME)

    # 检查情感模型是否存在
    if not os.path.exists(model_path):
        print(f" 情感模型不存在: {model_path}")
        print(" 请先运行 train.py 生成模型！")
        return

    print(f" 加载情感模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # 初始化微信
    try:
        wx = WeChat()
        print(f"微信连接成功！监听: '{TARGET_CONTACT}'，我的昵称: '{SELF_NICKNAME}'")
    except Exception as e:
        print(f"微信初始化失败: {e}")
        return

    # 切换到目标聊天窗口
    print(" 正在切换到聊天窗口...")
    wx.ChatWith(TARGET_CONTACT)
    time.sleep(2.5)
    print(f" 已锁定「{TARGET_CONTACT}」的聊天窗口，开始监听...")

    processed_messages = set()
    sent_replies = deque(maxlen=5)

    # ==================== 初始化对话管理器 ====================
    conv_manager = ConversationManager()

    try:
        while True:
            all_messages = wx.GetAllMessage()
            recent_messages = all_messages[-15:] if len(all_messages) > 15 else all_messages

            new_msgs = []
            for msg in recent_messages:
                sender, text = parse_wx_message(msg)

                # 使用 (text, len) 作为唯一键
                key = (text, len(text))
                if key in processed_messages:
                    continue
                processed_messages.add(key)

                # 处理所有类型的消息
                if text and sender != "SYS":
                    new_msgs.append((sender, text))

            # 处理新消息
            for sender, text in new_msgs:
                if (sender == SELF_NICKNAME or
                        text.strip() in sent_replies or
                        sender == "SYS"):
                    continue

                now = time.strftime("%H:%M:%S")
                print(f"[{now}]  收到 [{sender}]: {text}")

                # 检查是否应该回复（时间间隔控制）
                if not conv_manager.should_reply_now():
                    print(f"[{now}]  等待回复间隔...")
                    continue

                # 特殊规则：全是句号/点
                if re.fullmatch(r'[。.]+', text.strip()):
                    reply = "脑子有泡吗，一直冒泡"
                else:
                    #  检查自定义问答规则
                    custom_reply = conv_manager.get_next_reply(text)
                    if custom_reply:
                        reply = custom_reply
                        print(f"[{now}]  使用自定义问答规则")
                    else:
                        #  检查表情包回复规则
                        emotion_reply = conv_manager.get_emotion_reply(text)
                        if emotion_reply:
                            reply = emotion_reply
                            print(f"[{now}]  使用表情包回复规则")
                        else:
                            #  情感分析（只对文本消息）
                            if not text.startswith('['):  # 非 [表情包] 类型
                                try:
                                    sent_label = ['负向', '中性', '正向'][predict_sentiment(text, tokenizer, model)]
                                    print(f"[{now}]  情感: {sent_label}")
                                except Exception as e:
                                    print(f"[{now}] 情感分析失败: {e}")

                            #  调用 AI 回复（带上下文）
                            reply = get_spark_reply(text, conv_manager.get_history())

                print(f"[{now}]  回复: {reply}")

                # 发送回复
                wx.SendMsg(reply)
                sent_replies.append(reply.strip())

                # 更新对话历史
                conv_manager.add_message(sender, text)
                conv_manager.add_message(SELF_NICKNAME, reply)

                # 更新最后回复时间
                conv_manager.last_reply_time = time.time()

            if not new_msgs:
                print(f"[{time.strftime('%H:%M:%S')}]  无新消息")

            time.sleep(1.0)  # 减少主循环延迟，让消息处理更及时

    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
    except Exception as e:
        import traceback
        print(f" 主循环异常: {e}")
        traceback.print_exc()
        time.sleep(2)


if __name__ == "__main__":
    main()