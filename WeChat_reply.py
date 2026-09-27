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
try:
    from config import (
        SELF_NICKNAME, TARGET_CONTACT, MODEL_DIR_NAME,
        PERSONA, RELATIONSHIP, RELATIONSHIP_CONTEXT,
        JEV_API_KEY, DEEPSEEK_API_KEY,
    )
except ImportError:
    raise SystemExit("缺少配置文件：请先复制 config.example.py 为 config.py 并填写后重试")

NON_TEXT_MESSAGES = {"[图片]", "[视频]", "[动画表情]", "[文件]", "[语音]", "[链接]"}

# 按关系组织的沟通知识库（只注入当前关系的那一条，避免 prompt 臃肿）
RELATIONSHIP_KNOWLEDGE = {
    "恋人": (
        "对方是你的恋人。要点：先接住情绪再解决问题；多用昵称和亲昵语气；"
        "对方低落时先共情陪伴，别急着讲道理；可以有适当的占有欲和小撒娇。"
    ),
    "暧昧对象": (
        "你们在暧昧期。要点：保持一点神秘感和推拉，别太主动也别太冷；"
        "多制造轻松有趣的话题和下次见面的由头；对方分享生活时表现出兴趣，追问细节。"
    ),
    "朋友": (
        "对方是普通朋友。要点：随和自然，有来有回；可以开玩笑但有分寸；"
        "对方求助时给出实在的建议。"
    ),
    "损友": (
        "对方是你的损友/兄弟群。要点：互损互怼是日常，可以毒舌、开玩笑、爆粗口，但有底线不戳真痛处；"
        "对方真遇到事（负向情绪）时收起玩笑，先站他这边；群里回复要短，接得住梗。"
    ),
    "长辈": (
        "对方是长辈。要点：尊重礼貌，多用敬语；回复可以正式一些；"
        "多关心对方身体和生活，别发表情包敷衍。"
    ),
    "家人": (
        "对方是家人。要点：语气亲近随意，不用客套；多报平安、多关心；"
        "被唠叨时别顶撞，顺着应下来。"
    ),
    "同事": (
        "对方是同事。要点：友好但保持职业边界；工作的事给明确答复；"
        "闲聊轻松但避开敏感话题和过度吐槽公司。"
    ),
    "客户": (
        "对方是客户。要点：专业、及时、可靠；先确认对方需求再答复；"
        "不轻易承诺做不到的事，价格问题不主动让步。"
    ),
}

# 先接住情绪，再回应内容（根据本地情感模型结果引导语气）
EMOTION_GUIDE = {
    0: "对方这句话情绪偏负向。先接住他的情绪（认同、陪伴或站他这边），再回应事情本身，别讲大道理。",
    1: "对方这句话情绪中性。自然随意地接话即可，可以适当抖个机灵。",
    2: "对方这句话情绪正向。顺着他的开心聊，回应可以更热情、多捧场。",
}

# ==================== JEV 决策模型配置 ====================
# JEV_API_KEY / DEEPSEEK_API_KEY 均在 config.py 中配置
JEV_MODEL = "typesafe/jev-1.13"
JEV_MIN_REPLY_PROB = 0.6  # "适合自动回复"概率低于此值则跳过（全自动安全阀）
JEV_TIMEOUT = 15  # 一次请求并行 ~15 个问题，稍留余量

# JEV 情绪标签 → 本地情感极性（用于复用 EMOTION_GUIDE）
JEV_EMOTION_TO_SENTIMENT = {
    "开心": 2, "兴奋": 2, "平静": 1, "调侃": 1,
    "阴阳怪气": 0, "生气": 0, "难过": 0, "疲惫": 0,
}


def jev_decide(text: str, history: list = None):
    """
    调用 JEV 决策模型，一次请求并行判断：情绪、意图、是否适合自动回复。
    返回结构化答案 dict；未配置 Key 或调用失败返回 None（调用方降级本地模型）。
    """
    if not JEV_API_KEY:
        return None
    state = {"对方最新消息": text, "我与对方的关系": RELATIONSHIP}
    if history:
        state["最近对话"] = [f"{m['sender']}: {m['text']}" for m in history[-MAX_HISTORY:]]
    questions = {
        "emotion": {
            "type": "choice",
            "instructions": "这条消息发送者当前的情绪最接近哪一类？",
            "criteria": {
                "开心": "明显高兴、兴奋、分享好消息",
                "平静": "日常陈述，没有明显情绪波动",
                "调侃": "开玩笑、互损、抛梗，不带恶意",
                "阴阳怪气": "表面客气实际带刺、反讽",
                "生气": "愤怒、指责、不耐烦",
                "难过": "低落、委屈、受挫、累",
            },
        },
        "intent": {
            "type": "choice",
            "instructions": "发送者这条消息的意图最接近哪一类？",
            "criteria": {
                "提问": "在问问题、征求意见",
                "分享": "分享日常、心情、看到的东西",
                "吐槽": "抱怨发泄，想被认同而不是要解决方案",
                "邀约": "约着一起做事、打游戏、见面",
                "互损": "聊天互怼玩闹",
                "求助": "请求帮忙或建议",
            },
        },
        "should_reply": {
            "type": "noul",
            "instructions": "这是一个全自动微信回复助手。现在自动回复这条消息是否合适？",
            "criteria": {
                "true": "消息在抛话题、打招呼、问问题或期待回应，此时接话自然不突兀",
                "false": "消息明显在和群里别人说话、双方正在激烈争吵、或明确表示不想聊，自动回复会突兀或激化矛盾",
            },
        },
        # 以下两个维度借鉴 yanwai(言外) 的 JevProtocol：对话阶段 + 下一步动作
        "progress": {
            "type": "choice",
            "instructions": "当前这一步的对话在等待怎样的回应？只依据前文判断。",
            "criteria": {
                "sharing": "对方在分享经历或自然闲聊",
                "clarify": "对方在等具体事实或细节（时间、地点、名字、答案）",
                "reassure": "对方在等关心、重视或安抚的回应",
                "explain": "需要澄清误会或承认问题",
                "act": "已经解释过了，等具体行动落实",
                "accepted": "已明确接受安排，等执行或收尾",
                "closing": "明确告别或话题自然收尾",
            },
        },
        "action": {
            "type": "choice",
            "instructions": "结合上下文，我这一条回复最适合采取哪种动作？优先回应当前未回应的信息。",
            "criteria": {
                "answer_first": "对方在问问题或等确认：直接回答/拍板，给出具体信息，别绕弯",
                "catch_emotion": "对方在倾诉、抱怨或发泄：先接住情绪（认同、站TA这边），再回应内容，别讲道理",
                "confirm_plan": "对话正在敲定安排：给明确的时间/地点/选择，别含糊",
                "show_care": "对方在等关心：先表达在意，追问细节，让对方感到被重视",
                "play_along": "对方在开玩笑、互损、抛梗：顺着接梗适度反击，语气轻松",
                "return_topic": "话题快聊完了：简短回应后主动抛一个相关的新话题把球打回去",
                "none": "普通日常往来：简短自然回应即可，不用刻意延展",
            },
        },
    }
    # 事件解读卡问题（借鉴言外 detailPayload：每张卡独立核对，不适合就选 unclear）
    for card in EVENT_CARDS:
        questions[f"reading_{card['id']}"] = {
            "type": "choice",
            "instructions": (
                f"只在此问题适合当前语境时判断，否则选 unclear。{card['question']}"
                "signal 和 ordinary 是平等的备选解释，不因为某个更戏剧化就选择它。"
            ),
            "criteria": {
                "signal": card["signal"],
                "ordinary": card["ordinary"],
                "unclear": "信息不足或此问题不适用，不能确定",
            },
        }
    try:
        r = requests.post(
            "https://api.typesafe.ai/v1/systemone",
            headers={"Authorization": f"Bearer {JEV_API_KEY}", "Content-Type": "application/json"},
            json={"model": "jev-1.13.0", "state": state, "questions": questions},
            timeout=JEV_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("answers")
    except Exception as e:
        print(f" JEV决策异常: {e}")
        return None

# 对话上下文配置
MAX_HISTORY = 6  # 最多记住 6 条消息（3轮对话）

# JEV 决策 → 实质性回复建议（本地映射，借鉴言外的 action 候选机制）
PROGRESS_LABELS = {
    "sharing": "闲聊分享", "clarify": "等具体信息", "reassure": "等关心回应",
    "explain": "等澄清", "act": "等行动落实", "accepted": "已敲定等执行", "closing": "收尾告别",
}

ADVICE_MAP = {
    "answer_first": "直接回答对方问的事，给出具体信息（时间/地点/名字/选择），哪怕一句话也要答到位，结尾可以自然带个小问题",
    "catch_emotion": "先接住情绪再谈事：认同TA的感受、站TA这边，别讲道理、别急着给解决方案，等情绪落地再说事",
    "confirm_plan": "给明确答复：时间、地点、选哪个直接拍板，别用'都行''随便'糊弄",
    "show_care": "先表达在意：心疼一下或追问细节，让对方感到被重视，再自然延展",
    "play_along": "顺着梗接住并适度反击，可以损回去，保持轻松别当真",
    "return_topic": "简短回应当前话题后，主动抛一个和上下文相关的具体新话题（具体到事，别问'在干嘛'）",
    "none": "简短自然回应即可，不刻意延展",
}

def build_advice(decision: dict) -> str:
    """把 JEV 的结构化决策映射成一句实质性回复建议"""
    if not decision:
        return None
    action = decision.get("action", {}).get("choice", "none")
    emotion = decision.get("emotion", {}).get("choice", "平静")
    progress = decision.get("progress", {}).get("choice", "sharing")
    base = ADVICE_MAP.get(action, ADVICE_MAP["none"])
    # 情绪修饰：负面情绪语气放软，调侃可以皮
    if emotion in ("难过", "生气", "疲惫"):
        base += "；TA情绪不好，语气放软，别皮"
    elif emotion == "调侃" and action != "play_along":
        base += "；对方在开玩笑，可以带点玩笑感回"
    elif emotion == "阴阳怪气":
        base += "；对方话里有刺，别接错了当真话回"
    return base


# 事件解读卡片（照搬言外 ChatTemplates：signal/ordinary 是平等的备选解释）
EVENT_CARDS = [
    {"id": "invite_probe", "scene": "邀约安排", "question": "是在试探一起活动的意愿吗？",
     "signal": "在试探能否一起去", "ordinary": "只是分享活动信息"},
    {"id": "invite_plan", "scene": "邀约安排", "question": "现在是否在等时间地点？",
     "signal": "已有一起去的意向，等具体安排", "ordinary": "还没确认愿不愿意去"},
    {"id": "daily_vent", "scene": "日常分享", "question": "这会儿更需要倾听还是办法？",
     "signal": "想先吐槽、被理解", "ordinary": "明确在问解决办法"},
    {"id": "daily_share", "scene": "日常分享", "question": "这次分享更想得到什么？",
     "signal": "想让你参与这段经历", "ordinary": "只是告知一件事"},
    {"id": "care_checkin", "scene": "关心靠近", "question": "是在认真关心你的状态吗？",
     "signal": "根据具体近况主动关心", "ordinary": "普通寒暄或顺口问候"},
    {"id": "care_reassure", "scene": "关心靠近", "question": "是在确认自己有没有被重视吗？",
     "signal": "对被忽略表达担心，期待确认", "ordinary": "仅询问一个事实"},
    {"id": "tease_play", "scene": "玩笑试探", "question": "这句调侃是在轻松互动吗？",
     "signal": "双方语境支持友好的玩笑", "ordinary": "包含认真不满或不舒服"},
    {"id": "tease_probe", "scene": "玩笑试探", "question": "是在借玩笑试探你的态度吗？",
     "signal": "围绕明确话题试探你的态度", "ordinary": "单纯逗趣，没有足够试探线索"},
    {"id": "promise_followup", "scene": "约定记忆", "question": "是在追问之前约定的进展吗？",
     "signal": "希望知道约定是否落实", "ordinary": "只是回忆过去的事情"},
    {"id": "friction_hear", "scene": "委屈不满", "question": "这句不满更需要先被理解吗？",
     "signal": "在表达受伤或被忽略的感受", "ordinary": "主要在指出具体事实问题"},
]

def extract_focus(decision: dict):
    """从 reading_* 答案里挑最贴合的事件卡（choice=signal 且概率最高）"""
    if not decision:
        return None
    best = None
    for card in EVENT_CARDS:
        ans = decision.get(f"reading_{card['id']}")
        if not ans:
            continue
        probs = ans.get("probabilities", {})
        p_sig = probs.get("signal", 0.0)
        if ans.get("choice") == "signal" and p_sig >= 0.5 and (best is None or p_sig > best["p_signal"]):
            best = {**card, "p_signal": p_sig, "p_ordinary": probs.get("ordinary", 0.0)}
    return best


# ==================== 对话状态管理 ====================
class ConversationManager:
    def __init__(self, max_history=6):
        self.history = deque(maxlen=max_history * 2)
        self.last_reply_time = time.time()  # 上次回复时间
        self.reply_delay = 1.5  # 回复间隔（秒）

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

    def get_history(self):
        """获取对话历史"""
        return list(self.history)


# ==================== 消息解析 ====================

def parse_wx_message(msg):
    """
    把 wxauto 返回的单条消息对象规整成 (发送者, 文本内容) 二元组，交给主循环处理。
    归类规则：
    - 本人发的文本 → 发送者记为 SELF_NICKNAME
    - 对方发的文本 → 发送者记为 TARGET_CONTACT
    - 对方的表情/图片/语音/文件等非文本内容 → 转成 [表情包] 这类占位符
    - 时间戳、系统提示 → 统一标记为 SYS，主循环据此直接过滤
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


def build_system_prompt(sentiment: int = None, decision: dict = None) -> str:
    """组合系统提示词：人设 + 关系沟通知识 + 情绪/意图引导（JEV决策优先，本地BERT兜底）"""
    parts = [PERSONA]
    # 关系知识（找不到就用通用关系描述）
    knowledge = RELATIONSHIP_KNOWLEDGE.get(RELATIONSHIP, f"对方和你的关系是：{RELATIONSHIP}。")
    parts.append(knowledge)
    if RELATIONSHIP_CONTEXT:
        parts.append(f"补充背景：{RELATIONSHIP_CONTEXT}")
    # 情绪与意图引导（狗头军师思路：先接住人，再解决事）
    if decision:
        emo = decision.get("emotion", {})
        intent = decision.get("intent", {})
        probs = emo.get("probabilities", {})
        prob_str = "、".join(f"{k}{v:.0%}" for k, v in probs.items() if v > 0.05)
        progress = decision.get("progress", {}).get("choice", "sharing")
        progress_label = PROGRESS_LABELS.get(progress, progress)
        advice = build_advice(decision)
        parts.append(
            f"【军师决策】情绪「{emo.get('choice', '平静')}」（{prob_str}），"
            f"意图「{intent.get('choice', '分享')}」，对话阶段：{progress_label}。"
            f"回复策略：{advice}"
        )
        focus = extract_focus(decision)
        if focus:
            parts.append(
                f"【事件解读】{focus['question']} → {focus['signal']}（{focus['p_signal']:.0%}）。"
                "回复要贴合这个解读，不要答偏。"
            )
        mapped = JEV_EMOTION_TO_SENTIMENT.get(emo.get("choice"), 1)
        parts.append(EMOTION_GUIDE[mapped])
    elif sentiment in EMOTION_GUIDE:
        parts.append(EMOTION_GUIDE[sentiment])
    parts.append("如果对方发送的是[表情包]或[图片]，根据上下文猜测对方想表达什么，用幽默的方式回应。")
    return "\n".join(parts)


def get_ai_reply(last_message: str, conversation_history: list = None, sentiment: int = None, decision: dict = None) -> str:
    """调用 DeepSeek 大模型生成拟人化回复（带上下文、关系知识与情绪/意图引导）"""
    url = "https://api.deepseek.com/v1/chat/completions"

    # 构建对话历史
    messages = [{"role": "system", "content": build_system_prompt(sentiment, decision)}]

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
        "model": "deepseek-chat",  # DeepSeek-V3，日常聊天足够；要更强推理可改 deepseek-reasoner
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 200
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        return reply.strip()
    except Exception as e:
        print(f" DeepSeek API异常: {e}")
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
        consecutive_errors = 0
        while True:
            try:
                all_messages = wx.GetAllMessage()
                consecutive_errors = 0
            except Exception as e:
                # UIA 偶发断连（窗口刷新/焦点变化），不退出，重新锁定后继续
                consecutive_errors += 1
                print(f"[{time.strftime('%H:%M:%S')}]  读取消息异常({consecutive_errors}): {e}")
                time.sleep(2)
                if consecutive_errors % 5 == 0:
                    try:
                        wx.ChatWith(TARGET_CONTACT)
                        time.sleep(2)
                        print(f" 已重新锁定「{TARGET_CONTACT}」窗口")
                    except Exception as e2:
                        print(f" 重锁窗口失败: {e2}")
                continue
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
                    #  决策层：JEV 优先（情绪+意图+是否回复），无Key/失败降级本地BERT
                    sentiment = None
                    decision = None
                    decision = jev_decide(text, conv_manager.get_history())
                    if decision:
                        emo = decision.get("emotion", {}).get("choice", "未知")
                        emo_probs = decision.get("emotion", {}).get("probabilities", {})
                        emo_str = "、".join(f"{k}{v:.0%}" for k, v in emo_probs.items() if v > 0.05) or emo
                        intent = decision.get("intent", {}).get("choice", "未知")
                        progress = PROGRESS_LABELS.get(decision.get("progress", {}).get("choice", ""), "?")
                        p_reply = decision.get("should_reply", {}).get("noul", 1.0)
                        advice = build_advice(decision)
                        print(f"[{now}]  JEV: 情绪={emo}({emo_str}) 意图={intent} 阶段={progress} 适合回复={p_reply:.0%}")
                        focus = extract_focus(decision)
                        if focus:
                            print(f"[{now}]  事件：{focus['scene']}")
                            print(f"[{now}]  {focus['question']}")
                            print(f"[{now}]  · {focus['signal']}：{focus['p_signal']:.0%}")
                            print(f"[{now}]  · {focus['ordinary']}：{focus['p_ordinary']:.0%}")
                        print(f"[{now}]  └ 建议：{advice}")
                        # 全自动安全阀：JEV 判断不适合回复时跳过（但仍记入历史）
                        if p_reply < JEV_MIN_REPLY_PROB:
                            print(f"[{now}]  JEV门控：不适合自动回复，跳过")
                            conv_manager.add_message(sender, text)
                            continue
                    else:
                        # 降级：本地 BERT 情感分析
                        try:
                            sentiment = predict_sentiment(text, tokenizer, model)
                            sent_label = ['负向', '中性', '正向'][sentiment]
                            print(f"[{now}]  情感(BERT): {sent_label}")
                        except Exception as e:
                            print(f"[{now}] 情感分析失败: {e}")

                    #  调用 AI 回复（带上下文 + 关系知识 + 情绪/意图引导）
                    reply = get_ai_reply(text, conv_manager.get_history(), sentiment, decision)

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