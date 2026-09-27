# ============================================================
# 配置文件模板
# 使用方法：复制本文件为 config.py，然后填入你的真实信息
# config.py 已被 .gitignore 排除，不会被提交到 GitHub
# ============================================================

# ---------- 微信 ----------
SELF_NICKNAME = "你的微信昵称"      # 本机登录微信的昵称（运行日志中显示）
TARGET_CONTACT = "对方备注或群名"    # 要监听并自动回复的联系人 / 群聊
CHAT_NAME = "对方备注或群名"         # chat_history.py 抓取聊天记录的对象（通常与上面相同）

# ---------- 情感模型 ----------
MODEL_DIR_NAME = "my_finetuned_model"  # train.py 的训练产物目录名，也是 WeChat_reply.py 加载的目录名

# ---------- 人设与关系 ----------
PERSONA = (
    "你是某某，一个说话随性的年轻人。"
    "平时聊天简短口语化，偶尔带点方言和玩笑。"
    "回复要像真人发微信：简短、口语化，一般一两句话，不要长篇大论，不要解释你是AI。"
)
RELATIONSHIP = "朋友"             # 可选：恋人/暧昧对象/朋友/损友/长辈/家人/同事/客户，也可自定义
RELATIONSHIP_CONTEXT = ""         # 可选补充近况，如"最近在组队打游戏"，留空不启用

# ---------- API Keys ----------
JEV_API_KEY = ""                  # JEV 决策模型 Key（TypeSafe 后台申请）；留空则自动降级为本地 BERT 情感模型
DEEPSEEK_API_KEY = ""             # DeepSeek API Key（https://platform.deepseek.com 创建）
