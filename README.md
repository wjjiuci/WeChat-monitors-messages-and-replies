# WeChat-AI-Reply 🤖💬

基于 PC 微信的**全流程 AI 自动回复系统**：抓取聊天记录 → 微调情感模型 → JEV 五维决策（情绪/意图/阶段/动作/门控 + 事件解读卡片）→ 关系化 Prompt → DeepSeek 生成拟人回复。

> 决策协议借鉴开源项目 [言外 yanwai](https://github.com/YIRC99/yanwai) 的 JevProtocol，将其"事件解读卡片"机制移植到 PC 端。

## 功能特性

- **聊天记录抓取**：基于 wxauto 自动滚动收集微信群/好友历史消息，保存为纯文本
- **情感模型微调**：SnowNLP 自动打标 + bert-base-chinese 三分类（正/中/负），无需人工标注
- **JEV 智能决策**（可选）：单次 API 请求并行判断 15 个问题
  - 情绪（开心/平静/调侃/阴阳怪气/难过/生气/疲惫 + 概率分布）
  - 意图（提问/邀约/吐槽/求助/互损/确认）
  - 对话阶段（等具体信息/等关心/等澄清/等行动/已敲定/收尾）
  - 回复门控——识别"在跟别人说话/不适合插嘴"的消息，自动跳过
  - **事件解读卡片**：邀约安排/日常分享/关心靠近/玩笑试探/约定记忆/委屈不满 10 张卡，输出"事件 + 具体问题 + 概率解读"，格式与言外插件的分析卡片一致
- **关系化 Prompt**：内置恋人/暧昧对象/朋友/损友/长辈/家人/同事/客户 8 套沟通知识库，只注入当前关系，一行配置切换
- **防穿帮设计**：非文本消息（表情/图片/语音/文件）与系统消息自动过滤，不回复红包/转账
- **优雅降级**：不配 JEV Key 时自动降级本地 BERT 情感模型，程序不会崩

## 工作流程

```
新消息 → parse_wx_message 解析（过滤系统消息/非文本）
      → JEV 决策（情绪+意图+阶段+动作+门控+10张事件卡）
      │    ├─ 门控 < 60% → 跳过，不插嘴
      │    └─ 失败 → 降级本地 BERT 三分类
      → build_system_prompt（人设 + 关系知识 + 情绪引导 + 军师决策 + 事件解读）
      → DeepSeek 生成回复 → wxauto 发送
```

## 环境要求

- Windows 10/11
- **微信 3.9.x** PC 客户端（实测 3.9.12.1000；**不支持 4.x**——4.x 客户端进程为 Weixin.exe，需配合 wxauto4 自行改造）
  - 官网 https://pc.weixin.qq.com 默认推送 4.x，微信 3.9 历史版本的下载方式见 [wxauto-main/README.md](wxauto-main/README.md)
- Python 3.10+
- 有 NVIDIA 显卡训练更快（纯 CPU 也能跑，只是慢）

## 快速开始

### 1. 克隆仓库并安装依赖

```bash
git clone https://github.com/wjjiuci/WeChat-monitors-messages-and-replies.git
cd WeChat-monitors-messages-and-replies

# 安装依赖（requirements.txt 已包含本地 wxauto-main 源码安装）
pip install -r requirements.txt
```

### 2. 获取 API 密钥

#### DeepSeek API Key（用于生成回复，**必须**）

1. 打开 [DeepSeek 开放平台](https://platform.deepseek.com/)，点击「登录」，用手机号或邮箱注册账号
2. 登录后直接进入 API Keys 管理页：[platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)
3. 点击「创建 API Key」，输入名称（如 `wechat-ai`），点击确认
4. **立即复制**弹出的 Key（格式如 `sk-xxxxxxxx`），页面关闭后无法再次查看，只能重新创建
5. 新注册账号通常有赠送余额，用完后在「充值」页面按需充值（API 调用按 tokens 计费，价格见[官方定价页](https://api-docs.deepseek.com/quick_start/pricing)）

#### JEV API Key（用于智能决策，**可选但强烈建议**）

JEV 是 TypeSafe AI 的 System One 决策模型（代码里调用的端点是 `https://api.typesafe.ai/v1/systemone`），目前为**邀请制**：

1. 打开 TypeSafe 控制台 [console.typesafe.ai](https://console.typesafe.ai/)
2. 注册账号并登录；如提示未开通，需加入 waitlist 等待邀请（也可参考官方文档 [docs.typesafe.ai](https://docs.typesafe.ai/introduction/quickstart)，或通过官方 Discord [discord.gg/typesafe](https://discord.gg/typesafe) 咨询开通进度）
3. 账号开通后，在控制台创建 API Key 并复制保存
4. 填入 `config.py` 的 `JEV_API_KEY`
5. **没拿到 Key 也能用**：留空即可，程序会自动降级为本地 BERT 情感模型，只是决策精度和事件卡片解读会弱于 JEV

### 3. 配置文件

```bash
# 复制配置模板
copy config.example.py config.py

# 用编辑器打开 config.py，逐项填写：
#   SELF_NICKNAME    → 本机登录微信的昵称（微信设置里能看到）
#   TARGET_CONTACT   → 要自动回复的联系人备注名或群名（必须完全匹配）
#   CHAT_NAME        → 要抓取聊天记录的联系人或群名
#   DEEPSEEK_API_KEY → 上一步复制的 DeepSeek Key
#   JEV_API_KEY      → 上一步复制的 JEV Key（可选，留空则降级本地模型）
#   RELATIONSHIP     → 从恋人/暧昧对象/朋友/损友/长辈/家人/同事/客户中选一个
#   RELATIONSHIP_CONTEXT → 可选，补充近况背景（如"最近在组队打游戏"），留空不启用
#   PERSONA          → 你想让 AI 扮演的人设（说话风格、语气、口头禅等）
```

### 4. 抓取聊天记录（训练用）

1. **保持微信 PC 客户端已登录**
2. 在微信里找到目标聊天窗口，**手动往上翻**加载历史消息（翻得越多，训练效果越好，建议至少几百条）
3. 运行抓取脚本：

```bash
python chat_history.py
```

4. 收集完成后，记录保存在**项目根目录**下的 `{CHAT_NAME}所有聊天记录.txt`（train.py 按此文件名读取，请勿改名）

### 5. 训练情感模型

```bash
python train.py
```

- 脚本自动读取上一步生成的 `{CHAT_NAME}所有聊天记录.txt`，用 SnowNLP 打情感标签
- 然后基于 `bert-base-chinese` 微调一个三分类模型（正/中/负）
- **首次运行会自动下载 BERT 预训练模型**（约 400MB，脚本已内置 hf-mirror 国内镜像加速，无需代理）
- 训练好的模型保存在 `MODEL_DIR_NAME` 指定的目录，训练 checkpoints 在 `results/` 目录
- 有 NVIDIA 显卡会快很多，纯 CPU 也能跑（无显卡用户如嫌 PyTorch 太大，可先装 CPU 版：`pip install torch --index-url https://download.pytorch.org/whl/cpu`）

### 6. 启动自动回复

```bash
python WeChat_reply.py
```

- 启动后会自动切换到 `TARGET_CONTACT` 的聊天窗口并开始监听
- 收到新消息时，先走 JEV 决策（或本地 BERT 降级），再调用 DeepSeek 生成回复，最后通过 wxauto 自动发送
- **运行期间请保持微信窗口可见**，不要最小化或遮挡，也尽量不要手动操作鼠标键盘（UI 自动化依赖窗口可见性；偶发断连程序会自动重新锁定窗口）
- **按 `Ctrl + C` 可安全退出**

## 配置说明

| 配置项 | 说明 | 获取方式 |
|---|---|---|
| `SELF_NICKNAME` | 本机登录微信的昵称 | 微信客户端 → 左下角头像/设置 |
| `TARGET_CONTACT` | 自动回复的监听对象（联系人备注或群名） | 必须与微信里的显示名称**完全一致** |
| `CHAT_NAME` | 抓取聊天记录的对象（联系人备注或群名） | 可与 `TARGET_CONTACT` 相同或不同 |
| `MODEL_DIR_NAME` | 训练产物模型目录名 | 保持默认即可，会自动创建 |
| `PERSONA` | 人设提示词（说话风格、语气、方言习惯等） | 自由填写，越详细角色越鲜活 |
| `RELATIONSHIP` | 与对方的关系 | 从 `恋人/暧昧对象/朋友/损友/长辈/家人/同事/客户` 中选一个，也可自定义 |
| `RELATIONSHIP_CONTEXT` | 补充近况背景（可选） | 如"最近在组队打游戏"，留空不启用 |
| `JEV_API_KEY` | JEV 决策模型 Key | [TypeSafe 控制台](https://console.typesafe.ai/)（邀请制）；**留空则降级本地 BERT** |
| `DEEPSEEK_API_KEY` | DeepSeek 回复模型 Key | [DeepSeek 开放平台](https://platform.deepseek.com/api_keys) 注册后创建；**必须填写** |

## 项目结构

```
├── chat_history.py      # 抓取聊天记录（wxauto 滚动收集，自动重试）
├── train.py             # SnowNLP 打标 + BERT 微调三分类情感模型
├── WeChat_reply.py      # 主程序：监听 → JEV 决策 → DeepSeek 回复
├── wxauto-main/         # 随仓库附带的 wxauto 39.2.1 源码（适配微信 3.9.x，含 3.9 下载指引）
├── config.example.py    # 配置模板（复制为 config.py 填写）
├── requirements.txt
└── .gitignore           # 已排除 config.py / 聊天记录 / 模型等隐私文件
```

## 常见问题

- **报错「缺少配置文件」**：还没把 `config.example.py` 复制为 `config.py`，见快速开始第 3 步
- **微信 4.x 能用吗**：不能。4.x 进程是 Weixin.exe，需用微信 3.9.x（下载方式见 [wxauto-main/README.md](wxauto-main/README.md)），或自行迁移到 wxauto4
- **抓取不到消息 / 监听没反应**：确认微信窗口可见且未最小化；确认 `TARGET_CONTACT` / `CHAT_NAME` 与微信里显示的名称（备注）完全一致
- **train.py 提示聊天记录不存在**：确认先运行了 `chat_history.py`，且 `config.py` 里的 `CHAT_NAME` 与抓取时一致（文件名前缀必须匹配）
- **DeepSeek 想用更强推理模型**：编辑 `WeChat_reply.py`，把 `deepseek-chat` 改为 `deepseek-reasoner`（注意推理模型延迟和费用更高）

## 免责声明

- 本项目仅供学习交流，请勿用于违法或骚扰用途
- 自动回复存在误判风险，重要对话请谨慎开启
- UI 自动化操作微信可能违反微信使用条款，账号风险自负
- 聊天记录和训练模型包含个人隐私，已被 .gitignore 排除，请勿手动提交

## 致谢

- [言外 yanwai](https://github.com/YIRC99/yanwai) — JevProtocol 决策协议与事件解读卡片设计
- [wxauto](https://github.com/cluic/wxauto) — PC 微信 UI 自动化
- [goutoujunshi 狗头军师](https://github.com/shengjidaguai-china/goutoujunshi) — 按关系组织沟通知识的思路
