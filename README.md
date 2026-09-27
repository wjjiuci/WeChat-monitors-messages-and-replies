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

```bash
# 1. 安装依赖（requirements.txt 已包含本地 wxauto-main 源码安装）
pip install -r requirements.txt

# 2. 配置
copy config.example.py config.py
# 编辑 config.py：填微信昵称、监听对象、API Keys、人设

# 3. 抓取聊天记录（保持微信登录，先在微信里把历史消息往上翻加载出来）
python chat_history.py

# 4. 训练情感模型（用抓到的记录自动打标微调）
python train.py

# 5. 启动自动回复
python WeChat_reply.py
```

## 配置说明

| 配置项 | 说明 |
|---|---|
| `SELF_NICKNAME` | 本机登录微信的昵称 |
| `TARGET_CONTACT` / `CHAT_NAME` | 监听对象 / 抓取记录对象（联系人备注或群名） |
| `MODEL_DIR_NAME` | 训练产物模型目录名 |
| `PERSONA` | 人设提示词（说话风格、语气、方言习惯等） |
| `RELATIONSHIP` | 与对方的关系，决定注入哪套沟通知识 |
| `JEV_API_KEY` | JEV 决策模型 Key，留空则用本地 BERT |
| `DEEPSEEK_API_KEY` | DeepSeek 回复模型 Key |

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

## 免责声明

- 本项目仅供学习交流，请勿用于违法或骚扰用途
- 自动回复存在误判风险，重要对话请谨慎开启
- UI 自动化操作微信可能违反微信使用条款，账号风险自负
- 聊天记录和训练模型包含个人隐私，已被 .gitignore 排除，请勿手动提交

## 致谢

- [言外 yanwai](https://github.com/YIRC99/yanwai) — JevProtocol 决策协议与事件解读卡片设计
- [wxauto](https://github.com/cluic/wxauto) — PC 微信 UI 自动化
- [goutoujunshi 狗头军师](https://github.com/shengjidaguai-china/goutoujunshi) — 按关系组织沟通知识的思路
