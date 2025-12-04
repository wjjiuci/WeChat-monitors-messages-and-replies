# wxauto/WeChat.py (精简可用版)
import os
import time
import uiautomation as auto
from dataclasses import dataclass
from typing import List, Union, Tuple

@dataclass
class SelfTextMessage:
    content: str

@dataclass
class FriendTextMessage:
    name: str
    message: str

@dataclass
class TimeMessage:
    time: str

class WeChat:
    def __init__(self):
        self.weixin_window = auto.WindowControl(searchDepth=1, ClassName='WeChatMainWndForPC')
        if not self.weixin_window.Exists(2):
            raise Exception("未找到微信窗口，请确保微信已登录并打开！")
        self.weixin_window.SetActive()

    def ChatWith(self, who: str):
        edit = self.weixin_window.EditControl(searchDepth=1)
        edit.SendKeys('{Ctrl}f')
        search_box = self.weixin_window.EditControl(Name='搜索')
        search_box.SendKeys(who, waitTime=0.5)
        search_box.SendKeys('{Enter}', waitTime=1)

    def GetAllMessage(self) -> List[Union[SelfTextMessage, FriendTextMessage, TimeMessage]]:
        messages = []
        chat_window = self.weixin_window.PaneControl(searchDepth=1, foundIndex=7)
        if not chat_window.Exists():
            return messages

        items = chat_window.GetChildren()
        for item in items:
            try:
                # 判断是否为时间消息
                if item.ControlTypeName == 'TextControl' and ':' in item.Name and len(item.Name) <= 8:
                    messages.append(TimeMessage(time=item.Name))
                    continue

                # 获取气泡
                bubbles = item.GetChildren()
                for bubble in bubbles:
                    if bubble.ControlTypeName != 'TextControl':
                        continue
                    text = bubble.Name.strip()
                    if not text:
                        continue

                    # 判断左右位置（左：对方，右：自己）
                    rect = bubble.BoundingRectangle
                    center_x = (rect.left + rect.right) / 2
                    window_rect = self.weixin_window.BoundingRectangle
                    is_self = center_x > (window_rect.left + window_rect.width() * 0.6)

                    if is_self:
                        messages.append(SelfTextMessage(content=text))
                    else:
                        # 尝试获取发送者名字（简化：用 TARGET_CONTACT 代替）
                        # 实际中可从上一个“名字标签”获取，此处简化
                        messages.append(FriendTextMessage(name="枫", message=text))
            except:
                continue
        return messages

    def SendMsg(self, msg: str):
        edit = self.weixin_window.EditControl(searchDepth=1)
        edit.SendKeys(msg, waitTime=0.5)
        edit.SendKeys('{Enter}', waitTime=0.5)