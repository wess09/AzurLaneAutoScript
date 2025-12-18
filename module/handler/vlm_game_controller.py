# VLM 自主游戏控制器 - 专注于错误处理场景
# 使用视觉语言模型 (Vision-Language Model) 智能分析游戏卡死/错误界面
# 并生成、执行恢复操作，避免简单粗暴的重启策略

import base64
import io
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PIL import Image

from module.base.utils import get_bbox
from module.exception import RequestHumanTakeover
from module.logger import logger


class VisionAnalyzer:
    """
    视觉分析器：调用 VLM API 理解游戏界面状态
    """

    def __init__(self, config):
        """
        初始化视觉分析器
        
        Args:
            config: AzurLaneConfig 配置对象
        """
        self.config = config
        self.provider = config.VLMGameController_Provider  # openai, claude, gemini
        self.model = config.VLMGameController_Model
        self.api_key = config.VLMGameController_APIKey
        self.base_url = config.VLMGameController_BaseURL or None
        self.timeout = config.VLMGameController_Timeout

    def analyze_game_state(self, screenshot: Image.Image, context: dict) -> Optional[dict]:
        """
        分析当前游戏状态
        
        Args:
            screenshot: 游戏截图 (PIL Image)
            context: 上下文信息
                {
                    "task": "当前任务名称",
                    "error": "错误信息",
                    "expected_scene": "预期界面"  # 可选
                }
        
        Returns:
            分析结果字典，格式：
            {
                "scene": "当前场景类型",
                "description": "界面描述",
                "elements": [
                    {"type": "button", "text": "按钮文本", "position": [x, y]}
                ],
                "stuck_reason": "卡死原因",
                "suggested_actions": ["操作建议1", "操作建议2"],
                "confidence": 0.0-1.0
            }
            
            如果 VLM 调用失败，返回 None
        """
        try:
            # 将截图转换为 base64 编码
            image_base64 = self._encode_image(screenshot)

            # 构建提示词
            prompt = self._build_prompt(context)

            # 根据提供商调用不同的 API
            if self.provider == "openai":
                result = self._call_openai(prompt, image_base64)
            elif self.provider == "claude":
                result = self._call_claude(prompt, image_base64)
            elif self.provider == "gemini":
                result = self._call_gemini(prompt, image_base64)
            else:
                logger.error(f"不支持的 VLM 提供商: {self.provider}")
                return None

            # 解析 JSON 结果
            analysis = self._parse_response(result)
            return analysis

        except Exception as e:
            logger.error(f"VLM 分析失败: {e}")
            return None

    def _encode_image(self, image: Image.Image) -> str:
        """
        将 PIL Image 编码为 base64 字符串
        
        Args:
            image: PIL Image 对象
            
        Returns:
            base64 编码的图像字符串
        """
        buffered = io.BytesIO()
        # 转换为 RGB 模式（如果是 RGBA）
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def _build_prompt(self, context: dict) -> str:
        """
        构建发送给 VLM 的提示词
        
        Args:
            context: 上下文信息
            
        Returns:
            完整的提示词字符串
        """
        task = context.get("task", "未知")
        error = context.get("error", "")
        expected = context.get("expected_scene", "")

        prompt = f"""你是 Azur Lane (碧蓝航线) 游戏的视觉助手。当前游戏遇到问题，需要你分析截图并给出解决方案。

**上下文信息**：
- 当前执行任务: {task}
- 错误信息: {error}
- 预期界面: {expected or "自动判断"}

**你的任务**：
1. 识别当前游戏界面类型（主界面/战斗中/弹窗/错误提示/网络重连/加载中等）
2. 提取所有可见的可交互元素（按钮、选项卡等）及其大致位置
3. 判断游戏是否卡死，如果是，分析卡死原因
4. 给出具体的恢复操作建议

**常见场景参考**：
- 网络错误弹窗：通常有"重试"或"确定"按钮
- 系统公告/活动弹窗：有"关闭"或"我知道了"按钮
- 资源不足提示：如"燃料不足"、"金币不足"
- 战斗结算界面：有"确认"按钮
- 加载界面：通常有Loading动画，需要等待
- 意外界面：需要找到"返回"或"主页"按钮

**截图分辨率**：假设为 1280x720（坐标相对于此分辨率）

**返回格式**（必须是有效的 JSON）：
```json
{{
  "scene": "场景类型（如network_error_dialog、announcement_popup、main_menu等）",
  "description": "简短的界面描述",
  "elements": [
    {{"type": "button", "text": "按钮上的文字", "position": [x, y]}}
  ],
  "stuck_reason": "如果卡死，说明原因；否则为null",
  "suggested_actions": ["具体操作建议1", "具体操作建议2"],
  "confidence": 0.95
}}
```

请仔细观察截图，直接返回 JSON 格式结果，不要有其他内容。"""

        return prompt

    def _call_openai(self, prompt: str, image_base64: str) -> str:
        """
        调用 OpenAI Vision API
        
        Args:
            prompt: 提示词
            image_base64: base64 编码的图像
            
        Returns:
            API 返回的文本内容
        """
        try:
            import openai
        except ImportError:
            logger.error("请先安装 openai 库: pip install openai")
            raise

        # 配置客户端
        client_params = {
            "api_key": self.api_key,
            "timeout": self.timeout
        }
        if self.base_url:
            client_params["base_url"] = self.base_url

        client = openai.OpenAI(**client_params)

        # 调用 API
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500,
            temperature=0.1  # 降低随机性，提高稳定性
        )

        return response.choices[0].message.content

    def _call_claude(self, prompt: str, image_base64: str) -> str:
        """
        调用 Anthropic Claude Vision API
        
        Args:
            prompt: 提示词
            image_base64: base64 编码的图像
            
        Returns:
            API 返回的文本内容
        """
        try:
            import anthropic
        except ImportError:
            logger.error("请先安装 anthropic 库: pip install anthropic")
            raise

        client = anthropic.Anthropic(api_key=self.api_key)

        message = client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
            temperature=0.1
        )

        return message.content[0].text

    def _call_gemini(self, prompt: str, image_base64: str) -> str:
        """
        调用 Google Gemini Vision API
        
        Args:
            prompt: 提示词
            image_base64: base64 编码的图像
            
        Returns:
            API 返回的文本内容
        """
        try:
            import google.generativeai as genai
        except ImportError:
            logger.error("请先安装 google-generativeai 库: pip install google-generativeai")
            raise

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)

        # 准备图像
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))

        # 生成内容
        response = model.generate_content(
            [prompt, image],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1500
            )
        )

        return response.text

    def _parse_response(self, response_text: str) -> dict:
        """
        解析 VLM 返回的 JSON 响应
        
        Args:
            response_text: VLM 返回的原始文本
            
        Returns:
            解析后的字典
        """
        # 尝试提取 JSON（有时 VLM 会返回 ```json...``` 格式）
        text = response_text.strip()
        
        # 移除可能的 markdown 代码块标记
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # 解析 JSON
        try:
            result = json.loads(text)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"VLM 返回的内容不是有效的 JSON: {text[:200]}")
            raise


class ActionPlanner:
    """
    动作规划器：将 VLM 分析结果转换为可执行的操作序列
    """

    def __init__(self, config):
        self.config = config

    def plan_recovery_actions(self, analysis: dict, device_resolution: Tuple[int, int]) -> List[dict]:
        """
        根据 VLM 分析结果生成操作序列
        
        Args:
            analysis: VLM 分析结果
            device_resolution: 设备实际分辨率 (width, height)
            
        Returns:
            操作列表，每个操作为一个字典：
            [
                {"action": "click", "position": [x, y], "description": "点击XX按钮"},
                {"action": "wait", "seconds": 3, "description": "等待界面加载"},
            ]
        """
        actions = []
        
        # 如果没有建议操作，返回空列表
        if not analysis.get('suggested_actions'):
            logger.warning("VLM 未给出操作建议")
            return actions

        # 解析建议操作
        for suggestion in analysis['suggested_actions']:
            suggestion_lower = suggestion.lower()
            
            # 1. 点击类操作
            if '点击' in suggestion or 'click' in suggestion_lower:
                # 提取需要点击的按钮名称
                button_text = self._extract_button_text(suggestion)
                
                # 在 elements 中查找对应按钮
                position = self._find_button_position(button_text, analysis.get('elements', []))
                
                if position:
                    # 坐标转换：从 1280x720 转换到实际分辨率
                    scaled_pos = self._scale_position(position, device_resolution)
                    actions.append({
                        "action": "click",
                        "position": scaled_pos,
                        "description": f"点击 {button_text} 按钮"
                    })
                else:
                    logger.warning(f"未找到按钮 '{button_text}' 的位置")
            
            # 2. 等待类操作
            elif '等待' in suggestion or 'wait' in suggestion_lower:
                # 提取等待时间（秒）
                wait_time = self._extract_wait_time(suggestion)
                actions.append({
                    "action": "wait",
                    "seconds": wait_time,
                    "description": f"等待 {wait_time} 秒"
                })
            
            # 3. 返回/后退操作
            elif '返回' in suggestion or '后退' in suggestion or 'back' in suggestion_lower:
                # 通常返回按钮在左上角
                actions.append({
                    "action": "click",
                    "position": self._scale_position([60, 60], device_resolution),
                    "description": "点击返回按钮（左上角）"
                })

        # 如果没有生成任何操作，尝试通用恢复策略
        if not actions:
            actions = self._generate_fallback_actions(analysis, device_resolution)

        return actions

    def _extract_button_text(self, suggestion: str) -> str:
        """
        从建议中提取按钮文本
        例如："点击重试按钮" -> "重试"
        """
        # 移除常见的修饰词
        text = suggestion.replace('点击', '').replace('按钮', '').replace('选项', '')
        text = text.replace('click', '').replace('button', '').strip()
        
        # 提取引号中的内容
        if '"' in text:
            parts = text.split('"')
            if len(parts) >= 2:
                return parts[1].strip()
        if '"' in text:
            parts = text.split('"')
            if len(parts) >= 2:
                return parts[1].strip()
        
        return text.strip()

    def _find_button_position(self, button_text: str, elements: List[dict]) -> Optional[List[int]]:
        """
        在元素列表中查找指定按钮的位置
        
        Args:
            button_text: 按钮文本
            elements: VLM 识别的元素列表
            
        Returns:
            位置坐标 [x, y]，如果未找到返回 None
        """
        button_text_lower = button_text.lower()
        
        for element in elements:
            if element.get('type') == 'button':
                elem_text = element.get('text', '').lower()
                
                # 模糊匹配
                if button_text_lower in elem_text or elem_text in button_text_lower:
                    return element.get('position')
                
                # 检查常见的同义词
                synonyms = {
                    '重试': ['retry', '再试', '重新'],
                    '确定': ['ok', 'confirm', '确认'],
                    '取消': ['cancel', '关闭', 'close'],
                    '返回': ['back', '后退'],
                }
                
                for key, values in synonyms.items():
                    if key in button_text_lower:
                        for synonym in values:
                            if synonym in elem_text:
                                return element.get('position')
        
        return None

    def _extract_wait_time(self, suggestion: str) -> int:
        """
        从建议中提取等待时间（秒）
        例如："等待3秒" -> 3
        """
        import re
        match = re.search(r'(\d+)\s*秒', suggestion)
        if match:
            return int(match.group(1))
        
        # 默认等待时间
        return 5

    def _scale_position(self, position: List[int], device_resolution: Tuple[int, int]) -> List[int]:
        """
        将坐标从 1280x720 缩放到实际设备分辨率
        
        Args:
            position: 原始位置 [x, y]（基于 1280x720）
            device_resolution: 实际分辨率 (width, height)
            
        Returns:
            缩放后的位置 [x, y]
        """
        base_width, base_height = 1280, 720
        actual_width, actual_height = device_resolution
        
        scale_x = actual_width / base_width
        scale_y = actual_height / base_height
        
        return [
            int(position[0] * scale_x),
            int(position[1] * scale_y)
        ]

    def _generate_fallback_actions(self, analysis: dict, device_resolution: Tuple[int, int]) -> List[dict]:
        """
        生成通用的后备操作序列（当 VLM 没有明确建议时）
        
        Args:
            analysis: VLM 分析结果
            device_resolution: 设备分辨率
            
        Returns:
            操作列表
        """
        actions = []
        scene = analysis.get('scene', '').lower()
        
        # 根据场景类型生成通用操作
        if 'dialog' in scene or 'popup' in scene or '弹窗' in scene:
            # 弹窗类：尝试点击屏幕中央偏下（通常是确认按钮位置）
            center_x, center_y = device_resolution[0] // 2, device_resolution[1] * 3 // 5
            actions.append({
                "action": "click",
                "position": [center_x, center_y],
                "description": "点击屏幕中央（通用确认位置）"
            })
        
        elif 'loading' in scene or '加载' in scene:
            # 加载界面：等待
            actions.append({
                "action": "wait",
                "seconds": 10,
                "description": "等待加载完成"
            })
        
        else:
            # 未知场景：尝试点击返回按钮
            actions.append({
                "action": "click",
                "position": [60, 60],
                "description": "尝试点击返回按钮（左上角）"
            })
        
        return actions


class SafeExecutor:
    """
    安全执行器：执行操作序列并提供安全保障
    """

    def __init__(self, device, max_attempts=3):
        """
        初始化安全执行器
        
        Args:
            device: Device 设备对象
            max_attempts: 最大尝试次数
        """
        self.device = device
        self.max_attempts = max_attempts
        self.action_history = []  # 操作历史记录

    def execute_actions(self, actions: List[dict]) -> bool:
        """
        执行操作序列
        
        Args:
            actions: 操作列表
            
        Returns:
            bool: 是否成功执行所有操作
        """
        if not actions:
            logger.warning("没有可执行的操作")
            return False

        logger.info(f"准备执行 {len(actions)} 个操作")
        
        for i, action in enumerate(actions, 1):
            logger.info(f"[{i}/{len(actions)}] {action['description']}")
            
            # 检测循环（防止陷入死循环）
            if self._is_repetitive(action):
                logger.warning("检测到重复操作，可能陷入循环，中止执行")
                return False
            
            # 执行具体操作
            try:
                if action['action'] == 'click':
                    position = action['position']
                    
                    # 验证坐标合法性
                    if not self._validate_position(position):
                        logger.error(f"点击坐标超出屏幕范围: {position}")
                        continue
                    
                    self.device.click(*position)
                    logger.info(f"  -> 点击坐标: {position}")
                    self.device.sleep(1.5)  # 等待界面响应
                
                elif action['action'] == 'wait':
                    seconds = action['seconds']
                    logger.info(f"  -> 等待 {seconds} 秒")
                    self.device.sleep(seconds)
                
                else:
                    logger.warning(f"未知操作类型: {action['action']}")
                    continue
                
                # 记录操作历史
                self.action_history.append({
                    'action': action,
                    'timestamp': datetime.now()
                })
            
            except Exception as e:
                logger.error(f"执行操作时出错: {e}")
                return False
        
        logger.info("所有操作执行完毕")
        return True

    def _is_repetitive(self, action: dict) -> bool:
        """
        检测是否重复执行相同操作（防止死循环）
        
        Args:
            action: 当前操作
            
        Returns:
            bool: 是否为重复操作
        """
        # 检查最近 5 次操作
        recent_actions = self.action_history[-5:]
        
        # 统计与当前操作相似的次数
        similar_count = 0
        for record in recent_actions:
            if self._actions_similar(record['action'], action):
                similar_count += 1
        
        # 如果近期有 3 次相同操作，视为循环
        return similar_count >= 3

    def _actions_similar(self, action1: dict, action2: dict) -> bool:
        """
        判断两个操作是否相似
        
        Args:
            action1: 操作1
            action2: 操作2
            
        Returns:
            bool: 是否相似
        """
        if action1['action'] != action2['action']:
            return False
        
        if action1['action'] == 'click':
            # 点击操作：比较坐标（允许一定误差）
            pos1 = action1['position']
            pos2 = action2['position']
            distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
            return distance < 50  # 50像素内视为相同位置
        
        # 其他操作：直接比较描述
        return action1.get('description') == action2.get('description')

    def _validate_position(self, position: List[int]) -> bool:
        """
        验证点击坐标是否在屏幕范围内
        
        Args:
            position: 坐标 [x, y]
            
        Returns:
            bool: 坐标是否合法
        """
        width, height = self.device.resolution
        x, y = position
        
        return 0 <= x <= width and 0 <= y <= height

    def reset_history(self):
        """重置操作历史（新的恢复尝试开始时调用）"""
        self.action_history = []


class VLMGameController:
    """
    VLM 游戏控制器主类
    整合视觉分析、动作规划和安全执行
    """

    def __init__(self, config, device):
        """
        初始化 VLM 游戏控制器
        
        Args:
            config: AzurLaneConfig 配置对象
            device: Device 设备对象
        """
        self.config = config
        self.device = device
        
        self.vision_analyzer = VisionAnalyzer(config)
        self.action_planner = ActionPlanner(config)
        self.safe_executor = SafeExecutor(device, max_attempts=3)
        
        self.attempt_count = 0  # 当前问题的尝试次数
        self.max_attempts = config.VLMGameController_MaxRecoveryAttempts

    def attempt_recovery(self, screenshot: Image.Image, context: dict) -> bool:
        """
        尝试使用 VLM 恢复游戏状态
        
        Args:
            screenshot: 当前游戏截图
            context: 错误上下文
                {
                    "task": "当前任务",
                    "error": "错误信息"
                }
        
        Returns:
            bool: 是否成功恢复
        """
        self.attempt_count += 1
        logger.hr(f"VLM 恢复尝试 {self.attempt_count}/{self.max_attempts}", level=1)
        
        # 重置执行器历史
        self.safe_executor.reset_history()
        
        try:
            # 步骤 1: 视觉分析
            logger.info("步骤 1: 调用 VLM 分析游戏状态...")
            analysis = self.vision_analyzer.analyze_game_state(screenshot, context)
            
            if not analysis:
                logger.error("VLM 分析失败，无法继续")
                return False
            
            logger.info(f"场景识别: {analysis.get('scene', '未知')}")
            logger.info(f"描述: {analysis.get('description', '')}")
            logger.info(f"卡死原因: {analysis.get('stuck_reason', '无')}")
            logger.info(f"置信度: {analysis.get('confidence', 0):.2f}")
            
            # 步骤 2: 生成操作序列
            logger.info("步骤 2: 生成恢复操作序列...")
            actions = self.action_planner.plan_recovery_actions(
                analysis,
                self.device.resolution
            )
            
            if not actions:
                logger.warning("未能生成有效的操作序列")
                return False
            
            logger.info(f"已生成 {len(actions)} 个操作")
            
            # 步骤 3: 执行操作
            logger.info("步骤 3: 执行操作...")
            success = self.safe_executor.execute_actions(actions)
            
            if success:
                logger.info("✓ VLM 操作执行成功")
                # 等待界面稳定
                self.device.sleep(2)
                return True
            else:
                logger.warning("✗ VLM 操作执行失败")
                return False
        
        except Exception as e:
            logger.error(f"VLM 恢复过程中出现异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def can_retry(self) -> bool:
        """
        判断是否还可以继续尝试
        
        Returns:
            bool: 是否可以继续尝试
        """
        return self.attempt_count < self.max_attempts

    def reset_attempts(self):
        """重置尝试计数（新的错误场景时调用）"""
        self.attempt_count = 0
