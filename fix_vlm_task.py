import json

# 修正 zh-CN.json - 在 Task 部分添加 VLMGameController
zh_cn_path = 'module/config/i18n/zh-CN.json'
with open(zh_cn_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 在 Task 中添加
if 'Task' not in data:
    data['Task'] = {}

data['Task']['VLMGameController'] = {
    "name": "VLM智能控制器",
    "help": "使用视觉语言模型（AI视觉）自动处理游戏卡死和未知界面，避免简单重启。需要安装对应的Python库（openai/anthropic/google-generativeai）并配置API Key"
}

with open(zh_cn_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✓ zh-CN.json Task 部分更新完成")

# 修正 en-US.json
en_us_path = 'module/config/i18n/en-US.json'  
with open(en_us_path, 'r', encoding='utf-8') as f:
    data_en = json.load(f)

if 'Task' not in data_en:
    data_en['Task'] = {}

data_en['Task']['VLMGameController'] = {
    "name": "VLM Game Controller",
    "help": "Use Vision-Language Model (AI Vision) to automatically handle game stuck/unknown pages. Requires installing Python libraries (openai/anthropic/google-generativeai) and configuring API Key"
}

with open(en_us_path, 'w', encoding='utf-8') as f:
    json.dump(data_en, f, ensure_ascii=False, indent=2)

print("✓ en-US.json Task 部分更新完成")
