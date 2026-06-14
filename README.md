# Web AI Agent

一个基于 **Playwright + LLM** 的网页端 AI Agent，能够接收自然语言任务，自主操控浏览器完成网页操作。

## 功能

- 自然语言驱动：用一句话描述任务，Agent 自动拆解执行
- 浏览器操控：导航、点击、填写表单、提取内容、键盘操作
- 多模型支持：兼容 OpenAI / Anthropic / 任意 OpenAI 兼容 API
- ReAct 决策循环：边观察边思考，动态调整下一步
- 交互式模式：支持单次任务和交互式对话

## 快速开始

### 1. 安装

```bash
git clone https://github.com/Mangfluff/web-ai-agent.git
cd web-ai-agent
pip install -r requirements.txt
playwright install chromium
```

### 2. 配置

复制环境变量模板并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```ini
# LLM API 配置（支持 OpenAI / Anthropic / 任意兼容 API）
LLM_API_KEY=sk-your-api-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o

# 浏览器设置
BROWSER_HEADLESS=false
```

### 3. 运行

```bash
# 单次任务
python main.py "在 Google 搜索今天的天气"

# 指定模型
python main.py "打开 GitHub 搜索 ai agent 项目" --model claude-sonnet-4-20250514

# 无头模式（不显示浏览器窗口）
python main.py "访问百度首页" --headless

# 交互模式
python main.py -i
```

## 项目结构

```
web-ai-agent/
├── main.py              # CLI 入口
├── requirements.txt     # Python 依赖
├── .env.example         # 环境变量模板
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── browser.py       # 浏览器控制器（Playwright）
│   ├── llm.py           # LLM API 接口
│   └── agent.py         # AI Agent 核心逻辑（ReAct 循环）
```

## 技术原理

1. **ReAct 决策循环**：Agent 每次行动前先观察页面状态，由 LLM 决定下一步动作
2. **Playwright 驱动**：支持 Chromium 浏览器全自动化操作
3. **JSON Action 协议**：LLM 输出结构化的 JSON 指令，Agent 解析执行
4. **支持的动作类型**：`navigate`、`click`、`fill`、`press`、`extract`、`wait`、`done`

## 配置说明

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `LLM_API_KEY` | LLM API 密钥 | 必填 |
| `LLM_BASE_URL` | API 地址 | `https://api.openai.com/v1` |
| `LLM_MODEL` | 模型名称 | `gpt-4o` |
| `BROWSER_HEADLESS` | 是否无头模式 | `false` |

## 示例

```bash
# 搜索信息
python main.py "搜索 Python 异步编程的最新教程"

# 打开网站并提取内容
python main.py "打开 Hacker News 首页，列出前 5 条新闻标题"

# 多步骤操作
python main.py "访问 GitHub，搜索 playwright，按 stars 排序，打开第一个项目"
```

## 许可

MIT