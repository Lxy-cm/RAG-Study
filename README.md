RAG科研训练

## 问答接口

本项目现在提供一层 FastAPI 问答接口，并预留 Temporal workflow/activity 编排。

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

`.env` 至少需要配置千问 API Key，二选一即可：

```bash
QWEN_API_KEY=你的阿里云百炼APIKey
# 或
DASHSCOPE_API_KEY=你的阿里云百炼APIKey
```

可选配置：

```bash
DB_DIR=./database
QA_DB_PATH=C:\Users\86185\AppData\Local\Temp\RAG-Study-main\qa_messages.sqlite3
LLM_MODEL_NAME=qwen-plus
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QA_USE_TEMPORAL=true
TEMPORAL_ADDRESS=localhost:7233
TEMPORAL_TASK_QUEUE=qa-task-queue
```

### 2. 启动 HTTP 接口

历史消息现在默认保存在 SQLite 数据库：

```text
C:\Users\86185\AppData\Local\Temp\RAG-Study-main\qa_messages.sqlite3
```

如果之前用过 `data/qa_messages.json`，可以迁移旧历史：

```bash
python migrate_messages_to_db.py
```

如果还没有导入教材资料，先把 `.md` 或平台导出的 `course.json`、`knowledge_graph.json`、`problems.json`
放到 `data` 目录，然后构建知识库：

```bash
python build_knowledge_base.py
```

如果 Windows 在项目目录写 Chroma/SQLite 时出现 `disk I/O error`，可以把向量库放到用户临时目录：

```bash
set DB_DIR=C:\Users\86185\AppData\Local\Temp\rag_database_json
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
python build_knowledge_base.py
```

启动接口时也要使用同一个 `DB_DIR`。

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 3. 如果要走 Temporal

先启动 Temporal 服务，然后另开终端启动 worker：

```bash
python worker.py
```

接口默认会优先尝试 Temporal。Temporal 没启动时会自动降级为本地同步执行，方便前端联调。
如果希望 Temporal 连不上就直接报错，可以设置：

```bash
QA_USE_TEMPORAL=required
```

### 4. 测试接口

也可以直接打开测试页面：

```text
http://localhost:8000/frontend
```

非流式问答：

```bash
curl -X POST http://localhost:8000/conversations/conv_1/messages ^
  -H "Content-Type: application/json" ^
  -d "{\"content\":\"导数的几何意义是什么？\",\"retrieval\":{\"limit\":5}}"
```

流式问答：

```bash
curl -N -X POST http://localhost:8000/conversations/conv_1/messages/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"content\":\"导数的几何意义是什么？\",\"retrieval\":{\"limit\":5}}"
```

查询历史消息：

```bash
curl "http://localhost:8000/conversations/conv_1/messages?limit=20&offset=0"
```
