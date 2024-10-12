# GraphRAG 使用指南

## 1. 安装项目依赖

请确保安装项目依赖，具体依赖请参见主页的 `requirements.txt` 文件：

```bash
pip install -r requirements.txt
```

> **注意**: 本项目使用的 **GraphRAG** 版本为 `0.3.0`。

---

## 2. 创建 GraphRAG 所需文件夹

在 `GraphRAG/Index/` 目录下创建以下文件夹，后续所有操作均在该目录下进行：

```bash
mkdir -p ./input
mkdir -p ./storage
mkdir -p ./cache
```

---

## 3. 初始化项目

执行以下命令进行初始化：

```bash
python -m graphrag.index --init --root ./
```

---

## 4. 修改配置文件（.env 和 settings.yaml）

### 使用本地大模型 (Ollama 方案)

Ollama 是一个轻量级、跨平台的工具和库，专门为本地大语言模型 (LLM) 的部署和运行提供支持。它简化了本地环境中运行大模型的过程，无需依赖云服务或外部 API。

#### (1) 安装 Ollama

访问官网 [Ollama](https://ollama.com/) 下载并安装对应系统版本。

#### (2) 启动 Ollama 并安装模型

使用以下指令安装所需的本地模型：

```bash
ollama pull qwen2.5:32b
ollama pull nomic-embed-text:latest
```

本次使用的模型如下：
- **LLM 模型**: `qwen2.5:32b`
- **Embedding 模型**: `nomic-embed-text:latest`

#### (3) 修改 `.env` 文件

将 `.env` 文件修改为如下内容：

```bash
GRAPHRAG_CHAT_MODEL=qwen2.5:32b
GRAPHRAG_EMBEDDING_MODEL=nomic-embed-text:latest
```

---

## 5. 生成提示词文件

执行以下命令生成提示词文件：

```bash
python -m graphrag.prompt_tune --config ./settings.yaml --root ./ --no-entity-types --language Chinese --output ./prompts
```

---

## 6. 构建索引

执行以下命令构建索引：

```bash
python -m graphrag.index --root ./
```

---

## 7. 测试 GraphRAG

### (1) 修改 API 服务配置

进入 `tools` 文件夹，打开 `create_graphrag_api_server.py` 文件，修改如下内容：

- 文件路径：`INPUT_DIR = ".../artifacts"`
- 向量数据库的集合名称：`entity_description_embeddings`

根据实际情况自定义调整向量数据库的集合名称，代码如下：

```python
description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
```

该文件的目的是创建 API 服务，方便后续测试。

### (2) 运行测试文件

运行 `test_graphrag_from_api.py` 文件，根据需求修改 `messages` 中的 `query` 问题，具体细节请参考代码注释。

---

## 8. 使用 Neo4j 图数据库进行知识图谱可视化

### (1) 安装 Neo4j

使用 Docker 安装 Neo4j：

```bash
docker pull neo4j:latest
```

### (2) 启动 Neo4j

执行以下命令启动 Neo4j：

```bash
docker run \
    -p 7474:7474 -p 7687:7687 \
    --name neo4j \
    -v /usr/data:/data \
    -e NEO4J_AUTH=neo4j/testpassword \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4JLABS_PLUGINS=\["apoc","apoc-extended"\] \
    neo4j
```

### (3) 添加 APOC 插件

上述操作还不足以成功可视化，neo4j需要APOC插件才能完全发挥，因此我们需要下载插件。点击 [这里](https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/5.23.0/apoc-5.23.0-extended.jar) 下载 `APOC` 插件，并将其添加到容器内部目录 `/var/lib/neo4j/plugins/` 中。

### (4) 修改 `neo4j.conf` 文件

在 `neo4j.conf` 文件中添加以下语句：

```bash
dbms.security.procedures.unrestricted=apoc.*
dbms.security.allow_csv_import_from_file_urls=true
apoc.import.file.use_neo4j_config=true
apoc.export.file.enabled=true
apoc.import.file.enabled=true
```

### (5) 重启 Neo4j 容器

```bash
docker restart neo4j
```

### (6) 运行可视化脚本

运行 `create_visualization_by_neo4j.py`，然后在浏览器中访问 [localhost:7474](http://localhost:7474)，即可在 Neo4j 的可视化界面中查看生成的知识图谱。
