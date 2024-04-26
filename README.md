# Boosting-RAG-in-Education
## 集成RAG、微调与思维链的LLM应用程序实践分享

在这里，我们很高兴向您呈现一份从零到一构建集成RAG（Retrieval-Augmented Generation）、微调技术及思维链功能的LLM（Large Language Model）应用程序的实践经历。我们的目标是通过这份分享，为您提供有价值的参考与指导。

---

## 使用指南

### 1. 构建知识库

运行`ingest.py`脚本，用于搭建应用程序所需的知识库。您可在`config`文件中设定以下参数：

- **知识文件路径**：指定用于构建知识库的数据源位置。
- **embedding模型**：选择适用的嵌入模型以辅助知识库构建。

### 2. 选择与调用模型API

进入`API`文件夹，挑选您偏好的系列模型，如`qwen`或`chatglm`。这些模型的API调用代码可分别在其官方仓库中获取。若上述资源无法使用，可考虑采用`ollama`作为备选。

**提示**：如需启用**query-expansion**功能，建议启动两个不同规模的模型，例如一个7b参数量的模型与一个14b参数量的模型，以实现更好的效果。

### 3. 启动服务与初步测试

在成功配置并启动服务后，进行适当的测试以验证应用程序的运行状态与功能完整性。

### 4. 运行Web界面

如需部署Web界面供用户交互，直接执行`Chat`文件夹下的`app.py`脚本。在此之前，请确保已正确配置所有相关参数。

### 5. 数据评估

如需直接评估数据，运行`RAG`目录下的`build_ragas_date_full.py`脚本，该脚本将帮助您执行完整的评估流程。

---

**后续更新**：以上仅为现阶段的主要步骤与说明，未来如有更多实践经验或重要提示，我们将及时补充。

---

**硬件配置与依赖**：

- **显卡型号**：本项目采用NVIDIA A100进行开发与测试。
- **库文件**：所有必要的库及其版本要求已列出于`requirements.txt`文件中。请确保环境依照此文件进行配置以确保项目的顺利运行。
