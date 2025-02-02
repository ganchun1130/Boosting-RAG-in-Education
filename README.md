# 💡💡 Boosting-RAG-in-Education
## 🌐 集成RAG、微调与思维链的LLM应用系统实践

在这里，我们很高兴向您呈现一份从0到1构建集成RAG（Retrieval-Augmented Generation）、微调技术及思维链功能的LLM（Large Language Model）应用程序的实践研究。我们的目标是通过这份研究，为您提供有价值的参考与指导。

---
## 🔄 最新更新

2024.10.12我们对项目进行了以下更新：

- 对项目文件结构有一个大致的说明，详情请见**RREADME.md**文件的末尾。
- 增加了微软的**GraphRAG**的功能，当前处于测试阶段，后续将进行优化，相关细节请参阅 GraphRAG 文件夹中的内容。
- 计划下一步添加**Web界面**支持，并提供更详细的配置步骤。

未来还将陆续推出更多功能与优化，敬请关注！

---
## 🚀 使用指南

### 🌟 1. 构建知识库

运行`ingest.py`脚本，用于搭建应用系统所需的知识库。您可在`config`文件中设定以下参数：

- **知识文件路径**：指定用于构建知识库的数据源位置。
- **embedding模型**：选择适用的嵌入模型以辅助知识库构建。

### 🌟 2. 选择与调用模型API

进入`API`文件夹，挑选您偏好的系列模型，如`qwen`或`chatglm`。这些模型的API调用代码可分别在其官方仓库中获取。若上述资源无法使用，可考虑采用`ollama`作为备选。

**提示**：如需启用**query-expansion**功能，建议启动两个不同规模的模型，例如一个7b参数量的模型与一个14b参数量的模型，以实现更好的效果。

### 🌟 3. 启动服务与初步测试

在成功配置并启动服务后，进行适当的测试以验证应用程序的运行状态与功能完整性。

### 🌟 4. 运行Web界面

如需部署Web界面供用户交互，直接执行`Chat`文件夹下的`app.py`脚本。在此之前，请确保已正确配置所有相关参数。

### 🌟 5. 数据评估

如需直接评估数据，运行`RAG`目录下的`build_ragas_data_full.py`脚本，该脚本将帮助您执行完整的评估流程。

---

** 📚 后续更新 📚 **：以上仅为现阶段的主要步骤与说明，未来如有更多实践经验或重要提示，我们将及时补充。

---

** 💻 硬件配置与依赖 💻 **：

- **显卡型号**：本项目采用NVIDIA A100进行开发与测试。
- **库文件**：所有必要的库及其版本要求已列出于`requirements.txt`文件中。请确保环境依照此文件进行配置以确保项目的顺利运行。

---

## 📜 项目结构

 ```python
├── API  #  用于启动本地LLM的API服务
│   ├── ChatGLM
│   │   ├── api_server.py  #  启动本地模型的服务
│   │   │ 
│   │   └── openai_api_request.py  # 测试API服务是否启动成功
│   │
│   └── Qwen
│       └── 同上
│ 
├── Chat
│   ├── app.py  # 一个简单的应用程序，可以在本地编译器的命令行中与LLM交流
│   │   
│   └── MyLLM.py  # 用于创建本地LLM的类，当API服务启动完成后，这个类才有效果
│    
├── Evaluation
│   ├── data  # 存放数据集的文件夹，这是评估数据集的   
│   │     
│   ├── result  # 存放评估结果      
│   │      
│   ├── log # 日志
│   │   
│   ├── calculate.py  # 计算token，本人编写的一个小脚本，用于按token数分块时的对比
│   │   
│   ├── MyRagasLLM.py  # 这是RAGAS需要用的LLM类，和之前的MyLLM有一点点相似  
│   │
│   ├── reports_functions.py # 运行RAGAS需要用的一些函数，主要是读写文件的，也算是一个脚本
│   │ 
│   └── run_ragas.py # 顾名思义，运行RAGAS
...
先更新到这吧~
 ```

## 🌟 欢迎交流与探讨

如果您有任何创意想法或是遇到疑问，非常期待与您深入交流！随时欢迎通过以下方式联系我们：

- **QQ**: 2746992517

让我们一起在知识的海洋中航行，共同探讨，共同进步！🤝

---
