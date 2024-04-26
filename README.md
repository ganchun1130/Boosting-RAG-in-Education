# Boosting-RAG-in-Education
这是一份集成了RAG和微调以及思维链的LLM应用程序！在这里我们分享了从0到1的一份实践经历，希望对你们有所帮助！\n\n

--Usage\n
1.运行ingest.py文件，用来构建知识库。在config文件中可以指定知识文件路径以及选用的embedding模型。\n
2.打开API文件夹，选择一个系列模型，可以是qwen，也可以是chatglm，这些模型的api调用代码均可以在各自的官方仓库中找到，实在不行用ollama也可以。\n
3.服务启动后可以适当的测试一下，如果要使用query-expansion应该启动两个模型，最好是一个7b，一个14b。\n
4.如果你想使用web页面，你可以直接运行Chat文件夹下的app.py，但也要配置一些参数。\n
5.如果你想直接评估数据，你可以运行RAG目录下的build_ragas_date_full.py文件。\n

就先说到这里，以后想到什么再补充吧。\n\n

哦对了！我是用的显卡是A100!库文件在requirements.txt文件中！\n\n
