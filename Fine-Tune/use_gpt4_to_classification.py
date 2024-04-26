import os

import openai
import time
import pandas as pd


# 设置OpenAI API的参数
openai.api_type = ""
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""


# 读取CSV文件
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df


# 将DataFrame写入CSV文件
def write_csv(df, file_path):
    df.to_csv(file_path, index=False)


# 使用GPT4理解CSV文件
def understand_csv_with_gpt4(df):

    time.sleep(20)
    # 将DataFrame转换为字符串
    csv_string = df.to_string(index=False)

    # 将字符串传递给GPT4
    response = openai.ChatCompletion.create(
        engine="gpt4test",
        messages=[
            {
                "role": "system",
                "content": "你扮演一个CSV文件理解器，"
                           "现在有一份CSV文件，列名分别是：所属模块, 用户问题, 问答类型，这是一份没有空值的CSV文件。"
                           "我希望你能充分理解这份文件，然后我会上传另外一份需要你打标（只需要对所属模块和问答类型这两列进行打标）的文件。"
                           "在你打标的时候，一定不要超出这份CSV文件中已经存在的标签。"
            },
            {
                "role": "user",
                "content": csv_string
            }
        ],
        temperature=0.3,
        max_tokens=2000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    # 返回GPT4的理解
    return response['choices'][0]['message']['content']


# 使用GPT4分类CSV文件
def classify_csv_with_gpt4(content, prompt):

    time.sleep(10)
    # 将字符串传递给GPT4
    try:
        response = openai.ChatCompletion.create(
            engine="gpt4test",
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            temperature=0.3,
            max_tokens=2000,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None)

        # 返回GPT4的分类
        return response['choices'][0]['message']['content']

    except Exception as e:
        print(e)
        return None


# 运行主函数
if __name__ == "__main__":
    csv_file_path = ""

    prompt1 = """你扮演一个用户问题分类器，我现在要上传一条用户问题。
               请你依据我之前给你上传的已经分类好的CSV文件，为我的每一条用户问题的问答类型进行打标。
               请注意你的返回格式，我只需要你返回一个问答类型的标签即可。
               这些标签只能是'标签1' '标签2' '标签3' """
    prompt2 = """你扮演一个用户问题分类器，我现在要上传一条用户问题。
               请你依据我之前给你上传的已经分类好的CSV文件，为我的每一条用户问题的所属模块进行打标。
               请注意你的返回格式，我只需要你返回一个所属模块的标签即可(只需返回文字，不需要标点符号)。
               这个标签只能是：'标签1' '标签2' '标签3' 
                """

    # 读取CSV文件
    df1 = read_csv(os.path.join(csv_path, full_csv_name))

    # 使用GPT4理解CSV文件
    understanding = understand_csv_with_gpt4(df1)
    print("GPT4的理解：", understanding)

    # df = pd.read_csv(os.path.join(csv_path, with_chunk_csv_name))
    # df = pd.read_csv(os.path.join(csv_path, with_class_csv_name))
    df2 = pd.read_csv(os.path.join(csv_path, temp_csv_name))
    count = 0
    for index, row in df2.iterrows():
        count+=1
        print(f"正在处理第{count}条数据")

        if count % 50 == 0:
            # 使用GPT4理解CSV文件
            understanding = understand_csv_with_gpt4(df1)
            print("GPT4的理解：", understanding)

        query = row['用户问题']
        print(query)
        gpt_answer = classify_csv_with_gpt4(query, prompt=prompt1)
        # df2.at[index, '所属模块'] = gpt_answer
        df2.at[index, '问答类型'] = gpt_answer
        print(
            f"--------------------------\ncircle: {index + 1}\nanswer: {gpt_answer}\n------------------------\n\n")
        # count += 1
    df2.to_csv(os.path.join(csv_path, temp_csv_name))

