from openai import OpenAI
import os
import argparse
import json

from datasets import load_dataset

# 设置OpenAI客户端
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

ROWS = 68023

def is_chronic_disease_related(text, verbose=False, identifier=None):

    with open("prompt_v0.txt", "r") as f:
        prompt = f.read()
    
    if verbose:
        with open(f"output/actual_prompt_{identifier}.txt", "w") as f:
            f.write(f"{prompt}{text}")

    try:
        completion = client.chat.completions.create(
            model="qwen2.5-72b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}{text}"
                }
            ],
            # provider={
            #     "order": [
            #         "Fireworks",
            #         "Together",
            #         "DeepInfra"
            #     ]
            # },
            temperature=0
        )
        response = completion.choices[0].message.content
        # 确保只返回数字1或0
        return response.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None
    

def load_and_preprocess_hf_dataset(request_batch=10, start_range=0.0, end_range=1.0, verbose=False):
    print("Loading dataset...")
    start_idx = int(ROWS * start_range)
    end_idx = int(ROWS * end_range) 
    ds = load_dataset("Suprit/CMtMedQA", split=f"train[{start_idx}:{end_idx}]")
    print(f"Dataset length: {len(ds)}")

    if verbose:
        # 清空output子目录
        for file in os.listdir("output"):
            os.remove(f"output/{file}")

    print(f"Processing from index {start_idx} to {end_idx}")
    # 遍历数据集中的每条数据

    shuffled_ds = ds.shuffle(seed=42)

    _counter = 0
    text = ""
    results = {}

    actual_filtered_data = []
    for idx in range(start_idx, end_idx):
        history = shuffled_ds[idx]["history"]
        text += f"{idx + 1}:{history}\n"

        _counter += 1

        if _counter % request_batch == 0 or idx == end_idx - 1:
            result = is_chronic_disease_related(text, verbose, _counter)
            if verbose:
                print(f"Batch Result for {_counter}: {result}")
            results_split = result.split('\n')
            for item in results_split:
                judge_result = int(item.split(':')[1])
                if judge_result == 1:
                    actual_filtered_data.append(shuffled_ds[idx])
                results[int(item.split(':')[0])] = judge_result

            
            text = ""

    return results, actual_filtered_data

    




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_range", type=float, default=0.0)
    parser.add_argument("--end_range", type=float, default=1.0)
    parser.add_argument("--request_batch", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()  

    _, real_data = load_and_preprocess_hf_dataset(args.request_batch, args.start_range, args.end_range, args.verbose)
    
    # Export real_data to JSON file
    with open("output/filtered_data.json", "w", encoding="utf-8") as f:
        json.dump(real_data, f, ensure_ascii=False, indent=2)

    # with open("data_demo.txt", "r") as f:
    #     data_demo = f.read()

    # print(is_chronic_disease_related(data_demo))