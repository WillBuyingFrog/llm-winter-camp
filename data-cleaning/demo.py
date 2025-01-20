from openai import OpenAI
import os
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset

# 设置OpenAI客户端
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

ROWS = 68023

def is_chronic_disease_related(text, verbose=False, identifier=None):
    # 读取prompt
    with open("prompt_v0.txt", "r") as f:
        prompt = f.read()
    
    if verbose:
        # 保存实际的prompt
        with open(f"output/actual_prompt_{identifier}.txt", "w") as f:
            f.write(f"{prompt}{text}")
    
    try:
        # 调用API进行判断
        completion = client.chat.completions.create(
            model="qwen2.5-72b-instruct",
            messages=[{"role": "user", "content": f"{prompt}{text}"}],
            temperature=0
        )
        response = completion.choices[0].message.content
        return response.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_batch(batch_indices, ds, batch_size, verbose):
    text = ""
    start_idx = batch_indices[0]
    for idx in batch_indices:
        history = ds[idx]["history"]
        text += f"{idx + 1}:{history}\n"

    result = is_chronic_disease_related(text, verbose, start_idx)
    if verbose:
        print(f"Batch Result for batch starting at {start_idx}: {result}")

    batch_results = {}
    batch_filtered = []
    if result is not None:
        results_split = result.split('\n')
        for item in results_split:
            if not item.strip():
                continue
            parts = item.split(':', 1)
            if len(parts) != 2:
                continue
            try:
                code = int(parts[0].strip())
                judge_result = int(parts[1].strip())
                batch_results[code] = judge_result
                if judge_result == 1:
                    # 找到对应的idx
                    relative_idx = code - start_idx - 1
                    if relative_idx >= 0 and relative_idx < len(batch_indices):
                        actual_idx = batch_indices[relative_idx]
                        batch_filtered.append(ds[actual_idx])
            except Exception as e:
                print(f"Error processing item: {item}, error: {e}")

    return batch_results, batch_filtered

    

def load_and_preprocess_hf_dataset(request_batch=10, start_range=0.0, end_range=1.0, verbose=False, num_threads=5):
    print("Loading dataset...")
    start_idx = int(ROWS * start_range)
    end_idx = int(ROWS * end_range) 
    ds = load_dataset("Suprit/CMtMedQA", split=f"train[{start_idx}:{end_idx}]")
    print(f"Dataset length: {len(ds)}")

    if verbose:
        # 清空output子目录
        for file in os.listdir("output"):
            if file.endswith(".txt"):
                os.remove(f"output/{file}")

    print(f"Processing from index {start_idx} to {end_idx}")
    # 遍历数据集中的每条数据

    shuffled_ds = ds.shuffle(seed=42)
    total_samples = end_idx - start_idx

    # 将数据集分成多个批次，每个批次包含多个样本
    batch_size = request_batch
    num_batches = (total_samples + batch_size - 1) // batch_size
    batches = []
    for i in range(num_batches):
        start = i * batch_size + start_idx
        end = min((i + 1) * batch_size + start_idx, end_idx)
        batch_indices = list(range(start, end))
        batches.append(batch_indices)

    # 使用线程池处理每个批次

    _counter = 0
    output_id = 0
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for batch in batches:
            future = executor.submit(process_batch, batch, shuffled_ds, batch_size, verbose)
            futures.append(future)

        results = {}
        actual_filtered_data = []
        for future in as_completed(futures):
            batch_results, batch_filtered = future.result()
            if verbose:
                print(f"Batch Results: {batch_results}")
            _counter += 1
            results.update(batch_results)
            actual_filtered_data.extend(batch_filtered)

            if len(actual_filtered_data) >= 50:
                        # 导出数据到JSON文件
                with open(f"output/actual_filtered_data_{output_id}.json", "w", encoding="utf-8") as f:
                    json.dump(actual_filtered_data, f, ensure_ascii=False, indent=2)
                print(f"Exported {len(actual_filtered_data)} items to actual_filtered_data_{output_id}.json")
                output_id += 1
                actual_filtered_data = []

            if _counter % 50 == 0:
                print(f"Processed {_counter} batches")

    return results, actual_filtered_data

    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_range", type=float, default=0.0)
    parser.add_argument("--end_range", type=float, default=1.0)
    parser.add_argument("--request_batch", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num_threads", type=int, default=5)
    args = parser.parse_args()  

    _, real_data = load_and_preprocess_hf_dataset(
        args.request_batch, args.start_range, args.end_range, args.verbose, args.num_threads
    )
    
    # 导出过滤后的数据到JSON文件
    with open("output/filtered_data.json", "w", encoding="utf-8") as f:
        json.dump(real_data, f, ensure_ascii=False, indent=2)