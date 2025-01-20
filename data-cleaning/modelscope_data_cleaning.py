import pandas as pd
import json
import argparse


DATA_PATH = "/Users/frog_wch/playground/Projects/llm-winter-camp/repos/Chinese-medical-dialogue-data/Data_数据/Oncology_肿瘤科/肿瘤科5-10000.csv"


def read_data_as_pandas_df():
    df = pd.read_csv(DATA_PATH, encoding='GB18030')

    return df


def change_data_format(df, start_range=0.0, end_range=1.0):

    output_json_data = []

    start_index = int(start_range * len(df))
    end_index = int(end_range * len(df))

    for idx in range(start_index, end_index):
        row = df.iloc[idx]
        dialogue = {
            "query": row['ask'],
            "response": row['answer']
        }
        output_json_data.append(dialogue)

    return output_json_data


def main(args):

    df = read_data_as_pandas_df()
    output_json_data = change_data_format(df, args.start_range, args.end_range)
    
    with open("output/oncology_data.json", "w", encoding="utf-8") as f:
        json.dump(output_json_data, f, ensure_ascii=False, indent=2)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_range", type=float, default=0.0)
    parser.add_argument("--end_range", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
