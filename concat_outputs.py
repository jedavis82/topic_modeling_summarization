"""
For creating the output, I stored them in individual files.
I'll combine them into one overall CSV and one overall JSON here to share.
"""
import pandas as pd
from os import listdir
from os.path import isfile, join, basename

topics_dir = './model_output/topics/'
json_dir = './model_output/topics_json/'
output_df_file = './model_output/topics.csv'
output_json_file = './model_output/topics.json'


def main():
    topic_files = [join(topics_dir, f) for f in listdir(topics_dir) if isfile(join(topics_dir, f))]
    dfs_list = []
    for t in topic_files:
        df = pd.read_csv(t, encoding='utf-8')
        dfs_list.append(df)
    output_df = pd.concat(dfs_list)
    output_df.to_csv(output_df_file, encoding='utf-8', index=False, header=True)
    json_contents = output_df.to_json(orient='records', indent=4)
    with open(output_json_file, 'w', encoding='utf-8') as f:
        f.write(json_contents)

if __name__ == '__main__':
    main()
