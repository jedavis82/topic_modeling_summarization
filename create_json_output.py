"""
Create JSON files for each of the topic files and for the topic summary file.
Will be easier to read the summaries.
"""
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, basename, exists

topics_dir = './model_output/topics/'
topic_summary_file = './model_output/topic_summaries.csv'
json_out_dir = './model_output/topics_json/'
if not exists(json_out_dir):
    makedirs(json_out_dir)


def main():
    topic_files = [join(topics_dir, f) for f in listdir(topics_dir) if isfile(join(topics_dir, f))]
    for t in topic_files:
        df = pd.read_csv(t, encoding='utf-8')
        df_to_json = df.to_json(orient='records', indent=4)
        base_name = basename(t).split('.')[0]
        json_file = f'{json_out_dir}{base_name}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            f.write(df_to_json)


if __name__ == '__main__':
    main()
