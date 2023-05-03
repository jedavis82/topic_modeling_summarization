"""
There were some erroneous encodings created because the dataset itself had some malformed input.
Clean those up and re-save the csv files.
"""
import pandas as pd
from os.path import isfile, join
from os import listdir


topics_dir = './model_output/topics/'
topic_summary_file = './model_output/topic_summaries.csv'


def decode_error_strings(x):
    return str.encode(x, 'ascii', 'ignore').decode('utf-8')


def main():
    topic_files = [join(topics_dir, f) for f in listdir(topics_dir) if isfile(join(topics_dir, f))]
    for t in topic_files:
        df = pd.read_csv(t, encoding='utf-8')
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        df['title'] = df['title'].apply(decode_error_strings)
        df['content'] = df['content'].apply(decode_error_strings)
        df['topic_name'] = df['topic_name'].apply(decode_error_strings)
        df['summary'] = df['summary'].apply(decode_error_strings)
        df.to_csv(t, encoding='utf-8', index=False, header=True)

    df = pd.read_csv(topic_summary_file, encoding='utf-8')
    df['topic_name'] = df['topic_name'].apply(decode_error_strings)
    df['topic_words'] = df['topic_words'].apply(decode_error_strings)
    df['topic_summary'] = df['topic_summary'].apply(decode_error_strings)
    print()
    df.to_csv(topic_summary_file, encoding='utf-8', index=False, header=True)


if __name__ == '__main__':
    main()
