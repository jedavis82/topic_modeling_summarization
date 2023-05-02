"""
Store the model's topics and documents to a json file.
Perform sentiment analysis for each topic and store it to the json file also.
Perform summarization of each document; use these summaries to compute topic summaries.
Store the summaries of each document and topic summary to json.

"""

from top2vec import Top2Vec
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from transformers import pipeline
import config
import pandas as pd

# Have to set the CSV max limit high since the content is exceeding the pandas limit
import csv
csv.field_size_limit(25000000)


def generate_model_output(model=None, original_df=None, summarizer=None, nlp=None, output_dir=None):
    topic_words, topic_scores, topic_nums = model.get_topics()
    topic_sizes = model.get_topic_sizes()[0]

    # Array will be written to the output file for the topic output
    topic_summary_array = []
    for tn, ts in zip(topic_nums, topic_sizes):
        doc_scores, doc_ids = model.search_documents_by_topic(topic_num=tn, num_docs=10)
        top_words = topic_words[tn]
        top_words = [t for t in top_words if t not in nlp.Defaults.stop_words]

        # Grab the documents from the original CSV file
        orig_data = original_df[original_df['id'].isin(doc_ids)]

        # Create a "topic name" by concatenating the top 4 top_words and add to the dataframe
        top_name = '-'.join(t for t in top_words[:4])
        orig_data['topic_name'] = top_name
        # Summarize the articles from the extracted data and add to the dataframe
        # Careful how I add these in to the df
        docs = list(orig_data['content'])
        summaries = []
        for out in summarizer(docs, batch_size=2, truncation=True, min_length=10, max_length=250):
            summaries.append(out['summary_text'])

        orig_data['summary'] = summaries
        # I think this is everything, write the results out to a CSV file
        topic_output_file = f'{output_dir}topics/topic_{tn}.csv'
        orig_data.to_csv(topic_output_file, index=False, header=True, encoding='utf-8')

        # Concatenate the summaries from above. Summarize the summaries.
        # Store those in a dictionary or some other data structure that can be written out to
        # a csv or json file.
        all_summaries = '. '.join(s for s in summaries)
        topic_summary = summarizer(all_summaries, min_length=20, max_length=250, truncation=True)
        topic_summary = topic_summary[0]['summary_text']
        topic_summary_array.append({
            'topic_id': tn,
            'topic_name': top_name,
            'topic_words': top_words[:10],
            'topic_summary': topic_summary
        })
    topic_df = pd.DataFrame.from_dict(topic_summary_array)
    topic_summary_output = f'{output_dir}topic_summaries.csv'
    topic_df.to_csv(topic_summary_output, index=False, header=True, encoding='utf-8')
    print()


def main():
    model_file = config.model_file
    articles_csv_file = config.articles_csv_file
    output_dir = config.output_dir

    articles_df = pd.read_csv(articles_csv_file, index_col=[0])
    articles_df.drop(columns=['url'], inplace=True)
    spacy_nlp = spacy.load('en_core_web_lg', exclude=['tok2vec', 'tagger', 'parser',
                                                      'attribute_ruler', 'lemmatizer',
                                                      'ner'])
    spacy_nlp.add_pipe('spacytextblob')
    print(f'Loaded spacy pipes: {spacy_nlp.pipe_names}')

    print('Loading BART model for summarization')
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn',
                          tokenizer='facebook/bart-large-cnn', framework='pt', device=0)

    print(f'Loading topic model from {model_file}')
    model = Top2Vec.load(model_file)
    generate_model_output(model, articles_df, summarizer, spacy_nlp, output_dir)


if __name__ == '__main__':
    main()
