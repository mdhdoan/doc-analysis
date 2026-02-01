import json
import os
import sys

from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer

HEADERS = [
    'all_text'
]

def increase_count(count, character):
    count += 1
    print(character, end="", flush=True)
    return count


if __name__ == '__main__':
    model_name = sys.argv[1][2:]
    # model_name = 'data/docs_json'
    topic_model = BERTopic.load("model/" + 'docs_json')
    label_file_path = "model/" + 'docs_json'
    label_file_name = os.path.join(label_file_path, 'topics.json')
    with open(label_file_name, 'r') as l_file:
        label_data = json.load(l_file)
    label_list = label_data['custom_labels']
    bad_string_list = ['[',']','"','\n']
    label_dict = {}
    dict_length = len(label_list)
    for label in label_list:
        for bad_string in bad_string_list:
            label = label.replace(bad_string, '')
    for i in range (1,dict_length):
        label_dict[str(i)] = label_list[i]     
    print(label_dict)
    topic_model.set_topic_labels(label_dict)

    in_path, documents, count = sys.argv[1], [], 0
    errorfile = ''
    for file in os.listdir(in_path):
        file_name = os.path.join(in_path, file)
        if not os.path.isfile(file_name) or not file_name.endswith('.json'):
            continue
            
        with open(file_name, 'rt', encoding='utf-8') as in_file:
            doc = json.load(in_file)
            try:
                documents.append('\n'.join([doc[k] for k in HEADERS if doc[k]]))
            except Exception as e:
                print(f"\n{file_name}")
            count = increase_count(count, '.')
    print(f"\nRead {count} documents.\n")

    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(documents, show_progress_bar=True)

    os.makedirs('viz/' + model_name, exist_ok = True)
    viz_topics = topic_model.visualize_topics(top_n_topics = dict_length - 1)
    viz_topics.show()
    viz_topics.write_html("viz/" + model_name + 'method-topics.html')
    
    # viz_heatmap = topic_model.visualize_heatmap()
    viz_heatmap = topic_model.visualize_heatmap(custom_labels=True)
    viz_heatmap.show()
    viz_heatmap.write_html("viz/" + model_name + 'method-heatmap.html')

    viz_words = topic_model.visualize_barchart(top_n_topics = dict_length - 1)
    # viz_words = topic_model.visualize_barchart(top_n_topics=20, custom_labels=True)
    viz_words.show()
    viz_words.write_html("viz/" + model_name + 'method-words.html')

    # # Run the visualization with the original embeddings
    topic_model.visualize_documents(documents, embeddings=embeddings)

    # Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
    reduced_embeddings = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    viz_docs = topic_model.visualize_documents(documents, reduced_embeddings=reduced_embeddings, width=3096, height=1440, custom_labels=True)
    viz_docs.show()
    viz_docs.write_html("viz/" + model_name + 'method-docs.html')
    
    hierarchical_topics = topic_model.hierarchical_topics(documents)
    viz_hie_tops = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics, custom_labels=True)
    viz_hie_tops.show()
    viz_hie_tops.write_html("viz/" + model_name + 'method-hierarchy.html')
    