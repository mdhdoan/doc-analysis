from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from bertopic.representation import KeyBERTInspired

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

representation_model = KeyBERTInspired()
topic_model = BERTopic(representation_model=representation_model)
topics, probs = topic_model.fit_transform(docs)

topic_model.get_topic_info()

topic_model.get_document_info(docs)

topic_model.visualize_topics()