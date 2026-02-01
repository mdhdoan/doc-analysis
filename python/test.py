from bertopic import BERTopic

from pathlib import Path
import json

from model2vec import StaticModel

if __name__ == '__main__':
    data_dir = Path("mock_data/cleaned_data")
    items = [
        (obj["title"], obj["all_text"])
        for p in data_dir.glob("*.json")
        for obj in [json.loads(p.read_text(encoding="utf-8"))]
    ]
    titles, documents = zip(*items)  # titles: tuple[str], documents: tuple[str]
    documents = list(documents)      # BERTopic input

    model = StaticModel.from_pretrained("minishlab/potion-base-32M")

    topic_model = BERTopic(embedding_model=model)
    topics, probs = topic_model.fit_transform(documents)
    topic_model.get_topic_info()