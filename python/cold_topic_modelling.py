#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# -----------------------------
# Config
# -----------------------------

# zeroshot list: # Option A: Compact labels (12–16)
PRIVACY_GDPR_LABELS_COMPACT = [
    "Scope & Definitions",
    "Roles & Responsibilities",
    "Lawful Basis / Permission",
    "Consent Management",
    "Transparency / Notices",
    "Data Subject Rights",
    "Data Minimization & Purpose Limitation",
    "Retention & Deletion",
    "Security Safeguards",
    "Breach / Incident Response",
    "Third Parties & Processors",
    "International Transfers",
    "Risk & Assessments (DPIA)",
    "Governance & Accountability",
    "Enforcement / Complaints",
    "Cookies / Tracking (Online Identifiers)",
]

# Option B: Detailed labels (25–35)
PRIVACY_GDPR_LABELS_DETAILED = [
    "Personal Data Definition",
    "Sensitive / Special Category Data",
    "Children’s Data",
    "Anonymization / Pseudonymization",
    "Controller Duties",
    "Processor Duties",
    "Joint Controllers",
    "DPO / Privacy Office",
    "Records of Processing (ROPA)",
    "Lawful Basis: Consent",
    "Lawful Basis: Contract",
    "Lawful Basis: Legal Obligation",
    "Lawful Basis: Legitimate Interests",
    "Transparency: Privacy Notice Content",
    "Transparency: Collection Context / Just-in-time Notice",
    "Right: Access",
    "Right: Rectification",
    "Right: Erasure",
    "Right: Restriction",
    "Right: Portability",
    "Right: Objection / Marketing Opt-out",
    "Automated Decision-making / Profiling",
    "Purpose Limitation",
    "Data Minimization",
    "Accuracy",
    "Retention Schedule",
    "Deletion / Disposal",
    "Security: Access Control",
    "Security: Encryption / Key Mgmt",
    "Security: Logging / Monitoring",
    "Breach Detection",
    "Breach Notification (Authority)",
    "Breach Notification (Individuals)",
    "Vendor Contracts / DPAs",
    "Subprocessors",
    "Disclosures / Sharing",
    "International Transfers: Safeguards",
    "Risk Assessment / DPIA",
    "Privacy by Design / Default",
    "Complaints & Supervisory Authority",
    "Penalties / Enforcement",
]


DEFAULT_HEADERS = ("all_text",)


@dataclass(frozen=True)
class TopicConfig:
    embedding_model_id: str = "all-MiniLM-L6-v2"
    # If your corpus is meaningfully bilingual (EN/FR), consider:
    # embedding_model_id: str = "paraphrase-multilingual-MiniLM-L12-v2"

    umap_n_neighbors: int = 5
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    random_state: int = 42

    hdb_min_cluster_size: int = 3
    hdb_metric: str = "euclidean"
    hdb_cluster_selection_method: str = "eom"

    ngram_range: tuple[int, int] = (1, 3)
    min_df: int = 2
    stop_words: str | None = "english"  # Set to None if you want to avoid dropping French words

    top_n_words: int = 10
    calculate_probabilities: bool = True


# -----------------------------
# Helpers
# -----------------------------

def load_documents(in_dir: Path, headers: Sequence[str]) -> List[str]:
    docs: List[str] = []
    files = sorted(in_dir.glob("*.json"))

    if not files:
        raise FileNotFoundError(f"No .json files found in: {in_dir}")

    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            logging.warning("Failed to read JSON: %s", fp)
            continue

        parts: List[str] = []
        for k in headers:
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())

        joined = "\n".join(parts).strip()
        if joined:
            docs.append(joined)
        else:
            logging.debug("Skipping empty doc after headers=%s: %s", headers, fp)

    return docs


def sanitize_label(text: str, max_words: int = 5) -> str:
    # Remove brackets/quotes/newlines, collapse whitespace
    text = re.sub(r"[\[\]\"]", "", text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text).strip()

    # Keep only first max_words words
    words = text.split()
    return " ".join(words[:max_words]).strip()


def build_topic_model(cfg: TopicConfig, embedding_model: SentenceTransformer) -> BERTopic:
    umap_model = UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        n_components=cfg.umap_n_components,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        random_state=cfg.random_state,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg.hdb_min_cluster_size,
        metric=cfg.hdb_metric,
        cluster_selection_method=cfg.hdb_cluster_selection_method,
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words=cfg.stop_words,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
    )

    ctfidf_model = ClassTfidfTransformer(
        bm25_weighting=True,
        reduce_frequent_words=True,
    )

    representation_model = KeyBERTInspired()

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        top_n_words=cfg.top_n_words,
        calculate_probabilities=cfg.calculate_probabilities,
        zeroshot_topic_list=PRIVACY_GDPR_LABELS_COMPACT,
        verbose=True,
    )


def make_label_chain(ollama_model: str, temperature: float = 0.0):
    # Chat models + prompt templates are the modern LangChain path (LCEL).
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You create short topic labels for clustered documents. "
                "Return ONLY the topic label. No quotes, no bullets, no extra text.",
            ),
            (
                "human",
                "Documents:\n```{documents}```\n\n"
                "Keywords:\n```{keywords}```\n\n"
                "Constraints:\n"
                "- 5 words or fewer\n"
                "- No code\n"
                "- Use Title Case if appropriate\n"
                "Examples:\n"
                "Streamflow Measurement in Streams\n"
                "Wetland Habitat and Waterfowl Management",
            ),
        ]
    )

    llm = ChatOllama(model=ollama_model, temperature=temperature, validate_model_on_init=False)
    return prompt | llm | StrOutputParser()


def label_topics_with_llm(
    topic_model: BERTopic,
    chain,
    max_docs_per_topic: int = 3,
    max_words: int = 10,
) -> Dict[int, str]:
    info = topic_model.get_topic_info()  # DataFrame with Topic/Count/Name etc.
    topic_ids = [int(t) for t in info["Topic"].tolist()]

    labels: Dict[int, str] = {-1: "Outlier Topic"}

    for topic_id in topic_ids:
        if topic_id == -1:
            continue

        # Keywords from BERTopic (word, weight)
        topic_words = topic_model.get_topic(topic_id) or []
        keywords = [w for (w, _score) in topic_words][:10]
        kw_block = ", ".join(keywords)

        # Representative docs: use method if available; fallback to attribute if needed
        docs_list: List[str]
        try:
            docs_list = topic_model.get_representative_docs(topic_id)  # typically returns list[str]
        except Exception:
            docs_list = []
            rep = getattr(topic_model, "representative_docs_", None)
            if isinstance(rep, dict):
                docs_list = rep.get(topic_id, []) or []

        docs_block = "\n\n---\n\n".join((docs_list or [])[:max_docs_per_topic]).strip()
        if not docs_block:
            labels[topic_id] = sanitize_label(topic_model.get_topic_info().loc[info["Topic"] == topic_id, "Name"].iloc[0])
            continue

        raw = chain.invoke({"documents": docs_block, "keywords": kw_block})
        cleaned = sanitize_label(raw, max_words=max_words) or f"Topic {topic_id}"
        labels[topic_id] = cleaned

        logging.info("Topic %s labeled as: %s", topic_id, cleaned)

    return labels


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BERTopic training + LLM topic labeling (Ollama).")
    p.add_argument("in_path", type=Path, help="Directory containing *.json files")
    p.add_argument("--headers", nargs="+", default=list(DEFAULT_HEADERS), help="JSON keys to concatenate as document text")
    p.add_argument("--output_dir", type=Path, default=Path("model"), help="Where to save the model")
    p.add_argument("--model_name", type=str, default=None, help="Output folder name (default: input dir name)")
    p.add_argument("--embedding_model", type=str, default=TopicConfig.embedding_model_id, help="SentenceTransformers model id")
    p.add_argument("--ollama_model", type=str, default="llama3.1", help="Ollama model name for labeling")
    p.add_argument("--serialization", type=str, default="safetensors", choices=["safetensors", "pytorch", "pickle"])
    p.add_argument("--stop_words", type=str, default="english", help='CountVectorizer stop_words ("english" or "none")')
    p.add_argument("--min_cluster_size", type=int, default=TopicConfig.hdb_min_cluster_size)
    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    in_dir: Path = args.in_path
    if not in_dir.exists() or not in_dir.is_dir():
        raise NotADirectoryError(f"Invalid in_path: {in_dir}")

    stop_words = None if str(args.stop_words).lower() in {"none", "null", "false", "0"} else args.stop_words

    cfg = TopicConfig(
        embedding_model_id=args.embedding_model,
        hdb_min_cluster_size=args.min_cluster_size,
        stop_words=stop_words,
    )

    documents = load_documents(in_dir, headers=args.headers)
    logging.info("Loaded %d documents.", len(documents))

    # Embeddings
    embedding_model = SentenceTransformer(cfg.embedding_model_id)
    embeddings = embedding_model.encode(documents, show_progress_bar=True, normalize_embeddings=True)

    # BERTopic
    topic_model = build_topic_model(cfg, embedding_model=embedding_model)
    topics, probs = topic_model.fit_transform(documents, embeddings)
    logging.info("Model fit complete. Found %d topics (incl. outliers).", len(set(topics)))

    # LLM labeling via Ollama
    chain = make_label_chain(args.ollama_model, temperature=0.1)
    label_dict = label_topics_with_llm(topic_model, chain)
    topic_model.set_topic_labels(label_dict)

    # Save (safetensors is generally recommended by BERTopic)
    out_dir = args.output_dir / (args.model_name or in_dir.name)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # For safetensors/pytorch, BERTopic expects a *string pointer* for save_embedding_model.
    topic_model.save(
        str(out_dir),
        serialization=args.serialization,
        save_ctfidf=True,
        save_embedding_model=cfg.embedding_model_id,
    )

    logging.info("Saved model to: %s", out_dir)
    logging.info("Topic info:\n%s", topic_model.get_topic_info().to_string(index=False))


if __name__ == "__main__":
    main()
