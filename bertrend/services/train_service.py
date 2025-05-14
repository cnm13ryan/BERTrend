"""
bertrend.services.train_service
-------------------------------

Pure-function wrapper around the former BERTrend._train_by_period
method.  The body is copied verbatim so behaviour stays identical;
only the implicit `self` has become an explicit ``bertrend`` arg.

Keeping this logic out of the monolithic class lets us unit-test it
in isolation and (later) parallelise training without touching
BERTrend’s public API.
"""

from __future__ import annotations

from typing import Tuple, List

import numpy as np
import pandas as pd
from bertopic import BERTopic
from loguru import logger
from sentence_transformers import SentenceTransformer

from bertrend.utils.data_loading import TEXT_COLUMN
from bertrend.BERTopicModel import BERTopicModel  # for type hints only


def train_by_period(  # pylint: disable=too-many-locals
    bertrend: "BERTrend",
    period: pd.Timestamp,
    group: pd.DataFrame,
    embedding_model: SentenceTransformer | str,
    embeddings: np.ndarray,
) -> Tuple[BERTopic, List[str], np.ndarray]:
    """
    Train a BERTopic model for **one** time‐slice and enrich it with
    document / topic metadata.

    Parameters
    ----------
    bertrend : BERTrend
        Instance holding the shared BERTopicModel and config.
    period : pd.Timestamp
        Timestamp representing the slice to train.
    group : pd.DataFrame
        Sub-dataframe for this slice.
    embedding_model : SentenceTransformer | str
        Embedding model to pass to BERTopic.
    embeddings : np.ndarray
        Pre-computed embeddings (global array, will be subset).

    Returns
    -------
    tuple
        (trained BERTopic, documents list, embeddings subset)
    """
    docs = group[TEXT_COLUMN].tolist()
    embeddings_subset = embeddings[group.index]

    logger.debug("Processing period %s with %d documents", period, len(docs))

    logger.debug("Fitting topic model…")
    topic_model = bertrend.topic_model.fit(
        docs=docs,
        embeddings=embeddings_subset,
    ).topic_model
    logger.debug("Topic model fitted")

    # ---------- enrich with doc / topic metadata -----------------------
    doc_info_df = topic_model.get_document_info(docs=docs).rename(
        columns={"Document": "Paragraph"}
    )
    doc_info_df = doc_info_df.merge(
        group[[TEXT_COLUMN, "document_id", "source", "url"]],
        left_on="Paragraph",
        right_on=TEXT_COLUMN,
        how="left",
    ).drop(columns=[TEXT_COLUMN])

    topic_info_df = topic_model.get_topic_info()
    topic_doc_count_df = (
        doc_info_df.groupby("Topic")["document_id"]
        .nunique()
        .reset_index(name="Document_Count")
    )
    topic_sources_df = (
        doc_info_df.groupby("Topic")["source"].apply(list).reset_index(name="Sources")
    )
    topic_urls_df = (
        doc_info_df.groupby("Topic")["url"].apply(list).reset_index(name="URLs")
    )

    topic_info_df = (
        topic_info_df.merge(topic_doc_count_df, on="Topic", how="left")
        .merge(topic_sources_df, on="Topic", how="left")
        .merge(topic_urls_df, on="Topic", how="left")
        .loc[
            :,
            [
                "Topic",
                "Count",
                "Document_Count",
                "Representation",
                "Name",
                "Representative_Docs",
                "Sources",
                "URLs",
            ],
        ]
    )

    # attach the enriched frames
    topic_model.doc_info_df = doc_info_df
    topic_model.topic_info_df = topic_info_df

    return topic_model, docs, embeddings_subset
