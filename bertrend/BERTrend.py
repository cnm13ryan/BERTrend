#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
from bertopic import BERTopic
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from bertrend import (
    MODELS_DIR,
    CACHE_PATH,
    BERTREND_DEFAULT_CONFIG_PATH,
    load_toml_config,
)

from bertrend.BERTopicModel import BERTopicModel
from bertrend.config.parameters import (
    DOC_INFO_DF_FILE,
    TOPIC_INFO_DF_FILE,
    DOC_GROUPS_FILE,
    MODELS_TRAINED_FILE,
    EMB_GROUPS_FILE,
    HYPERPARAMS_FILE,
    BERTOPIC_SERIALIZATION,
)
from bertrend.trend_analysis.weak_signals import (
    _initialize_new_topic,
    update_existing_topic,
    _apply_decay_to_inactive_topics,
)
from bertrend.utils.data_loading import TEXT_COLUMN


class BERTrend:
    """
    A comprehensive trend analysis and weak signal detection tool using BERTopic.

    Key Parameters:
    - embedding_model_name: Name of the embedding model to use
    - granularity: Number of days to group documents
    - min_chars: Minimum character length for documents
    - split_by_paragraph: Whether to split documents by paragraph
    - sample_size: Number of documents to sample
    """

    def __init__(
        self,
        config_file: str | Path = BERTREND_DEFAULT_CONFIG_PATH,
        topic_model: BERTopicModel = None,
    ):
        """
        Instanciate a class from a TOML config file.
        `config_file` can be:
            - a `str` representing the TOML file
            - a `Path` to a TOML file

        To see file format and list of parameters: bertrend/config/bertrend_default_config.toml
        """
        # Load configuration file
        self.config_file = config_file
        self.config = self._load_config()

        # Initialize topic model
        self.topic_model = BERTopicModel() if topic_model is None else topic_model

        # State variables of BERTrend
        self._is_fitted = False
        self._are_models_merged = False

        # Variables related to time-based topic models
        # - topic_models: Dictionary of trained BERTopic models for each timestamp.
        self.topic_models: Dict[pd.Timestamp, BERTopic] = {}
        # - doc_groups: Dictionary of document groups for each timestamp.
        self.doc_groups: Dict[pd.Timestamp, List[str]] = {}
        # - emb_groups: Dictionary of document embeddings for each timestamp.
        self.emb_groups: Dict[pd.Timestamp, np.ndarray] = {}

        # Variables containing info about merged topics
        self.all_new_topics_df = None
        self.all_merge_histories_df = None
        self.merged_df = None

        # Variables containing info about topic popularity
        # - topic_sizes: Dictionary storing topic sizes and related information over time.
        self.topic_sizes: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: defaultdict(list)
        )
        # - topic_last_popularity: Dictionary storing the last known popularity of each topic.
        self.topic_last_popularity: Dict[int, float] = {}
        # - topic_last_update: Dictionary storing the last update timestamp of each topic.
        self.topic_last_update: Dict[int, pd.Timestamp] = {}

    def _load_config(self) -> dict:
        """
        Load the TOML config file as a dict when instanciating the class.
        """
        config = load_toml_config(self.config_file)
        return config

    def _train_by_period(
        self,
        period: pd.Timestamp,
        group: pd.DataFrame,
        embedding_model: SentenceTransformer,
        embeddings: np.ndarray,
    ) -> Tuple[
        BERTopic,
        List[str],
        np.ndarray,
    ]:
        """
        Train BERTopic models for a given time period from the grouped data.

        Args:
            period (pd.Timestamp): Timestamp of the time period
            group (pd.DataFrame): Group of data associated to that timestamp.
            embedding_model (SentenceTransformer): Sentence transformer model for embeddings.
            embeddings (np.ndarray): Precomputed document embeddings.


        Returns:
            Tuple[BERTopic, List[str], np.ndarray]:
                - topic_model: trained BERTopic models for this period.
                - doc_group: document groups for this period.
                - emb_group: document embeddings for this period.
        """
        docs = group[TEXT_COLUMN].tolist()
        embeddings_subset = embeddings[group.index]

        logger.debug(f"Processing period: {period}")
        logger.debug(f"Number of documents: {len(docs)}")

        logger.debug("Creating topic model...")
        topic_model = self.topic_model.fit(
            docs=docs,
            embedding_model=embedding_model,
            embeddings=embeddings_subset,
        ).topic_model

        logger.debug("Topic model created successfully")

        doc_info_df = topic_model.get_document_info(docs=docs)
        doc_info_df = doc_info_df.rename(columns={"Document": "Paragraph"})
        doc_info_df = doc_info_df.merge(
            group[[TEXT_COLUMN, "document_id", "source", "url"]],
            left_on="Paragraph",
            right_on=TEXT_COLUMN,
            how="left",
        )
        doc_info_df = doc_info_df.drop(columns=[TEXT_COLUMN])

        topic_info_df = topic_model.get_topic_info()
        topic_doc_count_df = (
            doc_info_df.groupby("Topic")["document_id"]
            .nunique()
            .reset_index(name="Document_Count")
        )
        topic_sources_df = (
            doc_info_df.groupby("Topic")["source"]
            .apply(list)
            .reset_index(name="Sources")
        )
        topic_urls_df = (
            doc_info_df.groupby("Topic")["url"].apply(list).reset_index(name="URLs")
        )

        topic_info_df = topic_info_df.merge(topic_doc_count_df, on="Topic", how="left")
        topic_info_df = topic_info_df.merge(topic_sources_df, on="Topic", how="left")
        topic_info_df = topic_info_df.merge(topic_urls_df, on="Topic", how="left")

        topic_info_df = topic_info_df[
            [
                "Topic",
                "Count",
                "Document_Count",
                "Representation",
                "Name",
                "Representative_Docs",
                "Sources",
                "URLs",
            ]
        ]

        topic_model.doc_info_df = doc_info_df
        topic_model.topic_info_df = topic_info_df
        return topic_model, docs, embeddings_subset

    def train_topic_models(
        self,
        grouped_data: Dict[pd.Timestamp, pd.DataFrame],
        embedding_model: SentenceTransformer,
        embeddings: np.ndarray,
    ):
        """
        Train BERTopic models for each timestamp in the grouped data.

        Stores Tuple[Dict[pd.Timestamp, BERTopic], Dict[pd.Timestamp, List[str]], Dict[pd.Timestamp, np.ndarray]]:
            - topic_models: Dictionary of trained BERTopic models for each timestamp.
            - doc_groups: Dictionary of document groups for each timestamp.
            - emb_groups: Dictionary of document embeddings for each timestamp.

        Args:
            grouped_data (Dict[pd.Timestamp, pd.DataFrame]): Dictionary of grouped data by timestamp.
            embedding_model (SentenceTransformer): Sentence transformer model for embeddings.
            embeddings (np.ndarray): Precomputed document embeddings.
        """
        topic_models = {}
        doc_groups = {}
        emb_groups = {}

        non_empty_groups = [
            (period, group) for period, group in grouped_data.items() if not group.empty
        ]

        # Set up progress bar
        # TODO: tqdm
        # progress_bar = st.progress(0)
        # progress_text = st.empty()

        for i, (period, group) in enumerate(non_empty_groups):
            try:
                logger.info(f"Training topic model {i+1}/{len(non_empty_groups)}...")
                topic_models[period], doc_groups[period], emb_groups[period] = (
                    self._train_by_period(period, group, embedding_model, embeddings)
                )  # TODO: parallelize?
                logger.debug(f"Successfully processed period: {period}")

            except Exception as e:
                logger.error(f"Error processing period {period}: {str(e)}")
                logger.exception("Traceback:")
                continue  # TODO: better error handling

            # Update progress bar
            """
            progress = (i + 1) / len(non_empty_groups)
            progress_bar.progress(progress)
            progress_text.text(
                f"Training BERTopic model for {period} ({i + 1}/{len(non_empty_groups)})"
            )
            """

        self._is_fitted = True

        # Update topic_models: Dictionary of trained BERTopic models for each timestamp.
        self.topic_models = topic_models
        # Update doc_groups: Dictionary of document groups for each timestamp.
        self.doc_groups = doc_groups
        # Update emb_groups: Dictionary of document embeddings for each timestamp.
        self.emb_groups = emb_groups
        logger.success("Finished training all topic models")

    def merge_all_models(
        self,
        min_similarity: int | None = None,
    ):
        """Merge together all topic models."""
        # Get default BERTrend config if argument is not provided
        if min_similarity is None:
            min_similarity = self.config["min_similarity"]

        # Check if model is fitted
        if not self._is_fitted:
            raise RuntimeError("You must fit the BERTrend model before merging models.")

        topic_dfs = {
            period: _preprocess_model(
                model, self.doc_groups[period], self.emb_groups[period]
            )
            for period, model in self.topic_models.items()
        }

        timestamps = sorted(topic_dfs.keys())
        merged_df_without_outliers = None
        all_merge_histories = []
        all_new_topics = []

        # TODO: tqdm
        merge_df_size_over_time = []

        for i, (current_timestamp, next_timestamp) in enumerate(
            zip(timestamps[:-1], timestamps[1:])
        ):
            df1 = topic_dfs[current_timestamp][
                topic_dfs[current_timestamp]["Topic"] != -1
            ]
            df2 = topic_dfs[next_timestamp][topic_dfs[next_timestamp]["Topic"] != -1]

            if merged_df_without_outliers is None:
                if not (df1.empty or df2.empty):
                    (
                        merged_df_without_outliers,
                        merge_history,
                        new_topics,
                    ) = _merge_models(
                        df1,
                        df2,
                        min_similarity=min_similarity,  # SessionStateManager.get("min_similarity"),
                        timestamp=current_timestamp,
                    )
            elif not df2.empty:
                (
                    merged_df_without_outliers,
                    merge_history,
                    new_topics,
                ) = _merge_models(
                    merged_df_without_outliers,
                    df2,
                    min_similarity=min_similarity,  # SessionStateManager.get("min_similarity"),
                    timestamp=current_timestamp,
                )
            else:
                continue

            all_merge_histories.append(merge_history)
            all_new_topics.append(new_topics)
            merge_df_size_over_time = merge_df_size_over_time  # SessionStateManager.get("merge_df_size_over_time")
            merge_df_size_over_time.append(
                (
                    current_timestamp,
                    merged_df_without_outliers["Topic"].max() + 1,
                )
            )

            # progress_bar.progress((i + 1) / len(timestamps))

        all_merge_histories_df = pd.concat(all_merge_histories, ignore_index=True)
        all_new_topics_df = pd.concat(all_new_topics, ignore_index=True)

        self.merged_df = merged_df_without_outliers
        self.all_merge_histories_df = all_merge_histories_df
        self.all_new_topics_df = all_new_topics_df

        logger.success("All models merged successfully")
        self._are_models_merged = True

    def calculate_signal_popularity(
        self,
        decay_factor: float | None = None,
        decay_power: float | None = None,
    ):
        """
        Compute the popularity of signals (topics) over time, accounting for merges and applying decay.
        Updates:
           - topic_sizes (Dict[int, Dict[str, Any]]): Dictionary storing topic sizes and related information over time.
           - topic_last_popularity (Dict[int, float]): Dictionary storing the last known popularity of each topic.
           - topic_last_update (Dict[int, pd.Timestamp]): Dictionary storing the last update timestamp of each topic.

        Args:
            all_merge_histories_df (pd.DataFrame): DataFrame containing all merge histories.
            granularity (int): Granularity of the timestamps in days.
            decay_factor (float): Factor for exponential decay calculation.
            decay_power (float): Power for exponential decay calculation.

        Returns:

        """
        # Get default BERTrend config if argument is not provided
        if decay_factor is None:
            decay_factor = self.config["decay_factor"]
        if decay_power is None:
            decay_power = self.config["decay_power"]

        # Check if models are merged
        if not self._are_models_merged:
            raise RuntimeWarning(
                "You must merge topic models first before computing signal popularity."
            )

        topic_sizes = defaultdict(lambda: defaultdict(list))
        topic_last_popularity = {}
        topic_last_update = {}

        min_timestamp = self.all_merge_histories_df["Timestamp"].min()
        max_timestamp = self.all_merge_histories_df["Timestamp"].max()
        granularity_timedelta = pd.Timedelta(days=self.config["granularity"])
        time_range = pd.date_range(
            start=min_timestamp.to_pydatetime(),
            end=(max_timestamp + granularity_timedelta).to_pydatetime(),
            freq=granularity_timedelta,
        )

        for current_timestamp in time_range:
            current_df = self.all_merge_histories_df[
                self.all_merge_histories_df["Timestamp"] == current_timestamp
            ]
            updated_topics = set()

            # Process active topics (those appearing in the current timestamp)
            for _, row in current_df.iterrows():
                current_topic = row["Topic1"]
                updated_topics.add(current_topic)

                if current_topic not in topic_sizes:
                    # Initialize new topic
                    _initialize_new_topic(
                        topic_sizes,
                        topic_last_popularity,
                        topic_last_update,
                        current_topic,
                        current_timestamp,
                        row,
                    )

                # Update existing topic
                update_existing_topic(
                    topic_sizes,
                    topic_last_popularity,
                    topic_last_update,
                    current_topic,
                    current_timestamp,
                    granularity_timedelta,
                    row,
                )

                # Mark the topic as updated for the next timestamp
                updated_topics.add(current_topic)

            # Apply decay to topics that weren't updated in this timestamp or the next
            next_timestamp = current_timestamp + granularity_timedelta
            next_df = self.all_merge_histories_df[
                self.all_merge_histories_df["Timestamp"] == next_timestamp
            ]
            topics_updated_next = set(next_df["Topic1"])

            _apply_decay_to_inactive_topics(
                topic_sizes,
                topic_last_popularity,
                topic_last_update,
                updated_topics,
                topics_updated_next,
                current_timestamp,
                granularity_timedelta,
                decay_factor,
                decay_power,
            )

        self.topic_sizes = topic_sizes
        self.topic_last_popularity = topic_last_popularity
        self.topic_last_update = topic_last_update

    def save_models(self, models_path: Path = MODELS_DIR):
        if models_path.exists():
            shutil.rmtree(models_path)
        models_path.mkdir(parents=True, exist_ok=True)

        # Save topic models using the selected serialization type
        for period, topic_model in self.topic_models.items():
            model_dir = models_path / period.strftime("%Y-%m-%d")
            model_dir.mkdir(exist_ok=True)
            embedding_model = topic_model.embedding_model
            topic_model.save(
                model_dir,
                serialization=BERTOPIC_SERIALIZATION,
                save_ctfidf=False,
                save_embedding_model=embedding_model,
            )

            topic_model.doc_info_df.to_pickle(model_dir / DOC_INFO_DF_FILE)
            topic_model.topic_info_df.to_pickle(model_dir / TOPIC_INFO_DF_FILE)

        # Save topic model parameters
        with open(CACHE_PATH / HYPERPARAMS_FILE, "wb") as f:
            pickle.dump(self.topic_model, f)
        # Save doc_groups file
        with open(CACHE_PATH / DOC_GROUPS_FILE, "wb") as f:
            pickle.dump(self.doc_groups, f)
        # Save emb_groups file
        with open(CACHE_PATH / EMB_GROUPS_FILE, "wb") as f:
            pickle.dump(self.emb_groups, f)
        # Save the models_trained flag
        with open(CACHE_PATH / MODELS_TRAINED_FILE, "wb") as f:
            pickle.dump(self._is_fitted, f)

        logger.info(f"Models saved to: {models_path}")

    @classmethod
    def restore_models(cls, models_path: Path = MODELS_DIR) -> "BERTrend":
        if not models_path.exists():
            raise FileNotFoundError(f"models_path={models_path} does not exist")

        logger.info(f"Loading models from: {models_path}")

        # Create BERTrend object
        bertrend = cls()

        # load topic model parameters
        with open(CACHE_PATH / HYPERPARAMS_FILE, "rb") as f:
            bertrend.topic_model = pickle.load(f)
        # load doc_groups file
        with open(CACHE_PATH / DOC_GROUPS_FILE, "rb") as f:
            bertrend.doc_groups = pickle.load(f)
        # load emb_groups file
        with open(CACHE_PATH / EMB_GROUPS_FILE, "rb") as f:
            bertrend.emb_groups = pickle.load(f)
        # load the models_trained flag
        with open(CACHE_PATH / MODELS_TRAINED_FILE, "rb") as f:
            bertrend._is_fitted = pickle.load(f)

        # Restore topic models using the selected serialization type
        topic_models = {}
        for period_dir in models_path.iterdir():
            if period_dir.is_dir():
                topic_model = BERTopic.load(period_dir)

                doc_info_df_file = period_dir / DOC_INFO_DF_FILE
                topic_info_df_file = period_dir / TOPIC_INFO_DF_FILE
                if doc_info_df_file.exists() and topic_info_df_file.exists():
                    topic_model.doc_info_df = pd.read_pickle(doc_info_df_file)
                    topic_model.topic_info_df = pd.read_pickle(topic_info_df_file)
                else:
                    logger.warning(
                        f"doc_info_df or topic_info_df not found for period {period_dir.name}"
                    )

                period = pd.Timestamp(period_dir.name.replace("_", ":"))
                topic_models[period] = topic_model
        bertrend.topic_models = topic_models

        return bertrend


def _preprocess_model(
    topic_model: BERTopic, docs: List[str], embeddings: np.ndarray
) -> pd.DataFrame:
    """
    Preprocess a BERTopic model by extracting topic information, document groups, document embeddings, and URLs.

    Args:
        topic_model (BERTopic): A fitted BERTopic model.
        docs (List[str]): List of documents.
        embeddings (np.ndarray): Precomputed document embeddings.

    Returns:
        pd.DataFrame: A DataFrame with topic information, document groups, document embeddings, and URLs.
    """
    topic_info = topic_model.topic_info_df
    doc_info = topic_model.doc_info_df
    doc_groups = doc_info.groupby("Topic")["Paragraph"].apply(list)

    topic_doc_embeddings = []
    topic_embeddings = []
    topic_sources = []
    topic_urls = []

    for topic_docs in doc_groups:
        doc_embeddings = [embeddings[docs.index(doc)] for doc in topic_docs]
        topic_doc_embeddings.append(doc_embeddings)
        topic_embeddings.append(np.mean(doc_embeddings, axis=0))
        topic_sources.append(
            doc_info[doc_info["Paragraph"].isin(topic_docs)]["source"].tolist()
        )
        topic_urls.append(
            doc_info[doc_info["Paragraph"].isin(topic_docs)]["url"].tolist()
        )

    topic_df = pd.DataFrame(
        {
            "Topic": topic_info["Topic"],
            "Count": topic_info["Count"],
            "Document_Count": topic_info["Document_Count"],
            "Representation": topic_info["Representation"],
            "Documents": doc_groups.tolist(),
            "Embedding": topic_embeddings,
            "DocEmbeddings": topic_doc_embeddings,
            "Sources": topic_sources,
            "URLs": topic_urls,
        }
    )

    return topic_df


def _merge_models(
    df1: pd.DataFrame, df2: pd.DataFrame, min_similarity: float, timestamp: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged_df = df1.copy()
    merge_history = []

    embeddings1 = np.stack(df1["Embedding"].values)
    embeddings2 = np.stack(df2["Embedding"].values)

    similarities = cosine_similarity(embeddings1, embeddings2)
    max_similarities = np.max(similarities, axis=0)
    max_similar_topics = df1["Topic"].values[np.argmax(similarities, axis=0)]

    new_topics_mask = max_similarities < min_similarity
    new_topics_data = df2[new_topics_mask].copy()
    new_topics_data["Topic"] = np.arange(
        merged_df["Topic"].max() + 1,
        merged_df["Topic"].max() + 1 + len(new_topics_data),
    )
    new_topics_data["Timestamp"] = timestamp

    merged_df = pd.concat([merged_df, new_topics_data], ignore_index=True)
    new_topics = new_topics_data.copy()

    merge_topics_mask = max_similarities >= min_similarity
    merge_topics_data = df2[merge_topics_mask]

    for max_similar_topic, group in merge_topics_data.groupby(
        max_similar_topics[merge_topics_mask]
    ):
        similar_row = df1[df1["Topic"] == max_similar_topic].iloc[0]
        index = merged_df[merged_df["Topic"] == max_similar_topic].index[0]

        merged_df.at[index, "Count"] += group["Count"].sum()
        merged_df.at[index, "Document_Count"] += group["Document_Count"].sum()

        # Update the 'Documents' field with only the new documents from the current timestamp
        new_documents = [doc for docs in group["Documents"] for doc in docs]
        merged_df.at[index, "Documents"] = similar_row["Documents"] + [
            (timestamp, new_documents)
        ]

        merged_df.at[index, "Sources"] += [
            source for sources in group["Sources"] for source in sources
        ]
        merged_df.at[index, "URLs"] += [url for urls in group["URLs"] for url in urls]

        merge_history.extend(
            {
                "Timestamp": timestamp,
                "Topic1": max_similar_topic,
                "Topic2": row["Topic"],
                "Representation1": similar_row["Representation"],
                "Representation2": row["Representation"],
                "Embedding1": similar_row["Embedding"],
                "Embedding2": row["Embedding"],
                "Similarity": max_similarities[row["Topic"]],
                "Count1": len(similar_row["Documents"]),
                "Count2": len(row["Documents"]),
                "Document_Count1": similar_row["Document_Count"],
                "Document_Count2": row["Document_Count"],
                "Documents1": similar_row["Documents"],
                "Documents2": row["Documents"],
                "Source1": similar_row["Sources"],
                "Source2": row["Sources"],
                "URLs1": similar_row["URLs"],
                "URLs2": row["URLs"],
            }
            for _, row in group.iterrows()
        )

    return merged_df, pd.DataFrame(merge_history), new_topics
