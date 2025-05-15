#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os

import requests
import json

from joblib import Parallel, delayed
from langchain_core.embeddings import Embeddings
from loguru import logger

import numpy as np
from typing import List

from bertrend.services.authentication import SecureAPIClient

MAX_N_JOBS = 4
BATCH_DOCUMENT_SIZE = 1000
MAX_DOCS_PER_REQUEST_PER_WORKER = 20000


class EmbeddingAPIClient(Embeddings):
    """
    Client for LM Studio’s OpenAI-style **/v1/embeddings** endpoint.

    Custom Embedding API client, can integrate seamlessly with langchain
    """

    def __init__(self, url: str, model_name: str, num_workers: int = 4):
        """
         Parameters
         ----------
         url
             Base URL of the LM Studio server (e.g. ``"http://localhost:1234"``).
         model_name
             Embedding model identifier (e.g. ``"text-embedding-bge-base-en-v1.5"``).
        num_workers
             Worker processes for client-side batching.  LM Studio offers no
             ``/num_workers`` endpoint, so the caller supplies a default.
        """
        self.url = url.rstrip("/")
        self.model_name = model_name
        self.num_workers = num_workers
        logger.debug(
            f"EmbeddingAPIClient(model_name={model_name!r}, num_workers={num_workers!r})"
        )

    # --------------------------------------------------------------------- #
    # Helpers                                                               #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _safe_json(resp: requests.Response, url: str) -> dict:
        """
        Parse *resp* as JSON and raise when LM Studio signals an error.

        LM Studio always returns HTTP 200; failures are encoded as
        ``{"error": "..."}`` bodies.
        """
        try:
            body = resp.json()
        except ValueError as exc:
            raise RuntimeError(f"{url} returned non-JSON: {resp.text[:120]}") from exc

        if isinstance(body, dict) and "error" in body:
            raise RuntimeError(f"{url} error: {body['error']}")
        return body

    def embed_query(
        self, text: str | list[str], show_progress_bar: bool = False
    ) -> list[float]:
        if isinstance(text, str):
            text = [text]

        endpoint = f"{self.url}/v1/embeddings"
        payload = {"model": self.model_name, "input": text}

        logger.debug(f"POST {endpoint} [1 query]")
        body = self._safe_json(
            requests.post(endpoint, json=payload, timeout=30), endpoint
        )

        embeddings = np.array([d["embedding"] for d in body["data"]])
        return embeddings.tolist()[0]

    def embed_batch(
        self, texts: list[str], show_progress_bar: bool = True
    ) -> list[list[float]]:
        endpoint = f"{self.url}/v1/embeddings"
        payload = {"model": self.model_name, "input": texts}

        body = self._safe_json(
            requests.post(endpoint, json=payload, timeout=60), endpoint
        )

        embeddings = np.array([d["embedding"] for d in body["data"]])
        logger.debug("Computing embeddings done for batch")
        return embeddings.tolist()

    def embed_documents(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
        batch_size: int = BATCH_DOCUMENT_SIZE,
    ) -> list[list[float]]:
        if len(texts) > MAX_DOCS_PER_REQUEST_PER_WORKER * self.num_workers:
            # Too many documents to embed in one request, refuse it
            logger.error(
                f"Error: Too many documents to be embedded ({len(texts)} chunks, max {MAX_DOCS_PER_REQUEST_PER_WORKER * self.num_workers})"
            )
            raise ValueError(
                f"Error: Too many documents to be embedded ({len(texts)} chunks, max {MAX_DOCS_PER_REQUEST_PER_WORKER * self.num_workers})"
            )

        logger.debug(f"Calling EmbeddingAPI using model: {self.model_name}")

        # Split texts into chunks
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        logger.debug(
            f"Computing embeddings on {len(texts)} documents using ({len(batches)}) batches..."
        )

        def _embed_batch_request(
            base_url: str, model_name: str, texts: list[str]
        ) -> list[list[float]]:
            endpoint = f"{base_url}/v1/embeddings"
            payload = {"model": model_name, "input": texts}
            body = EmbeddingAPIClient._safe_json(
                requests.post(endpoint, json=payload, timeout=60), endpoint
            )
            return [d["embedding"] for d in body["data"]]

        # Parallel request
        results = Parallel(n_jobs=MAX_N_JOBS)(
            delayed(_embed_batch_request)(self.url, self.model_name, batch)
            for batch in batches
        )

        # Check results
        if any(result == [] for result in results):
            raise ValueError(
                "At least one batch processing failed. Documents are not embedded."
            )

        # # Compile results
        # embeddings = [embedding for result in results for embedding in result]

        # Flatten batch results
        embeddings: List[List[float]] = [emb for result in results for emb in result]
        assert len(embeddings) == len(texts)
        return embeddings

    async def aembed_query(self, text: str) -> list[float]:
        # FIXME!
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        # FIXME!
        return self.embed_documents(texts)
