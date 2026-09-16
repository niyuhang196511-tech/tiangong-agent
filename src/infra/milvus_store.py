import asyncio
import json
from typing import Iterable

from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore
from langgraph.store.base import SearchItem, Op, Result, GetOp, PutOp, SearchOp, ListNamespacesOp
from pymilvus import Collection, utility, FieldSchema, DataType, CollectionSchema

COLLECTION_NAME = "agent_long_term_memory"

def _ensure_collection(alias: str, dims: int) -> Collection:
    """确保 Milvus Collection 存在"""
    if utility.has_collection(COLLECTION_NAME, using=alias):
        return Collection(COLLECTION_NAME, alias=alias, using=alias)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
        FieldSchema(name="namespace", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="key", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="value", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dims)
        FieldSchema(name="created_at", dtype=DataType.DOUBLE),
    ]
    schema = CollectionSchema(fields, description="Agent long term memory")
    collection = Collection(COLLECTION_NAME, schema=schema, using=alias)

    # 创建索引向量
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
    )
    collection.load()
    return collection

class MilvusStore(BaseStore):
    def __init__(self, alias: str, embeddings: Embeddings, dims: int = 1024):
        self._alias = alias
        self._embeddings = embeddings
        self._dims = dims
        self._collection = _ensure_collection(alias, dims)

    def _ns_to_str(self, namespace: tuple[str, ...]) -> str:
        return ",".join(namespace)

    def _make_id(self, namespace: tuple[str, ...], key: str) -> str:
        return f"{self._ns_to_str(namespace)}::{key}"

    def _embed_text(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    def _aembed_text(self, text: str) -> list[float]:
        return self._embeddings.aembed_query(text)

    def _item_form_hit(self, hit: dict) -> SearchItem:
        value = json.loads(hit["value_json"])
        ns_str = hit["namespace"]
        namespace = tuple(ns_str.split("/")) if ns_str else ()
        return SearchItem(
            namespace=namespace,
            key=hit["key"],
            value=value,
            created_at=hit.get("created_at"),
            updated_at=hit.get("updated_at"),
            score=hit.get("score"),
        )

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return asyncio.run(self.abatch(list(ops)))

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(await self._aget(op.namespace, op.key))
            elif isinstance(op, PutOp):
                await self._aput(op.namespace, op.key)
                results.append(None)
            elif isinstance(op, SearchOp):
                items = await self._asearch(op.namespace_prefix, op.query, op.limit)
                results.append(items)
            elif isinstance(op, ListNamespacesOp):
                results.append([])
            else:
                results.append(None)
        return results
