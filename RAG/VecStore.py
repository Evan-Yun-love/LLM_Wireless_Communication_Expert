from typing import List, Union, Tuple, Optional
from pathlib import Path
import json
import numpy as np
import faiss
from langchain.embeddings import HuggingFaceEmbeddings


class VectorStore:
    """VectorStore: 文本向量化、FAISS 索引管理与检索。

    支持索引类型：
        - flatl2   : IndexFlatL2  (欧氏距离，精确)
        - flatip   : IndexFlatIP  (内积/余弦，相似度)
        - ivfflat  : IndexIVFFlat (倒排 + 精确)
        - ivfpq    : IndexIVFPQ   (倒排 + 产品量化)
    """

    DEFAULT_NLIST = 100  # IVF 聚类中心个数
    DEFAULT_PQ_M = 64    # PQ 子空间数（更常用于768维嵌入）

    def __init__(self, model_path: Union[str, Path], db_path: Union[str, Path], embedding_device: str = "cuda"):
        self.model_path = str(model_path)
        self.db_path = Path(db_path)
        self.embedding_device = embedding_device

        self.documents: List[str] = []
        self.metadata: List[dict] = []
        self.vectors: Optional[np.ndarray] = None
        self.vector_store: Optional[faiss.Index] = None

        print(f"[INFO] 初始化 VectorStore -> {self.db_path}")

        if self.db_path.exists():
            self.load_vector(self.db_path)

    def _init_model(self):
        if not hasattr(self, 'model'):
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_path,
                model_kwargs={"device": self.embedding_device},
                encode_kwargs={"normalize_embeddings": True},
            )
            print("[INFO] 嵌入模型已加载")

    def load_documents_and_metadata(self, json_path: Union[str, Path]) -> None:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.documents = [item["chunk"] for item in data]
        self.metadata = [item["metadata"] for item in data]
        print(f"[INFO] 加载文档 {len(self.documents)} 条")

    def _embed_documents(self, docs: List[str], batch_size: int = 32) -> np.ndarray:
        self._init_model()
        all_vecs = []
        for i in range(0, len(docs), batch_size):
            vecs = self.model.embed_documents(docs[i : i + batch_size])
            all_vecs.append(vecs)
        return np.vstack(all_vecs).astype("float32")

    def build_from_documents(
        self,
        documents: List[str],
        metadata: List[dict],
        index_type: str = "flatl2",
        n_list: int = DEFAULT_NLIST,
        pq_m: int = DEFAULT_PQ_M,
        batch_size: int = 32,
    ) -> None:
        assert len(documents) == len(metadata), "documents 与 metadata 长度不一致"
        self.documents = documents
        self.metadata = metadata
        self.vectors = self._embed_documents(documents, batch_size=batch_size)
        self._build_index(index_type, n_list, pq_m)

    def rebuild_index_from_vectors(self, index_type: str = "flatl2", n_list: int = DEFAULT_NLIST, pq_m: int = DEFAULT_PQ_M) -> None:
        if self.vectors is None:
            raise ValueError("self.vectors 为空，无法重建索引")
        self._build_index(index_type, n_list, pq_m)

    def _build_index(self, index_type: str, n_list: int, pq_m: int):
        dim = int(self.vectors.shape[1])
        index_type = index_type.lower()
        if index_type == "flatl2":
            self.vector_store = faiss.IndexFlatL2(dim)
        elif index_type == "flatip":
            print("[WARN] 使用 IndexFlatIP，请确保使用 normalize_embeddings=True")
            self.vector_store = faiss.IndexFlatIP(dim)
        elif index_type == "ivfflat":
            quantizer = faiss.IndexFlatL2(dim)
            self.vector_store = faiss.IndexIVFFlat(quantizer, dim, n_list)
            print("[INFO] 训练 IVFFlat …")
            self.vector_store.train(self.vectors)
        elif index_type == "ivfpq":
            quantizer = faiss.IndexFlatL2(dim)
            self.vector_store = faiss.IndexIVFPQ(quantizer, dim, n_list, pq_m, 8)
            print("[INFO] 训练 IVFPQ …")
            self.vector_store.train(self.vectors)
        else:
            raise ValueError(f"不支持的 index_type: {index_type}")

        self.vector_store.add(self.vectors)
        if hasattr(self.vector_store, "nprobe"):
            self.vector_store.nprobe = min(10, n_list // 10)
        print(f"[INFO] 索引 {index_type} 构建完成，ntotal={self.vector_store.ntotal}")

    def persist(self) -> None:
        if self.vector_store is None:
            raise RuntimeError("索引未初始化")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.vector_store, str(self.db_path))
        meta = {
            "index_type": self.vector_store.__class__.__name__,
            "dimension": int(self.vector_store.d),
            "ntotal": int(self.vector_store.ntotal),
        }
        meta_path = self.db_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] 索引与 meta 已保存 -> {self.db_path}")

    def load_vector(self, db_path: Union[str, Path]) -> None:
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(f"索引文件不存在: {path}")
        self.vector_store = faiss.read_index(str(path))
        print(f"[INFO] 索引加载成功: {path}")
        print(f"[INFO] 类型: {type(self.vector_store).__name__}, 维度: {self.vector_store.d}, 数量: {self.vector_store.ntotal}")

    def save_vectors(self, path: Union[str, Path]):
        if self.vectors is None:
            raise ValueError("self.vectors 为空，无法保存")
        np.save(Path(path), self.vectors)
        print(f"[INFO] 向量矩阵保存 -> {path}")

    def load_vectors(self, path: Union[str, Path]):
        self.vectors = np.load(Path(path)).astype("float32")
        print(f"[INFO] 向量矩阵加载 shape={self.vectors.shape}")

    @staticmethod
    def _convert_distance(dist: float, mode: str = "reciprocal") -> float:
        if mode == "reciprocal":
            return 1.0 / (1.0 + dist)
        elif mode == "negative":
            return -dist
        elif mode == "linear":
            return max(0.0, 1.0 - dist)
        else:
            raise ValueError(f"未知 score_mode: {mode}")

    def search(self, query: str, k: int = 5, score_mode: str = "reciprocal", min_score: float = 0.0) -> List[Tuple[str, float, dict]]:
        if self.vector_store is None:
            raise RuntimeError("请先构建或加载索引")
        self._init_model()
        query_vec = np.array([self.model.embed_query(query)]).astype("float32")
        distances, indices = self.vector_store.search(query_vec, k)
        results: List[Tuple[str, float, dict]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                score = self._convert_distance(dist, mode=score_mode)
                if score >= min_score:
                    results.append((self.documents[idx], score, self.metadata[idx]))
        return results

    def describe(self):
        print("\n[INFO] VectorStore 状态描述：")
        print("- 文档数:", len(self.documents))
        print("- 向量数:", len(self.vectors) if self.vectors is not None else 0)
        if self.vector_store:
            print("- 索引类型:", type(self.vector_store).__name__)
            print("- 向量维度:", self.vector_store.d)
            print("- 向量总数:", self.vector_store.ntotal)
