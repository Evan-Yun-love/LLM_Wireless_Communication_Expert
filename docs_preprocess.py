import os
import pickle
import hashlib
from pathlib import Path
from typing import List, Union
import faiss
import numpy as np
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import re

class DocumentProcessor:
    """文档处理与向量化存储工具类
    
    功能：加载PDF/DOCX文档，进行文本清洗、分块处理，生成向量化索引并存储
    
    属性：
        SUPPORTED_FORMATS: 支持处理的文件格式集合
        docs_path: 原始文档存储路径
        db_path: FAISS索引存储路径
        meta_path: 元数据存储路径
        hash_path: 内容哈希值存储路径
        model: 文本向量化模型
        index: FAISS向量索引
        metadata_list: 文档块元数据列表
        hash_set: 文档块哈希集合（用于去重）
    """
    SUPPORTED_FORMATS = {'.pdf', '.docx'}

    def __init__(self, hug_embed: HuggingFaceEmbeddings, docs_path: Union[str, Path], db_path: Union[str, Path]):
        """初始化文档处理器
        Args:
            hug_embed: 文本嵌入模型，用于生成文本向量
            docs_path: 原始文档存储目录路径
            db_path: FAISS索引存储路径（自动生成关联的元数据和哈希文件）
        """
        # 初始化路径相关配置
        self.docs_path = Path(docs_path)
        self.db_path = Path(db_path)
        self.meta_path = Path(str(db_path) + ".meta.pkl")
        self.hash_path = Path(str(db_path) + ".hash.pkl")
        self.model = hug_embed

        # 加载已有索引或初始化新索引
        if self.db_path.exists():
            print(f"Loading existing FAISS index from {self.db_path} ...")
            self.index = faiss.read_index(str(self.db_path))
            self.metadata_list = pickle.load(open(self.meta_path, "rb")) if self.meta_path.exists() else []
            self.hash_set = pickle.load(open(self.hash_path, "rb")) if self.hash_path.exists() else set()
        else:
            self.index = None
            self.metadata_list = []
            self.hash_set = set()

    def read_docs(self, file_path: Path):
        """读取指定格式的文档内容
        Args:
            file_path: 目标文件路径
        Returns:
            文档内容对象列表
        Raises:
            ValueError: 遇到不支持的文件格式时抛出
        """
        # 根据文件后缀选择对应的文档加载器
        if file_path.suffix.lower() == '.pdf':
            return PyMuPDFLoader(str(file_path)).load()
        elif file_path.suffix.lower() == '.docx':
            return Docx2txtLoader(str(file_path)).load()
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def docs_clean(self, text: str) -> str:
        """执行多阶段文本清洗处理
        Args:
            text: 原始文本内容
        Returns:
            清洗后的标准化文本
        """
        # 统一换行符和特殊字符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('–', '-').replace('—', '-').replace(' ', ' ')
        
        # 正则表达式处理序列（处理格式噪声、页眉页脚、版权信息等）
        text = re.sub(r'(?<=\b)(?:[a-zA-Z]\s){2,}[a-zA-Z](?=\b)', lambda m: m.group(0).replace(' ', ''), text)  # 修复被空格分割的单词
        text = re.sub(r'(?:^.{1,20}\n){3,}', lambda m: m.group(0).replace('\n', ' '), text, flags=re.MULTILINE)  # 合并短行
        text = re.sub(r'3GPP TS \d+\.\d+ V\d+\.\d+\.\d+ \(\d{4}-\d{2}\)', '', text)  # 删除技术规范编号
        text = re.sub(r'Release \d+', '', text, flags=re.IGNORECASE)  # 删除版本信息
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)  # 删除页码
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # 删除纯数字行
        text = text.replace('\f', '')  # 删除换页符
        text = re.sub(r'Copyright Notification.*?All rights reserved\.', '', text, flags=re.DOTALL | re.IGNORECASE)  # 删除版权声明
        text = re.sub(r'Postal address.*?http://www\.3gpp\.org', '', text, flags=re.DOTALL | re.IGNORECASE)  # 删除联系信息
        text = re.sub(r'UMTS™.*?GSM® and the GSM logo.*?GSM Association', '', text, flags=re.DOTALL | re.IGNORECASE)  # 删除商标信息
        text = re.sub(r'^\s*(\d+\.){1,4}[^\n]*\d{1,4}\s*$', '', text, flags=re.MULTILINE)  # 删除章节编号
        text = re.sub(r'\s+\d{1,4}$', '', text, flags=re.MULTILINE)  # 删除行尾页码
        text = re.sub(r'^\s*(\d+\.){1,5}\s*', '', text, flags=re.MULTILINE)  # 删除多级编号
        text = re.sub(r'^\s*(Foreword|Contents|General|Scope|References|Introduction|Abbreviations|Definitions)\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)  # 删除章节标题
        text = re.sub(r'^[-=_]{3,}$', '', text, flags=re.MULTILINE)  # 删除分隔线
        text = re.sub(r'\n{3,}', '\n\n', text)  # 压缩多余空行
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)  # 删除行尾空格
        text = re.sub(r'\s{2,}', ' ', text)  # 压缩连续空格
        return text.strip()

    def chunks_split(self, text: str) -> List[str]:
        """将文本分割为指定大小的块
        Args:
            text: 清洗后的完整文本
        Returns:
            文本块列表，每个块约800字符，块间重叠100字符
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return splitter.split_text(text)

    def chunks_tokenize(self, chunks: List[str], batch_size=256) -> np.ndarray:
        """将文本块批量转换为向量
        Args:
            chunks: 文本块列表
            batch_size: 批量处理大小（防止内存溢出）
        Returns:
            形状为(n_chunks, embedding_dim)的向量矩阵
        """
        all_vecs = []
        # 分批次处理文本块
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            vecs = self.model.embed_documents(batch)
            all_vecs.append(vecs)
        return np.vstack(all_vecs)

    def process_documents(self) -> None:
        """文档处理主流程：遍历目录、处理文档、更新索引"""
        processed = 0
        # 递归遍历文档目录
        for file_path in self.docs_path.rglob('*'):
            # 跳过非支持格式文件
            if not file_path.is_file() or file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                continue
            try:
                print(f"Processing {file_path} ...")
                docs = self.read_docs(file_path)
                document_chunks = []
                chunk_hashes = []

                # 处理单个文档的多个页面内容
                for doc in docs:
                    cleaned_text = self.docs_clean(doc.page_content)
                    chunks = self.chunks_split(cleaned_text)
                    # 计算哈希值进行重复检测
                    for chunk in chunks:
                        h = hashlib.md5(chunk.encode('utf-8')).hexdigest()
                        if h in self.hash_set:
                            print(f"Duplicate chunk detected in {file_path}, skipping entire document.")
                            raise ValueError("Duplicate document")
                        document_chunks.append(chunk)
                        chunk_hashes.append(h)

                # 将有效块添加到索引系统
                if document_chunks:
                    vectors = self.chunks_tokenize(document_chunks)
                    # 初始化索引（如果不存在）
                    if self.index is None:
                        dim = vectors.shape[1]
                        self.index = faiss.IndexFlatL2(dim)
                    self.index.add(vectors.astype('float32'))

                    # 保存元数据和哈希值
                    for i, chunk in enumerate(document_chunks):
                        self.metadata_list.append({
                            "file": str(file_path.relative_to(self.docs_path)),
                            "chunk_idx": i,
                            "content_preview": chunk[:40]
                        })
                        self.hash_set.add(chunk_hashes[i])
                    processed += 1

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

        # 处理完成后持久化存储
        if self.index is not None:
            faiss.write_index(self.index, str(self.db_path))
            pickle.dump(self.metadata_list, open(self.meta_path, "wb"))
            pickle.dump(self.hash_set, open(self.hash_path, "wb"))
            print(f"All done! Processed {processed} documents. Index size: {self.index.ntotal}.")
        else:
            print("No documents processed!")

if __name__ == "__main__":
    """主程序：初始化嵌入模型并启动文档处理"""
    embed_model = HuggingFaceEmbeddings(
        model_name='../all-MiniLM-L6-v2',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    processor = DocumentProcessor(
        hug_embed=embed_model,
        docs_path="./documents",
        db_path="./faiss_3gpp.index"
    )
    processor.process_documents()