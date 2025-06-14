# **LLM_Wireless_Communication_Expert**

🎯**实现功能：**
1. 回答通信领域知识

🚩**实现步骤:**
1. 获取通信知识文档：3GPP协议、通信电子书
2. 文档读取，数据清洗（用正则），向量化
3. 构建通信RAG知识库
4. 根据通信场景，设计提问Propmt、模版
5. 进行微调，使得模型回答更像通信专家
6. 增加通信公式计算，将一些公式封装为函数，根据提示以及给定参数进行计算

## 系统框图
![系统框图](./LWCE系统框图.png)

## 一、文档处理
1. 收集文档：  
收集常用5G NR 3GPP协议以及通信电子书

![image](https://github.com/user-attachments/assets/c488cfbc-df9a-43ed-8a4b-36b121cb7cac)

2. 文档读取：  
对于docs，用`Docx2txtLoader`读取; 对于pdf，用`PyMuPDFLoader`读取。目前仅支持这两种格式

3. 数据清洗：  
用正则表达式对文档进行清洗。常见噪声主要是：技术规范编号、商标信息、联系方式、链接、联系方式、版权声明、标题、章节号、页码、页眉页脚等等。因为文档比较少，格式比较固定，所以数据清洗让GPT代劳即可。

4. 文本切割：  
由于技术文档上下文关系较强，需要尽量保留段落、句子完整，减少截断，所以使用`RecursiveCharacterTextSplitter`。  
关于```chunk_size```和```chunk_overlap```的设置，chunk_size的设置要考虑使用的embed_model的大小,因为算力有限，目前采用的模型是```all-MiniLM-L6-v2```，模型比较小，编码维度```dim=384```，所以不宜设置太大的chunk_size，否则向量难以表征chunk语义。所以设置是```chunk_size=800, chunk_overlap=160```，chunk_overlap按照chunk_size的``20%``来计。  

```python
import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class PageDocument:
    """
    表示单页文档的数据类
    
    属性:
        content: 文档页面的文本内容
        metadata: 包含文档元数据的字典（如文件路径、页码等）
    """
    content: str
    metadata: dict

class DocumentProcessor:
    """
    文档处理管道，支持PDF/DOCX格式的读取、文本清洗和分块处理
    
    主要功能:
        - 递归扫描目录获取支持格式的文件列表
        - 加载PDF/DOCX文档并提取每页内容
        - 对文本进行多级清洗处理
        - 将文档分割成指定大小的文本块
        - 支持保存/加载处理结果
    
    属性:
        SUPPORTED_FORMATS: 支持处理的文件扩展名集合
        docs_path: 文档存储的根目录路径
        files: 扫描到的有效文件路径列表
        documents: 原始页面文档对象列表
        cleaned_documents: 清洗后的页面文档对象列表
        chunks: 分割后的文本块内容列表
        chunk_metadata: 对应文本块的元数据列表
    """
    SUPPORTED_FORMATS = {'.pdf', '.docx'}

    def __init__(self, docs_path: Union[str, Path]):
        """
        初始化文档处理器
        
        参数:
            docs_path: 包含文档的目录路径
        """
        self.docs_path = Path(docs_path)
        self.files: List[str] = self._get_files()

        # 中间结果
        self.documents: List[PageDocument] = []
        self.cleaned_documents: List[PageDocument] = []
        self.chunks: List[str] = []
        self.chunk_metadata: List[dict] = []

    def _get_files(self) -> List[str]:
        """
        递归扫描文档目录，获取所有支持的文档文件路径
        
        返回:
            符合条件的文件绝对路径列表
        """
        file_list = []
        for filepath, _, filenames in os.walk(self.docs_path):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.SUPPORTED_FORMATS:
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def load_documents(self) -> None:
        """
        加载所有文档文件内容
        
        使用PyMuPDFLoader处理PDF文件，Docx2txtLoader处理DOCX文件
        将每页内容封装为PageDocument对象并存储原始文档列表
        """
        for file_path in self.files:
            try:
                # 根据扩展名选择加载器
                if file_path.endswith('.pdf'):
                    pages = [p.page_content for p in PyMuPDFLoader(file_path).load()]
                elif file_path.endswith('.docx'):
                    pages = [p.page_content for p in Docx2txtLoader(file_path).load()]
                else:
                    continue
                # 封装每页内容并记录元数据
                for idx, content in enumerate(pages, 1):
                    self.documents.append(PageDocument(content=content, metadata={"document": file_path, "page": idx}))
            except Exception as e:
                print(f"[WARN] 读取失败: {file_path}，错误: {e}")

    def clean_documents(self) -> None:
        """
        对原始文档进行文本清洗处理
        
        遍历所有PageDocument对象，应用clean_text方法清洗文本内容
        将清洗结果存储到cleaned_documents列表
        """
        for doc in self.documents:
            cleaned = self.clean_text(doc.content)
            self.cleaned_documents.append(PageDocument(content=cleaned, metadata=doc.metadata))

    def split_documents(self, chunk_size: int = 800, chunk_overlap: int = 160) -> None:
        """
        将清洗后的文档分割成文本块
        
        参数:
            chunk_size: 每个文本块的目标字符数
            chunk_overlap: 块间重叠的字符数
            
        使用递归字符分割器处理每个文档，记录分块内容及其元数据
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for doc in self.cleaned_documents:
            chunks = splitter.split_text(doc.content)
            # 为每个块添加索引信息到元数据
            for i, chunk in enumerate(chunks):
                self.chunks.append(chunk)
                meta = dict(doc.metadata)
                meta["chunk_idx"] = i
                self.chunk_metadata.append(meta)

    def run_all(self):
        """
        执行完整处理流程
        
        依次执行: 加载文档 -> 清洗文本 -> 分割文本
        打印各阶段处理统计信息
        """
        print(f"[INFO] 开始处理 {len(self.files)} 个文件...")
        self.load_documents()
        print(f"[INFO] 已加载 {len(self.documents)} 页")
        self.clean_documents()
        print(f"[INFO] 清洗完成")
        self.split_documents()
        print(f"[INFO] 分割完成，生成 chunk 数: {len(self.chunks)}")

    def save_chunks_and_metadata(self, save_path: Union[str, Path]) -> None:
        """
        保存分块结果和元数据到JSON文件
        
        参数:
            save_path: 结果保存路径
        """
        save_path = Path(save_path)
        # 组合分块内容和元数据
        data = [{"chunk": c, "metadata": m} for c, m in zip(self.chunks, self.chunk_metadata)]
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 已保存至: {save_path}")

    def load_chunks_and_metadata(self, load_path: Union[str, Path]) -> None:
        """
        从JSON文件加载分块结果和元数据
        
        参数:
            load_path: 结果文件路径
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"{load_path} 不存在")
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 从加载的数据中分离内容和元数据
        self.chunks = [item["chunk"] for item in data]
        self.chunk_metadata = [item["metadata"] for item in data]
        print(f"[INFO] 成功加载 {len(self.chunks)} 个 chunk")

    @staticmethod
    def clean_text(text: str) -> str:
        """
        执行多级文本清洗和标准化处理
        
        包括:
          - 换行符标准化
          - 特殊字符替换
          - 移除特定格式文本（如3GPP文档头、页码等）
          - 清除URL和电子邮件
          - 删除非ASCII字符
          - 多余空格和空行压缩
        
        参数:
            text: 原始文本字符串
        返回:
            清洗后的文本字符串
        """
        # 统一换行符和空格
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('–', '-').replace('—', '-').replace(' ', ' ')
        
        # 修复被空格分隔的单词
        text = re.sub(r'(?<=\b)(?:[a-zA-Z]\s){2,}[a-zA-Z](?=\b)', lambda m: m.group(0).replace(' ', ''), text)
        # 合并短行
        text = re.sub(r'(?:^.{1,20}\n){3,}', lambda m: m.group(0).replace('\n', ' '), text, flags=re.MULTILINE)
        
        # 移除特定格式文本
        text = re.sub(r'3GPP TS \d+\.\d+ V\d+\.\d+\.\d+ \(\d{4}-\d{2}\)', '', text)
        text = re.sub(r'Release \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = text.replace('\f', '')
        
        # 移除版权声明和联系信息
        text = re.sub(r'Copyright Notification.*?All rights reserved\.', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Postal address.*?http://www\.3gpp\.org', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'UMTS™.*?GSM® and the GSM logo.*?GSM Association', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 清除文档编号和章节标题
        text = re.sub(r'^\s*(\d+\.){1,4}[^\n]*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+\d{1,4}$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*(\d+\.){1,5}\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*(Foreword|Contents|General|Scope|References|Introduction|Abbreviations|Definitions)\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # 清理格式标记和多余空白
        text = re.sub(r'^[-=_]{3,}$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s{2,}', ' ', text)
        
        # 移除URL和网络地址
        text = re.sub(r'(https?://[^\s]+)', '', text)
        text = re.sub(r'(www\.[^\s]+)', '', text)
        text = re.sub(r'\b\S+\.(com|org|net|edu|gov|cn)\b', '', text)
        text = re.sub(r'[Ff]ax[:：]?\s*[\(\)\d\-– —]+', '', text)
        
        # 过滤非ASCII字符
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        return text.strip()

```

5. chunk向量化:  
采用的模型是```all-MiniLM-L6-v2```，比较小型化，直接部署本地。支持中英文。
```python
embed_model = HuggingFaceEmbeddings(
        model_name='../all-MiniLM-L6-v2',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
```
```python
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
```
6. 构建FAISS向量库：  
递归查询目录下的文档，读取pdf,docx，逐文档处理，同时加入重复检测功能，避免多次添加同一文档到向量库中，可以增量、持久化添加索引。
```python
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
```

## 二、向量检索
专业术语缩写问题处理

## 三、Prompt构建

## 四、LLM微调

## 五、性能评估

## 遇到的一些问题，以及解决方案
1. 测试集的构建，人工还是自动，自动的话，调API，把chunk喂给llm，让它生成对应的提问和回答
2. 专业术语处理，一些书里面没有专业术语的全称说明，llm没有这方面知识，会乱编。解决方案就是把专业术语进行扩展

