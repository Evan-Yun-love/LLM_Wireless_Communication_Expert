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

## 一、构建通信知识库
1. 收集文档：  
收集常用5G NR 3GPP协议以及通信电子书

![image](https://github.com/user-attachments/assets/c488cfbc-df9a-43ed-8a4b-36b121cb7cac)

2. 文档读取：  
对于docs，用`Docx2txtLoader`读取; 对于pdf，用`PyMuPDFLoader`读取。目前仅支持这两种格式
```python
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
```

3. 数据清洗：  
用正则表达式对文档进行清洗。常见噪声主要是：技术规范编号、商标信息、联系方式、链接、联系方式、版权声明、标题、章节号、页码、页眉页脚等等。因为文档比较少，格式比较固定，所以数据清洗让GPT代劳即可。
```python
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
```
4. 文本切割：  
由于技术文档上下文关系较强，需要尽量保留段落、句子完整，减少截断，所以使用`RecursiveCharacterTextSplitter`。  
关于```chunk_size```和```chunk_overlap```的设置，chunk_size的设置要考虑使用的embed_model的大小,因为算力有限，目前采用的模型是```all-MiniLM-L6-v2```，模型比较小，编码维度```dim=384```，所以不宜设置太大的chunk_size，否则向量难以表征chunk语义。所以设置是```chunk_size=800, chunk_overlap=160```，chunk_overlap按照chunk_size的``20%``来计。  
```python
def chunks_split(self, text: str) -> List[str]:
        """将文本分割为指定大小的块
        Args:
            text: 清洗后的完整文本
        Returns:
            文本块列表，每个块约800字符，块间重叠100字符
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=160)
        return splitter.split_text(text)
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

## 三、Prompt构建

## 四、LLM微调

## 五、性能评估


