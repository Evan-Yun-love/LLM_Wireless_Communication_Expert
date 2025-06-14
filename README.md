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

# 一、文档处理
## 📄 文档智能分块处理管道

这是一个用于**批量处理、清洗和分块PDF/DOCX文档**的Python管道，可为大语言模型RAG检索、知识库构建、信息检索等任务高效准备数据。

## 功能亮点

* 📂 **递归目录扫描**，自动查找所有PDF/DOCX文档
* 📑 **按页提取文档内容**（支持PDF和Word）
* 🧹 **多级文本清洗**，去除噪声、格式统一
* ✂️ **自定义分块**，支持块大小与重叠设置
* 🔖 **丰富的元数据**（文档名、页码、块序号）
* 💾 **支持分块数据与元数据的保存/加载（JSON格式）**

## 环境依赖

```bash
pip install langchain langchain_community pymupdf docx2txt
```

> 需要依赖 `PyMuPDF`（PDF支持）、`docx2txt`（DOCX支持）

## 用法示例

```python
from document_processor import DocumentProcessor

# 1. 初始化处理器，指定文档目录
proc = DocumentProcessor(docs_path="./docs")

# 2. 一键执行：加载文档 -> 清洗文本 -> 分块处理
proc.run_all()

# 3. 保存分块结果与元数据
proc.save_chunks_and_metadata("chunks.json")

# --- 如需下游复用：
proc.load_chunks_and_metadata("chunks.json")
print(proc.chunks[0])
print(proc.chunk_metadata[0])
```

### 目录结构示例

```
project/
├── docs/
│   ├── 文件1.pdf
│   └── 文件2.docx
├── document_processor.py
├── your_script.py
└── README.md
```

## 参数说明

| 方法                         | 参数名             | 含义          | 默认值 |
| -------------------------- | --------------- | ----------- | --- |
| `split_documents`          | `chunk_size`    | 每个文本块的目标字符数 | 800 |
| `split_documents`          | `chunk_overlap` | 块间重叠的字符数    | 160 |
| `save_chunks_and_metadata` | `save_path`     | 输出JSON文件路径  | -   |
| `load_chunks_and_metadata` | `load_path`     | 加载JSON文件路径  | -   |

## 输出数据格式

保存的JSON（如 `chunks.json`）内容示例：

```json
[
  {
    "chunk": "某个文本块的内容……",
    "metadata": {
      "document": "docs/文件1.pdf",
      "page": 3,
      "chunk_idx": 0
    }
  },
  ...
]
```

## 文本清洗细节

* 统一换行、空格与特殊字符
* 移除：

  * 各类3GPP/电信文档头/尾、章节号、页码、常见标题
  * URL、邮箱、电话/传真等联系方式
  * 非ASCII字符、冗余空行空格等噪声内容
* 清洗逻辑可在`clean_text()`方法中自定义扩展

# 二、向量化&检索
## 🔎 VectorStore 向量库管理与检索工具

`VectorStore` 是一个支持**文本向量化、FAISS 向量索引构建与高效检索**的Python工具类，适用于大规模文档、知识库、RAG检索增强等应用场景。支持多种索引类型与灵活的嵌入模型接入。

## ✨ 主要功能

- 🤖 **自动文本向量化**（支持HuggingFace本地/云模型）
- ⚡ **FAISS多索引类型管理**：`IndexFlatL2`、`IndexFlatIP`、`IVFFlat`、`IVFPQ`
- 📄 **支持元数据存储与检索**（每条文本配合自定义metadata）
- 🏷️ **向量与索引文件可持久化、加载**
- 🔍 **相似度检索、分数排序、过滤**
- 🚦 **灵活的分数转换与模式选择**

## 🛠️ 环境依赖

```bash
pip install faiss-cpu numpy langchain
# 若需GPU支持可用 faiss-gpu
# 若需中文嵌入推荐结合: transformers, sentence-transformers
````

## 🚀 用法示例

### 1. 初始化与文档/元数据加载

```python
from vector_store import VectorStore

embedding_model = "BAAI/bge-base-zh-v1.5"
vector_db = VectorStore(model_path=embedding_model, db_path="faiss.index", embedding_device="cuda")

# 加载 JSON 数据（格式见下文）
vector_db.load_documents_and_metadata("chunks.json")
```

### 2. 构建向量索引

```python
vector_db.build_from_documents(
    documents=vector_db.documents,
    metadata=vector_db.metadata,
    index_type="flatl2"   # 可选 flatl2, flatip, ivfflat, ivfpq
)
vector_db.persist()  # 持久化索引文件
```

### 3. 检索

```python
query = "中国的5G标准有哪些关键内容？"
results = vector_db.search(query, k=3)
for chunk, score, meta in results:
    print(f"得分: {score:.4f} | 来源: {meta['document']} 第{meta['page']}页")
    print(chunk[:100])
```

### 4. 加载/保存向量与索引

```python
vector_db.save_vectors("vectors.npy")
vector_db.load_vectors("vectors.npy")
vector_db.load_vector("faiss.index")
```

### 5. 描述当前向量库状态

```python
vector_db.describe()
```

## 📁 数据格式说明

* 输入JSON需为如下结构（建议与前述文档分块器配合）：

```json
[
  {
    "chunk": "文本内容……",
    "metadata": {
      "document": "docs/xx.pdf",
      "page": 3,
      "chunk_idx": 0
    }
  }
]
```

## ⚙️ 方法与参数说明

| 方法                            | 参数                                                            | 说明                              |
| ----------------------------- | ------------------------------------------------------------- | ------------------------------- |
| `__init__`                    | model\_path, db\_path, embedding\_device                      | 初始化，指定嵌入模型、索引文件路径、推理设备          |
| `load_documents_and_metadata` | json\_path                                                    | 加载预处理好的文档及元数据（JSON文件）           |
| `build_from_documents`        | documents, metadata, index\_type, n\_list, pq\_m, batch\_size | 构建嵌入向量并创建索引，支持指定类型和批量           |
| `rebuild_index_from_vectors`  | index\_type, n\_list, pq\_m                                   | 基于已存在向量重新构建指定类型索引               |
| `persist`                     | 无                                                             | 保存当前FAISS索引和meta信息到磁盘           |
| `load_vector`                 | db\_path                                                      | 加载已保存的FAISS索引文件                 |
| `save_vectors`                | path                                                          | 保存向量矩阵为npy文件                    |
| `load_vectors`                | path                                                          | 加载npy格式的向量矩阵                    |
| `search`                      | query, k, score\_mode, min\_score                             | 对输入query进行相似度检索，返回（文本、分数、元数据）列表 |
| `describe`                    | 无                                                             | 输出当前数据库/索引的简要信息                 |

> **备注：**
>
> * `index_type` 支持 `flatl2`（欧式距离）、`flatip`（内积/余弦）、`ivfflat`、`ivfpq`
> * `score_mode` 支持 `reciprocal`（1/(1+距离)）、`negative`（-距离）、`linear`（1-距离）
> * `metadata` 为每条文本块的自定义信息（如文件、页码、块号等）

## 三、Prompt构建

## 四、LLM微调

## 五、性能评估

## 遇到的一些问题，以及解决方案
1. 测试集的构建，人工还是自动，自动的话，调API，把chunk喂给llm，让它生成对应的提问和回答
2. 专业术语处理，一些书里面没有专业术语的全称说明，llm没有这方面知识，会乱编。解决方案就是把专业术语进行扩展

