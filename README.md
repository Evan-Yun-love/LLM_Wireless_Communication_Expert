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
```
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
```
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
```
def chunks_split(self, text: str) -> List[str]:
        """将文本分割为指定大小的块
        Args:
            text: 清洗后的完整文本
        Returns:
            文本块列表，每个块约800字符，块间重叠100字符
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return splitter.split_text(text)
```



