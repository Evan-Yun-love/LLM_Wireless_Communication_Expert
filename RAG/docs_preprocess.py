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

    def __init__(self, docs_path: Union[str, Path], chunk_size: int = 800, chunk_overlap: int = 160, min_length = 50):
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
        self.chunks_size = chunk_size
        self.chunks_overlap = chunk_overlap
        self.min_length = min_length

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
        self.cleaned_documents = []
        for doc in self.documents:
            cleaned = self.clean_text(doc.content)
            if cleaned.strip():  # 只保留清洗后内容不为空的页面
                self.cleaned_documents.append(PageDocument(content=cleaned, metadata=doc.metadata))

    def split_documents(self) -> None:
        """
        将清洗后的文档分割成文本块
        
        参数:
            chunk_size: 每个文本块的目标字符数
            chunk_overlap: 块间重叠的字符数
            
        使用递归字符分割器处理每个文档，记录分块内容及其元数据
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunks_size, 
            chunk_overlap=self.chunks_overlap,
            separators=["\n\n", "\n", ".", " ", ""],)
        
        skipped = 0
        for doc in self.cleaned_documents:
            chunks = splitter.split_text(doc.content)
            # 为每个块添加索引信息到元数据
            for i, chunk in enumerate(chunks):
                if len(chunk.strip())<self.min_length:
                    skipped += 1
                    continue
                self.chunks.append(chunk)
                meta = dict(doc.metadata)
                meta["chunk_idx"] = i
                self.chunk_metadata.append(meta)
        if skipped>0:
            print(f"[INFO]共跳过{skipped}个过短的chunk")
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
    def clean_text(text: str, page_num: int = 10) -> str:
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
        
        if page_num is not None and page_num <= 5:
            irrelevant_patterns = [
                r'^\s*(Table of Contents|Contents|目录)\s*$',    
                r'^\s*(Acknowledgements?|致谢)\s*$',             
                r'^\s*(Preface|Foreword|序言|前言)\s*$',         
                r'^\s*(Index|Appendix|附录)\s*$',               
            ]
            for pat in irrelevant_patterns:
                if re.search(pat, text, re.IGNORECASE | re.MULTILINE):
                    return ""
            # 目录点线页过滤（仅前几页判定，且比例设高点）
            dot_leader_lines = re.findall(r'^\s*[\d\.]{1,7}.*\.{3,}.*\d+\s*$', text, flags=re.MULTILINE)
            if len(text.splitlines()) > 0 and len(dot_leader_lines) / len(text.splitlines()) >= 0.8 and len(dot_leader_lines) >= 5:
                return ""

        # 统一换行符和空格
        text = re.sub(r'(?<=\b)(?:[a-zA-Z]\s){2,}[a-zA-Z](?=\b)', lambda m: m.group(0).replace(' ', ''), text)
        text = re.sub(r'(?:^.{1,20}\n){3,}', lambda m: m.group(0).replace('\n', ' '), text, flags=re.MULTILINE)
        
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
    

if __name__ == "__main__":
    dp = DocumentProcessor(docs_path="./documents", chunk_size=2000, chunk_overlap = 200)
    dp.run_all()
    dp.save_chunks_and_metadata(save_path="chunks_size_2000_overlap_200_min_50.json")