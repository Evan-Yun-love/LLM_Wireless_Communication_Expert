from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from docs_preprocess import clean_docs, chunks_split, chunks_tokenize, faiss_index

model_dir = "../deepseek-llm-7b-chat"

model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.1, max_length=1024, do_sample=True)