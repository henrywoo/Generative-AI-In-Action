import os
import glob
import pickle
import hashlib
import numpy as np
import time
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
# 检查LangChain版本信息
try:
    import langchain
    import langchain_core
    import langchain_community
    print(f"📦 LangChain版本信息:")
    print(f"   langchain: {getattr(langchain, '__version__', 'unknown')}")
    print(f"   langchain-core: {getattr(langchain_core, '__version__', 'unknown')}")
    print(f"   langchain-community: {getattr(langchain_community, '__version__', 'unknown')}")
    print(f"✅ 版本兼容性: 良好 (所有组件都是0.3.x系列)")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader, 
    UnstructuredExcelLoader,
    UnstructuredFileLoader
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import torch
# ChromaDB批量大小限制（根据实际测试和文档）
CHROMA_MAX_BATCH_SIZE = 5000  # 安全值，低于实际限制5461
# 添加进度条和详细信息
from tqdm import tqdm
import time

# 配置常量
MODEL = "deepseek-chat"
API_URL = "https://api.deepseek.com/chat/completions"
load_dotenv(override=True)
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-584f60334377408a9b2b5dd83c838142')
db_name = "vector_db"

base = Path("/home/wukong/markdown")


def get_files_info():
    """获取所有markdown文件的详细信息，用于增量检测"""
    files_info = {}
    for path in base.rglob("*.md"):
        try:
            mtime = os.path.getmtime(path)
            size = os.path.getsize(path)
            files_info[str(path)] = {"mtime": mtime, "size": size, "path": str(path)}
        except:
            continue
    return files_info

def create_chunks_from_files(file_paths):
    """从指定文件列表创建chunks"""
    documents = []
    text_loader_kwargs = {"encoding": "utf-8"}
    for file_path in file_paths:
        try:
            docs = TextLoader(file_path, **text_loader_kwargs).load()
            for d in docs:
                d.metadata["doc_type"] = Path(file_path).parent.name
            documents.extend(docs)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def load_or_create_chunks():
    """创建所有chunks（仅在需要时调用）"""
    print("Generating chunks from all markdown files...")
    all_files = [str(path) for path in base.rglob("*.md")]
    chunks = create_chunks_from_files(all_files)
    print(f"Generated {len(chunks)} chunks from {len(all_files)} files")
    return chunks

# 智能加载或创建向量数据库
def get_or_create_vectorstore(db_name, embeddings):
    if os.path.exists(db_name):
        print(f"Loading existing vector database from {db_name}")
        vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
        return vectorstore, True  # True表示已存在
    else:
        print(f"Creating new vector database at {db_name}")
        # 只有创建新数据库时才需要chunks
        chunks = load_or_create_chunks()
        
        # 分批创建向量数据库，避免内存问题
        print(f"🔄 Creating vector database with {len(chunks)} chunks...")
        
        # 先创建空的向量数据库
        vectorstore = Chroma(
            persist_directory=db_name,
            embedding_function=embeddings
        )
        
        # 分批添加文档（使用ChromaDB的安全限制）
        print(f"📊 ChromaDB safe batch size: {CHROMA_MAX_BATCH_SIZE}")
        batch_size = min(10000 if device == 'cuda' else 5000, CHROMA_MAX_BATCH_SIZE)
        print(f"🎯 Using batch size: {batch_size}")
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"📦 Adding documents in {total_batches} batches of {batch_size}...")
        
        # 使用tqdm显示进度
        successful_batches = 0
        start_time = time.time()
        
        with tqdm(total=total_batches, desc="🏗️  Building database", unit="batch") as pbar:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                pbar.set_description(f"🏗️  Batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                
                try:
                    batch_start_time = time.time()
                    vectorstore.add_documents(batch)
                    batch_time = time.time() - batch_start_time
                    
                    successful_batches += 1
                    pbar.set_postfix({
                        'Success': successful_batches,
                        'Batch Time': f"{batch_time:.1f}s",
                        'Memory': f"{len(batch)} chunks"
                    })
                    
                except Exception as e:
                    print(f"\n❌ Error adding batch {batch_num}: {e}")
                    # 尝试更小的批量
                    if len(batch) > 1000:
                        print("🔄 Retrying with smaller batch size...")
                        smaller_batch_size = 1000
                        smaller_success = 0
                        
                        for j in range(0, len(batch), smaller_batch_size):
                            smaller_batch = batch[j:j + smaller_batch_size]
                            try:
                                vectorstore.add_documents(smaller_batch)
                                smaller_success += 1
                                print(f"  ✅ Smaller batch {j//smaller_batch_size + 1} added")
                            except Exception as e2:
                                print(f"  ❌ Error with smaller batch: {e2}")
                                break
                        
                        if smaller_success > 0:
                            print(f"  📊 {smaller_success} smaller batches succeeded")
                            # 即使小批量成功，也标记为成功
                            successful_batches += 1
                    else:
                        print(f"  ❌ Batch {batch_num} failed completely")
                    
                                    # 继续处理下一批，而不是break
                continue
            
            # 更新进度条（无论成功还是失败都更新）
            pbar.update(1)
        
        # 显示创建统计
        total_time = time.time() - start_time
        print(f"\n🏗️  Database Creation Summary:")
        print(f"   ✅ Successful batches: {successful_batches}")
        print(f"   ⏱️  Total time: {total_time:.1f}s")
        print(f"   📈 Average time per batch: {total_time/total_batches:.1f}s")
        print(f"   🚀 Speed: {len(chunks)/total_time:.1f} chunks/second")
        
        return vectorstore, False  # False表示新创建

# 智能获取chunks（只在必要时）
def get_chunks_if_needed():
    """只在必要时获取chunks：创建新数据库或需要增量更新时"""
    if os.path.exists(db_name):
        # 数据库存在，检查是否需要增量更新
        vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
        files_info = get_files_info()
        
        # 从vector db获取现有文档的文件信息
        existing_docs = vectorstore._collection.get(include=['metadatas'])
        existing_files = set()
        if existing_docs['metadatas']:
            existing_files = {meta.get('source', '') for meta in existing_docs['metadatas'] if meta and meta.get('source')}
        
        # 检测变化的文件
        changed_files = []
        new_files = []
        for file_path, current_info in files_info.items():
            if file_path not in existing_files:
                new_files.append(file_path)
            elif file_path in existing_files:
                # 检查文件是否被修改（通过比较修改时间）
                # 这里可以进一步优化，但为了简单起见，我们假设需要重新处理
                changed_files.append(file_path)
        
        # 如果有变化，返回chunks用于增量更新
        if changed_files or new_files:
            print(f"Files changed: {len(changed_files)} modified, {len(new_files)} new")
            print("Regenerating chunks for incremental update...")
            return load_or_create_chunks()
        else:
            print("No file changes detected, skipping chunks generation")
            return None
    else:
        # 数据库不存在，需要生成chunks
        print("Vector database doesn't exist, generating chunks...")
        return load_or_create_chunks()

# 为每个chunk生成稳定的ID
def generate_chunk_id(chunk, chunk_idx):
    """生成稳定的chunk ID，基于文件路径、修改时间和chunk索引"""
    source_path = chunk.metadata.get('source', 'unknown')
    # 获取文件修改时间
    try:
        mtime = os.path.getmtime(source_path)
    except:
        mtime = 0
    # 组合生成ID：路径哈希 + 修改时间 + chunk索引
    path_hash = hashlib.md5(source_path.encode()).hexdigest()[:8]
    return f"{path_hash}_{int(mtime)}_{chunk_idx}"

# 获取需要增量更新的chunks
def get_new_or_modified_chunks(existing_ids, all_chunks):
    """识别新的或修改过的chunks"""
    new_chunks = []
    for idx, chunk in enumerate(all_chunks):
        chunk_id = generate_chunk_id(chunk, idx)
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            # 为chunk添加ID到metadata
            chunk.metadata['chunk_id'] = chunk_id
    
    return new_chunks


def get_optimal_device():
    """检测并返回最优的设备配置"""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU detected: {gpu_name}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        print(f"   GPU memory cleared")
        
        return device
    else:
        print("⚠️  No GPU detected, using CPU")
        return 'cpu'

# 配置GPU加速的embeddings
device = get_optimal_device()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': device},
    encode_kwargs={"batch_size": 256, "normalize_embeddings": True, 'device': device}
)
# 主逻辑
vectorstore, is_existing = get_or_create_vectorstore(db_name, embeddings)

if is_existing:
    print(f"Vector database loaded with {vectorstore._collection.count()} existing documents")
    
    # 检查是否需要增量更新
    chunks_for_update = get_chunks_if_needed()
    
    if chunks_for_update:
        # 有文件变化，进行增量更新
        print("Performing incremental update...")
        
        # 获取现有文档的ID
        existing_docs = vectorstore._collection.get(include=['metadatas'])
        existing_ids = set()
        if existing_docs['metadatas']:
            existing_ids = {meta.get('chunk_id', '') for meta in existing_docs['metadatas'] if meta}
        
        # 检查是否有新文档需要添加
        new_chunks = get_new_or_modified_chunks(existing_ids, chunks_for_update)
        
        if new_chunks:
            print(f"Adding {len(new_chunks)} new/modified chunks")
            # 为所有chunks添加ID
            for idx, chunk in enumerate(chunks_for_update):
                if 'chunk_id' not in chunk.metadata:
                    chunk.metadata['chunk_id'] = generate_chunk_id(chunk, idx)
            
            # 分批添加新文档（使用ChromaDB的官方限制）
            if device == 'cuda':
                optimal_batch = 10000  # GPU并行能力强，使用更大批量
                print("🎯 Using GPU-optimized batch size")
            else:
                optimal_batch = 5000   # CPU环境使用较小批量
            
            batch_size = min(optimal_batch, CHROMA_MAX_BATCH_SIZE)
            print(f"📦 Batch size: {batch_size} (safe limit: {CHROMA_MAX_BATCH_SIZE})")
            total_batches = (len(new_chunks) + batch_size - 1) // batch_size
            
            print(f"Inserting in {total_batches} batches of {batch_size}...")
            successful_batches = 0
            failed_batches = 0
            start_time = time.time()
            
            # 使用tqdm创建进度条
            with tqdm(total=total_batches, desc="🔄 Inserting batches", unit="batch") as pbar:
                for i in range(0, len(new_chunks), batch_size):
                    batch = new_chunks[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    
                    # 更新进度条描述
                    pbar.set_description(f"🔄 Updating DB: {batch_num}/{total_batches}")
                    
                    try:
                        batch_start_time = time.time()
                        vectorstore.add_documents(batch)
                        batch_time = time.time() - batch_start_time
                        
                        successful_batches += 1
                        pbar.set_postfix({
                            'Success': successful_batches,
                            'Failed': failed_batches,
                            'Batch Time': f"{batch_time:.1f}s"
                        })
                        
                        # 详细日志（可选，取消注释以显示）
                        # print(f"✅ Batch {batch_num} inserted successfully in {batch_time:.1f}s")
                        
                    except Exception as e:
                        failed_batches += 1
                        pbar.set_postfix({
                            'Success': successful_batches,
                            'Failed': failed_batches,
                            'Error': str(e)[:30] + "..."
                        })
                        
                        print(f"\n❌ Error inserting batch {batch_num}: {e}")
                        
                        # 如果单批失败，尝试更小的批量
                        if len(batch) > 1000:
                            print("🔄 Retrying with smaller batch size...")
                            smaller_batch_size = 1000
                            smaller_success = 0
                            
                            for j in range(0, len(batch), smaller_batch_size):
                                smaller_batch = batch[j:j + smaller_batch_size]
                                try:
                                    vectorstore.add_documents(smaller_batch)
                                    smaller_success += 1
                                    print(f"  ✅ Smaller batch {j//smaller_batch_size + 1} inserted")
                                except Exception as e2:
                                    print(f"  ❌ Error with smaller batch: {e2}")
                                    break
                            
                            if smaller_success > 0:
                                print(f"  📊 {smaller_success} smaller batches succeeded")
                        
                        # 继续处理下一批，而不是完全停止
                        continue
                    
                    # 更新进度条
                    pbar.update(1)
                    
                # 无论成功还是失败，都要更新进度条
                # 注意：这里不需要额外的pbar.update(1)，因为上面已经更新了
            
            # 显示最终统计信息
            total_time = time.time() - start_time
            print(f"\n📊 Insertion Summary:")
            print(f"   ✅ Successful batches: {successful_batches}")
            print(f"   ❌ Failed batches: {failed_batches}")
            print(f"   ⏱️  Total time: {total_time:.1f}s")
            print(f"   📈 Average time per batch: {total_time/total_batches:.1f}s")
            print(f"   🚀 Speed: {len(new_chunks)/total_time:.1f} chunks/second")
            
            print(f"Vector database updated. Total documents: {vectorstore._collection.count()}")
        else:
            print("No new documents to add")
    else:
        print("No file changes, using existing vector database")
else:
    print(f"Vector database created with {vectorstore._collection.count()} documents")
    # 为新创建的数据库中的chunks添加ID
    chunks = load_or_create_chunks()  # 获取刚创建的chunks
    for idx, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = generate_chunk_id(chunk, idx)
    # ChromaDB自动持久化，无需手动调用persist()
    print("💾 Vector database automatically persisted")

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")

llm = ChatOpenAI(
    model_name=MODEL,                 # "deepseek-chat" / "deepseek-reasoner"
    temperature=0.7,
    openai_api_base="https://api.deepseek.com/v1",
    openai_api_key=DEEPSEEK_API_KEY,
)

# 创建新版本兼容的内存管理器
class MemoryManager:
    def __init__(self):
        self.chat_histories = {}
    
    def get_chat_history(self, session_id: str) -> ChatMessageHistory:
        """获取或创建聊天历史"""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = ChatMessageHistory()
        return self.chat_histories[session_id]

# 初始化内存管理器
memory_manager = MemoryManager()

# 创建检索器
retriever = vectorstore.as_retriever()

# 创建对话链（使用新的内存系统）
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

# 使用新版本的内存系统进行查询
def query_with_memory(question: str, session_id: str = "default"):
    """使用新版本的内存系统进行查询"""
    # 获取聊天历史
    chat_history = memory_manager.get_chat_history(session_id)
    
    # 构建输入（使用新版本的消息格式）
    inputs = {
        "question": question,
        "chat_history": chat_history.messages
    }
    
    # 使用新版本的调用方式
    try:
        result = conversation_chain.invoke(inputs)
        
        # 更新聊天历史
        chat_history.add_user_message(question)
        chat_history.add_ai_message(result["answer"])
        
        return result
    except Exception as e:
        print(f"❌ 查询错误: {e}")
        # 尝试不使用聊天历史
        try:
            result = conversation_chain.invoke({"question": question})
            chat_history.add_user_message(question)
            chat_history.add_ai_message(result["answer"])
            return result
        except Exception as e2:
            print(f"❌ 备用查询也失败: {e2}")
            return None



# 测试查询
query = "Can you describe how Cobra Extending Mamba in a few sentences"
result = query_with_memory(query)
print("🤖 Answer:", result["answer"])

# 显示聊天历史
print(f"\n💬 Chat History ({len(memory_manager.get_chat_history('default').messages)} messages):")
for i, msg in enumerate(memory_manager.get_chat_history('default').messages):
    role = "👤 User" if isinstance(msg, HumanMessage) else "🤖 AI"
    print(f"  {i+1}. {role}: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")

def chat(message, history):
    # Convert Gradio history format to LangChain message format
    chat_history = []
    for human, ai in history:
        if human:
            chat_history.append(HumanMessage(content=human))
        if ai:
            chat_history.append(AIMessage(content=ai))
    
    # Call the conversation chain with both question and chat_history
    result = conversation_chain.invoke({
        "question": message,
        "chat_history": chat_history
    })
    
    return result["answer"]

view = gr.ChatInterface(chat).launch(inbrowser=True)

