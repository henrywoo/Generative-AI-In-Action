#!/usr/bin/env python3
"""
RAG Demo - Production Ready Script
A production-ready script for building and querying vector databases from documents.
"""

import os
import glob
import pickle
import hashlib
import numpy as np
import time
import argparse
import traceback
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
try:
    import langchain
    import langchain_core
    import langchain_community
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
        UnstructuredFileLoader,
        PyPDFLoader,
        UnstructuredHTMLLoader
    )
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_deepseek import ChatDeepSeek
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    print(f"❌ Required LangChain packages not found: {e}")
    print("Please install: pip install langchain langchain-core langchain-community langchain-openai langchain-deepseek langchain-chroma langchain-huggingface")
    exit(1)

# Optional imports
try:
    import torch
    from tqdm import tqdm
    import gradio as gr
except ImportError as e:
    print(f"⚠️  Optional packages not found: {e}")
    print("Some features may not work properly")

# Precompiled patterns and sanitization utilities
_RE_ONLY_NUMS = re.compile(r'^[\d\s\.-]+$')
_RE_ONLY_SYMBOLS = re.compile(r'^[^\w\s]+$')
_CONTROL_REPLACEMENTS = {
    '\x00': ' ', '\x01': ' ', '\x02': ' ', '\x03': ' ', '\x04': ' ', '\x05': ' ',
    '\x06': ' ', '\x07': ' ', '\x08': ' ', '\x0b': ' ', '\x0c': ' ', '\x0e': ' ',
    '\x0f': ' ', '\x10': ' ', '\x11': ' ', '\x12': ' ', '\x13': ' ', '\x14': ' ',
    '\x15': ' ', '\x16': ' ', '\x17': ' ', '\x18': ' ', '\x19': ' ', '\x1a': ' ',
    '\x1b': ' ', '\x1c': ' ', '\x1d': ' ', '\x1e': ' ', '\x1f': ' ',
}

def sanitize_text(value: Any) -> Optional[str]:
    """Return a cleaned string or None if invalid, with enhanced PDF handling.

    - Filters None/NaN/null/empty
    - Strips control chars (including extended control chars)
    - Normalizes intra-line whitespace
    - Drops lines that are too short, numeric-only, or symbol-only
    - More aggressive filtering for problematic content
    - Enhanced PDF text cleaning
    """
    if value is None:
        return None
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        return None

    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    if text.lower() in ("nan", "none", "null"):
        return None

    # Replace control characters
    for k, v in _CONTROL_REPLACEMENTS.items():
        text = text.replace(k, v)
    
    # Additional PDF-specific cleaning
    # Remove extended control characters and non-printable characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', ' ', text)
    
    # Remove common PDF text artifacts and keep only printable characters
    text = re.sub(r'[^\x20-\x7E\u00A0-\uFFFF]', ' ', text)

    # Process per-line to preserve line boundaries
    lines = text.splitlines()
    filtered_lines: List[str] = []
    for line in lines:
        line = re.sub(r'[ \t\f\v]+', ' ', line).strip()
        if not line:
            continue
            
        # More aggressive line filtering for PDF content
        if len(line) < 3:
            continue
            
        # Skip lines that are mostly non-alphabetic (but allow markdown headers)
        alpha_chars = sum(1 for c in line if c.isalpha())
        if alpha_chars < len(line) * 0.15:  # Reduced to 15% for markdown content
            continue
            
        # Skip lines that are only numbers, symbols, or special characters
        if _RE_ONLY_NUMS.match(line):
            continue
        if _RE_ONLY_SYMBOLS.match(line):
            continue
            
        # Skip lines that are too long (likely corrupted)
        if len(line) > 1000:
            continue
            
        filtered_lines.append(line)

    cleaned = '\n'.join(filtered_lines).strip()
    if not cleaned or len(cleaned) < 10:
        return None
    if cleaned.lower() in ("nan", "none", "null"):
        return None
    
    # Additional aggressive filtering for problematic content
    # Check if the cleaned text contains mostly control characters or is too short
    if len(cleaned) < 15:  # Reduced minimum length requirement for PDF
        return None
    
    # Check if text contains too many control characters
    control_char_count = sum(1 for c in cleaned if ord(c) < 32)
    if control_char_count > len(cleaned) * 0.05:  # More than 5% control chars
        return None
    
    # Check if text is mostly whitespace or control characters
    printable_chars = sum(1 for c in cleaned if c.isprintable() and not c.isspace())
    if printable_chars < len(cleaned) * 0.4:  # Less than 40% printable chars
        return None
        
    # Additional PDF-specific checks
    # Check for excessive special characters
    special_chars = sum(1 for c in cleaned if c in '!@#$%^&*()_+-=[]{}|;:,.<>?')
    if special_chars > len(cleaned) * 0.3:  # More than 30% special chars
        return None
    
    return cleaned

def filter_and_clean_chunks(chunks: List[Document]) -> List[Document]:
    """Clean and validate chunks in-place using sanitize_text, returning valid ones."""
    print(f"🔍 Validating and cleaning {len(chunks)} chunks...")
    valid: List[Document] = []
    for i, chunk in enumerate(chunks):
        if chunk is None:
            print(f"⚠️  Invalid chunk {i}: chunk is None")
            continue
        if not hasattr(chunk, 'page_content'):
            print(f"⚠️  Invalid chunk {i}: missing page_content attribute")
            continue
        
        # Get file source for better debugging
        file_source = chunk.metadata.get('source', 'unknown') if hasattr(chunk, 'metadata') else 'unknown'
        
        cleaned = sanitize_text(chunk.page_content)
        if not cleaned:
            preview = None if chunk.page_content is None else repr(chunk.page_content)[:100]
            print(f"⚠️  Dropped chunk {i}: invalid after cleaning, file: {file_source}")
            print(f"   Content preview: {preview}")
            continue
        
        # Final safety check: ensure the cleaned content is actually valid for tokenization
        if not _is_safe_for_tokenization(cleaned):
            preview = repr(cleaned)[:100]
            print(f"⚠️  Dropped chunk {i}: failed final safety check, file: {file_source}")
            print(f"   Cleaned content preview: {preview}")
            continue
            
        chunk.page_content = cleaned
        valid.append(chunk)
    if len(valid) != len(chunks):
        print(f"⚠️  Filtered out {len(chunks) - len(valid)} invalid/empty chunks")
    return valid

def _is_safe_for_tokenization(text: str) -> bool:
    """Final safety check to ensure text is safe for tokenization with enhanced PDF support."""
    if not text or not isinstance(text, str):
        return False
    
    # Check for minimum meaningful content (reduced for PDF)
    if len(text.strip()) < 15:
        return False
    
    # Check for excessive control characters
    control_chars = sum(1 for c in text if ord(c) < 32)
    if control_chars > len(text) * 0.03:  # More than 3% control chars
        return False
    
    # Check for sufficient printable content (reduced for PDF and markdown)
    printable_chars = sum(1 for c in text if c.isprintable() and not c.isspace())
    if printable_chars < len(text) * 0.3:  # Reduced to 30% for markdown content
        return False
    
    # Check for sufficient alphabetic content (important for PDF and markdown)
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars < len(text) * 0.1:  # Reduced to 10% for markdown content
        return False
    
    # Check for common problematic patterns
    problematic_patterns = [
        r'^[\x00-\x1f\s]+$',  # Only control chars and whitespace
        r'^[^\w\s]+$',        # Only symbols
        r'^[\d\s\.-]+$',      # Only numbers, spaces, dots, dashes
        r'^[ⅡⅣⅥⅧⅩ]+$',       # Only Roman numerals
        r'^[^\x20-\x7E\u00A0-\uFFFF]+$',  # Only non-printable characters
    ]
    
    for pattern in problematic_patterns:
        if re.match(pattern, text.strip()):
            return False
    
    # Additional PDF-specific checks
    # Check for excessive special characters (but allow markdown syntax)
    special_chars = sum(1 for c in text if c in '!@#$%^&*()_+-=[]{}|;:,.<>?')
    if special_chars > len(text) * 0.4:  # Increased to 40% for markdown content
        return False
    
    # Check for reasonable line lengths (allow longer lines for markdown content)
    lines = text.split('\n')
    for line in lines:
        if len(line.strip()) > 1500:  # Increased to 1500 chars for markdown content
            return False
    
    return True



# Configuration
class Config:
    """Configuration class for RAG settings"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv(override=True)
        
        # API Configuration
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.openai_api_base = os.getenv('OPENAI_API_BASE', 'https://api.deepseek.com/v1')
        
        # Model Configuration
        self.model_name = os.getenv('MODEL_NAME', 'deepseek-chat')
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        
        # Database Configuration
        self.db_name = os.getenv('DB_NAME', 'vector_db')
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
        self.max_batch_size = int(os.getenv('MAX_BATCH_SIZE', '5000'))
        
        # File Processing Configuration
        self.base_path = Path(os.getenv('BASE_PATH', '/home/wukong/Zotero/storage'))
        self.supported_extensions = ['.md', '.pdf', '.html']
        
        # Directory filtering configuration
        self.excluded_directories = self._get_excluded_directories()
        self.excluded_patterns = self._get_excluded_patterns()
        
        # Device Configuration
        self.device = self._get_optimal_device()
        

    
    def _get_excluded_directories(self) -> set:
        """Get directories to exclude from processing"""
        # Get from environment variable (comma-separated)
        excluded_dirs = os.getenv('EXCLUDED_DIRECTORIES', '')
        if excluded_dirs:
            return {dir_name.strip() for dir_name in excluded_dirs.split(',') if dir_name.strip()}
        
        # Default exclusions for common problematic directories
        default_exclusions = {
            'node_modules', '.git', '.svn', '.hg',  # Version control and package directories
            'temp', 'tmp', 'cache', 'logs',         # Temporary and cache directories
            'build', 'dist', 'target',              # Build output directories
            'venv', 'env', '.venv', '.env',         # Virtual environment directories
            '__pycache__', '.pytest_cache',         # Python cache directories
            '.idea', '.vscode',                      # IDE directories
            'ZV57IZ8F', 'YNWF5S9T', 'Q3EW7QZM',    # Zotero specific problematic dirs
            '5M8QRA2Y', '2TBX76BP', 'K8VLV9BV'     # (can be overridden via env var)
        }
        return default_exclusions
    
    def _get_excluded_patterns(self) -> list:
        """Get file patterns to exclude from processing"""
        excluded_patterns = os.getenv('EXCLUDED_PATTERNS', '')
        if excluded_patterns:
            return [pattern.strip() for pattern in excluded_patterns.split(',') if pattern.strip()]
        
        # Default patterns to exclude
        return [
            '*.tmp', '*.temp', '*.bak', '*.backup',  # Temporary files
            '*.log', '*.out', '*.err',               # Log files
            '*.swp', '*.swo', '~*',                  # Editor swap files
            'Thumbs.db', '.DS_Store',                # System files
            'desktop.ini'                            # Windows system files
        ]
    
    def should_exclude_directory(self, dir_name: str) -> bool:
        """Check if a directory should be excluded"""
        # Check exact name match
        if dir_name in self.excluded_directories:
            return True
        
        # Check pattern matches
        for pattern in self.excluded_patterns:
            if self._matches_pattern(dir_name, pattern):
                return True
        
        # Check if it's a hidden directory (starts with .)
        if dir_name.startswith('.'):
            return True
        
        return False
    
    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if a name matches a pattern (simple glob-like matching)"""
        if '*' not in pattern:
            return name == pattern
        
        # Simple glob pattern matching
        if pattern.startswith('*'):
            return name.endswith(pattern[1:])
        elif pattern.endswith('*'):
            return name.startswith(pattern[:-1])
        else:
            # Pattern like 'a*b' - split and check start/end
            parts = pattern.split('*')
            if len(parts) == 2:
                return name.startswith(parts[0]) and name.endswith(parts[1])
        
        return False
    
    def _get_optimal_device(self) -> str:
        """Detect and return optimal device configuration"""
        try:
            if torch.cuda.is_available():
                device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🚀 GPU detected: {gpu_name}")
                print(f"   CUDA version: {torch.version.cuda}")
                print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                torch.cuda.empty_cache()
                return device
            else:
                print("⚠️  No GPU detected, using CPU")
                return 'cpu'
        except Exception as e:
            print(f"⚠️ Could not detect device, using CPU: {e}")
            return 'cpu'
    
    def validate(self, mode: str = 'interactive') -> bool:
        """Validate configuration"""
        # For build mode, we don't need API keys since we're just processing documents
        if mode != 'build' and not self.deepseek_api_key:
            print("❌ DEEPSEEK_API_KEY not set")
            return False
        
        if not self.base_path.exists():
            print(f"❌ Base path does not exist: {self.base_path}")
            return False
        
        return True

class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.text_loader_kwargs = {"encoding": "utf-8"}
    
    def get_files_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all supported files"""
        files_info = {}
        
        def safe_search_files(pattern: str, file_type: str) -> List[str]:
            """Safely search for files, using configuration-based directory filtering"""
            found_files = []
            
            try:
                for root, dirs, files in os.walk(self.config.base_path):
                    # Filter out problematic directories using configuration
                    dirs[:] = [d for d in dirs if not self.config.should_exclude_directory(d)]
                    
                    for file in files:
                        if file.endswith(pattern):
                            try:
                                file_path = os.path.join(root, file)
                                mtime = os.path.getmtime(file_path)
                                size = os.path.getsize(file_path)
                                files_info[file_path] = {
                                    "mtime": mtime,
                                    "last_modified": mtime,  # Add last_modified for consistency
                                    "size": size, 
                                    "path": file_path, 
                                    "type": file_type
                                }
                                found_files.append(file_path)
                            except OSError:
                                continue
            except Exception as e:
                print(f"⚠️  Error searching {file_type} files: {e}")
            
            return found_files
        
        # Search for various file types
        for ext in self.config.supported_extensions:
            file_type = ext[1:]  # Remove the dot
            found_files = safe_search_files(ext, file_type)
            print(f"📄 Found {len(found_files)} {file_type} files")
        
        print(f"📚 Total files found: {len(files_info)}")
        return files_info
    
    def create_chunks_from_files(self, file_paths: List[str], callback=None) -> List[Document]:
        """Create chunks from specified file list with optional callback for real-time processing"""
        documents = []
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            try:
                file_path = Path(file_path)
                file_extension = file_path.suffix.lower()
                
                if file_extension == '.md':
                    docs = self._process_markdown(file_path)
                elif file_extension == '.pdf':
                    docs = self._process_pdf(file_path)
                elif file_extension == '.html':
                    docs = self._process_html(file_path)
                else:
                    print(f"⚠️ Unsupported file type: {file_path}")
                    continue
                
                documents.extend(docs)
                
                # If callback is provided, process chunks immediately
                if callback and docs:
                    chunks = self._split_documents_to_chunks(docs)
                    if chunks:
                        callback(chunks, file_path)
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
        
        # If no callback, return all chunks at once (backward compatibility)
        if not callback:
            chunks = self._split_documents_to_chunks(documents)
            print(f"✅ Generated {len(chunks)} chunks from {len(documents)} documents")
            return chunks
        
        return []
    
    def _split_documents_to_chunks(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            return []
        
        text_splitter = CharacterTextSplitter(
            chunk_size=self.config.chunk_size, 
            chunk_overlap=self.config.chunk_overlap
        )
        return text_splitter.split_documents(documents)
    
    def _process_markdown(self, file_path: Path) -> List[Document]:
        """Process markdown files"""
        docs = TextLoader(str(file_path), **self.text_loader_kwargs).load()
        for doc in docs:
            doc.metadata["doc_type"] = file_path.parent.name
            doc.metadata["file_type"] = "markdown"
            doc.metadata["source"] = str(file_path)
            try:
                doc.metadata["last_modified"] = os.path.getmtime(str(file_path))
                doc.metadata["size"] = os.path.getsize(str(file_path))
            except Exception:
                pass
        return docs
    
    def _process_pdf(self, file_path: Path) -> List[Document]:
        """Process PDF files with fallback options"""
        try:
            docs = PyPDFLoader(str(file_path)).load()
            for doc in docs:
                doc.metadata["doc_type"] = file_path.parent.name
                doc.metadata["file_type"] = "pdf"
                doc.metadata["page_number"] = doc.metadata.get("page", "unknown")
                doc.metadata["source"] = str(file_path)
                try:
                    doc.metadata["last_modified"] = os.path.getmtime(str(file_path))
                    doc.metadata["size"] = os.path.getsize(str(file_path))
                except Exception:
                    pass
            print(f"✅ Successfully processed PDF: {file_path.name}")
            return docs
        except Exception as pdf_error:
            print(f"⚠️ PDF processing failed for {file_path}: {pdf_error}")
            # Try UnstructuredFileLoader as fallback
            try:
                docs = UnstructuredFileLoader(str(file_path)).load()
                for doc in docs:
                    doc.metadata["doc_type"] = file_path.parent.name
                    doc.metadata["file_type"] = "pdf"
                    doc.metadata["source"] = str(file_path)
                    try:
                        doc.metadata["last_modified"] = os.path.getmtime(str(file_path))
                        doc.metadata["size"] = os.path.getsize(str(file_path))
                    except Exception:
                        pass
                print(f"✅ PDF processed with UnstructuredFileLoader: {file_path.name}")
                return docs
            except Exception as unstructured_error:
                print(f"❌ UnstructuredFileLoader also failed for {file_path}: {unstructured_error}")
                return []
    
    def _process_html(self, file_path: Path) -> List[Document]:
        """Process HTML files with fallback options"""
        try:
            docs = UnstructuredHTMLLoader(str(file_path)).load()
            for doc in docs:
                doc.metadata["doc_type"] = file_path.parent.name
                doc.metadata["file_type"] = "html"
                doc.metadata["source"] = str(file_path)
                try:
                    doc.metadata["last_modified"] = os.path.getmtime(str(file_path))
                    doc.metadata["size"] = os.path.getsize(str(file_path))
                except Exception:
                    pass
            print(f"✅ Successfully processed HTML: {file_path.name}")
            return docs
        except Exception as html_error:
            print(f"⚠️ HTML processing failed for {file_path}: {html_error}")
            # Try UnstructuredFileLoader as fallback
            try:
                docs = UnstructuredFileLoader(str(file_path)).load()
                for doc in docs:
                    doc.metadata["doc_type"] = file_path.parent.name
                    doc.metadata["file_type"] = "html"
                    doc.metadata["source"] = str(file_path)
                    try:
                        doc.metadata["last_modified"] = os.path.getmtime(str(file_path))
                        doc.metadata["size"] = os.path.getsize(str(file_path))
                    except Exception:
                        pass
                print(f"✅ HTML processed with UnstructuredFileLoader: {file_path.name}")
                return docs
            except Exception as unstructured_error:
                print(f"❌ UnstructuredFileLoader also failed for {file_path}: {unstructured_error}")
                return []

class VectorDatabaseManager:
    """Manages vector database operations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = self._create_embeddings()
        self.vectorstore = None
    
    def _create_embeddings(self):
        """Create embeddings based on device configuration"""
        if self.config.device == 'cuda':
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': self.config.device},
                encode_kwargs={"batch_size": 256, "normalize_embeddings": True, 'device': self.config.device}
            )
            print(f"🚀 GPU acceleration: using {model_name}")
        else:
            embeddings = OpenAIEmbeddings(
                model_name="text-embedding-ada-002",
                openai_api_base=self.config.openai_api_base,
                openai_api_key=self.config.deepseek_api_key
            )
            print("💻 CPU mode: using DeepSeek embeddings")
        
        return embeddings
    
    def get_or_create_vectorstore(self) -> Tuple[Chroma, bool]:
        """Get existing vectorstore or create new one"""
        if os.path.exists(self.config.db_name):
            print(f"Loading existing vector database from {self.config.db_name}")
            self.vectorstore = Chroma(
                persist_directory=self.config.db_name, 
                embedding_function=self.embeddings
            )
            return self.vectorstore, True
        else:
            print(f"Creating new vector database at {self.config.db_name}")
            return self._create_new_vectorstore()
    
    def _create_new_vectorstore(self) -> Tuple[Chroma, bool]:
        """Create new vector database"""
        # Create empty vector database
        self.vectorstore = Chroma(
            persist_directory=self.config.db_name,
            embedding_function=self.embeddings
        )
        
        # Process documents and add to database
        processor = DocumentProcessor(self.config)
        files_info = processor.get_files_info()
        
        if not files_info:
            print("⚠️ No files found to process")
            return self.vectorstore, False
        
        # Get all file paths
        all_files = list(files_info.keys())
        chunks = processor.create_chunks_from_files(all_files)
        
        if not chunks:
            print("⚠️ No chunks generated")
            return self.vectorstore, False
        
        # Add chunks to database in batches
        self._add_chunks_in_batches(chunks)
        
        return self.vectorstore, False
    
    def _add_chunks_in_batches(self, chunks: List[Document]):
        """Add chunks to database in batches"""
        batch_size = min(
            10000 if self.config.device == 'cuda' else 5000, 
            self.config.max_batch_size
        )
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"📦 Adding {len(chunks)} chunks in {total_batches} batches of {batch_size}")
        
        successful_batches = 0
        start_time = time.time()
        
        with tqdm(total=total_batches, desc="🏗️  Building database", unit="batch") as pbar:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                try:
                    batch_start_time = time.time()
                    self.vectorstore.add_documents(batch)
                    batch_time = time.time() - batch_start_time
                    
                    successful_batches += 1
                    pbar.set_postfix({
                        'Success': successful_batches,
                        'Batch Time': f"{batch_time:.1f}s"
                    })
                    
                except Exception as e:
                    print(f"❌ Error adding batch {batch_num}: {e}")
                    print(traceback.format_exc())
                    # Try smaller batch size
                    if len(batch) > 1000:
                        self._add_chunks_in_smaller_batches(batch)
                
                pbar.update(1)
        
        total_time = time.time() - start_time
        print(f"🏗️  Database creation completed in {total_time:.1f}s")
        print(f"   ✅ Successful batches: {successful_batches}")
        print(f"   🚀 Speed: {len(chunks)/total_time:.1f} chunks/second")
    
    def _add_chunks_in_smaller_batches(self, chunks: List[Document]):
        """Add chunks in smaller batches as fallback"""
        smaller_batch_size = 1000
        smaller_success = 0
        
        for j in range(0, len(chunks), smaller_batch_size):
            smaller_batch = chunks[j:j + smaller_batch_size]
            try:
                self.vectorstore.add_documents(smaller_batch)
                smaller_success += 1
            except Exception as e:
                print(f"❌ Error with smaller batch: {e}")
                break
        
        if smaller_success > 0:
            print(f"📊 {smaller_success} smaller batches succeeded")

class RAGSystem:
    """Main RAG system class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = VectorDatabaseManager(config)
        self.vectorstore = None
        self.conversation_chain = None
        self.memory_manager = None
    
    def initialize(self) -> bool:
        """Initialize the RAG system"""
        try:
            # Get or create vectorstore
            self.vectorstore, is_existing = self.db_manager.get_or_create_vectorstore()
            
            if is_existing:
                print(f"Vector database loaded with {self.vectorstore._collection.count()} documents")
            else:
                print(f"Vector database created with {self.vectorstore._collection.count()} documents")
            
            # Initialize conversation components
            self._initialize_conversation_components()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            return False
    
    def _initialize_conversation_components(self):
        """Initialize conversation chain and memory manager"""
        try:
            # Create LLM
            llm = ChatOpenAI(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                openai_api_base=self.config.openai_api_base,
                openai_api_key=self.config.deepseek_api_key
            )
            
            # Create retriever
            retriever = self.vectorstore.as_retriever()
            
            # Create conversation chain
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                verbose=True
            )
            
            # Create memory manager
            self.memory_manager = MemoryManager()
            
            print("✅ Conversation components initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize conversation components: {e}")
    
    def query(self, question: str, session_id: str = "default") -> Optional[Dict[str, Any]]:
        """Query the RAG system"""
        if not self.conversation_chain:
            print("❌ RAG system not initialized")
            return None
        
        try:
            chat_history = self.memory_manager.get_chat_history(session_id)
            
            inputs = {
                "question": question,
                "chat_history": chat_history.messages
            }
            
            result = self.conversation_chain.invoke(inputs)
            
            # Update chat history
            chat_history.add_user_message(question)
            chat_history.add_ai_message(result["answer"])
            
            return result
            
        except Exception as e:
            print(f"❌ Query error: {e}")
            return None
    
    def update_database(self) -> bool:
        """Update existing database with new/modified documents"""
        try:
            print("🔄 Starting incremental database update...")
            
            # Get current database info
            current_docs = self.vectorstore._collection.count()
            print(f"📊 Current database has {current_docs} documents")
            
            # Get file information for incremental update
            processor = DocumentProcessor(self.config)
            files_info = processor.get_files_info()
            
            if not files_info:
                print("⚠️ No files found to process")
                return True
            
            # Get existing document metadata to track changes
            existing_metadata = self._get_existing_document_metadata()
            
            # Identify new and modified files
            new_files, modified_files = self._identify_changed_files(files_info, existing_metadata)
            
            if not new_files and not modified_files:
                print("✅ Database is up to date, no changes detected")
                return True
            
            print(f"📝 Found {len(new_files)} new files and {len(modified_files)} modified files")
            
            # Process new and modified files with real-time chunk processing
            all_changed_files = new_files + modified_files
            
            # Remove old versions of modified documents first
            if modified_files:
                self._remove_modified_documents(modified_files)
            
            # Process files and add chunks in real-time
            total_chunks_processed = 0
            total_chunks_added = 0
            
            def process_chunks_realtime(chunks: List[Document], file_path: Path):
                nonlocal total_chunks_processed, total_chunks_added
                
                # Validate and clean chunks
                valid_chunks = filter_and_clean_chunks(chunks)
                if not valid_chunks:
                    print(f"⚠️ No valid chunks from {file_path.name}")
                    return
                
                total_chunks_processed += len(chunks)
                print(f"📝 {file_path.name}: {len(chunks)} chunks -> {len(valid_chunks)} valid chunks")
                
                # Add chunks to database immediately
                if valid_chunks:
                    if self._add_chunks_in_batches(valid_chunks):
                        total_chunks_added += len(valid_chunks)
                        print(f"✅ Added {len(valid_chunks)} chunks from {file_path.name}")
                    else:
                        print(f"❌ Failed to add chunks from {file_path.name}")
            
            # Process files with real-time callback
            processor.create_chunks_from_files(all_changed_files, callback=process_chunks_realtime)
            
            if total_chunks_processed == 0:
                print("⚠️ No chunks generated from changed files")
                return True
            
            print(f"📊 Total chunks processed: {total_chunks_processed}")
            print(f"📊 Total chunks added: {total_chunks_added}")
            
            if total_chunks_added == 0:
                print("❌ No chunks were successfully added to database")
                return False
            
            # Update conversation components with new database
            self._initialize_conversation_components()
            
            final_docs = self.vectorstore._collection.count()
            print(f"✅ Database update completed. Total documents: {final_docs} (+{final_docs - current_docs})")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to update database: {e}")
            return False
    
    def _clean_chunk_content(self, content: str) -> str:
        """Deprecated: kept for backward compatibility. Uses sanitize_text."""
        cleaned = sanitize_text(content)
        return cleaned or ""
    
    def _get_existing_document_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata of existing documents in the database.

        Note: Some Chroma versions require include=['metadatas'] to return metadatas.
        """
        try:
            # Explicitly request metadatas to ensure they are included
            # Some Chroma versions don't support requesting "ids" via include
            results = self.vectorstore._collection.get(include=["metadatas"])
            metadata: Dict[str, Dict[str, Any]] = {}

            if results and results.get('metadatas'):
                for i, doc_metadata in enumerate(results['metadatas']):
                    if not doc_metadata:
                        continue
                    source = doc_metadata.get('source')
                    if not source:
                        continue
                    # Always keep the most recent doc metadata seen for the same source
                    metadata[source] = {
                        'metadata': doc_metadata
                    }

            return metadata
        except Exception as e:
            print(f"⚠️ Could not retrieve existing metadata: {e}")
            return {}
    
    def _identify_changed_files(self, files_info: Dict[str, Dict[str, Any]], 
                               existing_metadata: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Identify new and modified files.

        - A file is considered new if there's no record for its source in the DB.
        - A file is considered modified only if we can confidently detect change
          via 'last_modified' or 'size'. If metadata is missing, we assume unchanged
          to avoid unnecessary reprocessing.
        """
        new_files: List[str] = []
        modified_files: List[str] = []

        for file_path, file_info in files_info.items():
            existing_info = existing_metadata.get(file_path)

            if not existing_info:
                # Double-check existence by querying collection directly to avoid false negatives
                try:
                    res = self.vectorstore._collection.get(
                        where={"source": file_path}, include=["metadatas"], limit=1
                    )
                    has_entry = bool(res and res.get('metadatas'))
                except Exception:
                    has_entry = False

                if has_entry:
                    # Treat as existing but unchanged when metadata map missed it
                    continue
                else:
                    new_files.append(file_path)
                    continue

            # Check for modification using robust signals
            stored_meta = existing_info.get('metadata', {})
            stored_mtime = stored_meta.get('last_modified')
            stored_size = stored_meta.get('size')
            current_mtime = file_info.get('last_modified') or file_info.get('mtime')
            current_size = file_info.get('size')

            # Only mark modified when we have comparable fields and they differ
            if stored_mtime is not None and current_mtime is not None and current_mtime > stored_mtime:
                modified_files.append(file_path)
                continue
            if stored_size is not None and current_size is not None and current_size != stored_size:
                modified_files.append(file_path)

        return new_files, modified_files
    
    def _remove_modified_documents(self, modified_files: List[str]):
        """Remove old versions of modified documents"""
        try:
            for file_path in modified_files:
                # Find and remove documents with this source
                results = self.vectorstore._collection.get(
                    where={"source": file_path}
                )
                if results and 'ids' in results and results['ids']:
                    self.vectorstore._collection.delete(ids=results['ids'])
                    print(f"🗑️  Removed old version of: {file_path}")
        except Exception as e:
            print(f"⚠️ Could not remove modified documents: {e}")
    
    def _add_chunks_in_batches(self, chunks: List[Document]):
        """Add chunks to database in batches"""
        batch_size = min(
            10000 if self.config.device == 'cuda' else 5000, 
            self.config.max_batch_size
        )
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"📦 Adding {len(chunks)} chunks in {total_batches} batches of {batch_size}")
        
        successful_batches = 0
        failed_batches = 0
        total_added_chunks = 0
        start_time = time.time()
        
        with tqdm(total=total_batches, desc="🔄 Updating database", unit="batch") as pbar:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                try:
                    batch_start_time = time.time()
                    self.vectorstore.add_documents(batch)
                    batch_time = time.time() - batch_start_time
                    
                    successful_batches += 1
                    total_added_chunks += len(batch)
                    pbar.set_postfix({
                        'Success': successful_batches,
                        'Failed': failed_batches,
                        'Batch Time': f"{batch_time:.1f}s"
                    })
                    
                    pbar.update(1)
                    
                except Exception as e:
                    failed_batches += 1
                    print(f"❌ Failed to add batch {batch_num}: {e}")
                    print(traceback.format_exc())
                    print(f"   Batch size: {len(batch)}")
                    print(f"   First chunk content preview: {str(batch[0].page_content)[:100]}...")
                    pbar.update(1)
                    continue
        
        total_time = time.time() - start_time
        if failed_batches > 0:
            print(f"⚠️  Batch processing completed with {failed_batches} failed batches")
            print(f"   Successful batches: {successful_batches}")
            print(f"   Failed batches: {failed_batches}")
            print(f"   Total chunks processed: {total_added_chunks}/{len(chunks)}")
        else:
            print(f"✅ Successfully added all {total_added_chunks} chunks in {total_time:.1f}s")
        
        return total_added_chunks > 0

class MemoryManager:
    """Manages chat history for multiple sessions"""
    
    def __init__(self):
        self.chat_histories = {}
    
    def get_chat_history(self, session_id: str) -> ChatMessageHistory:
        """Get or create chat history for a session"""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = ChatMessageHistory()
        return self.chat_histories[session_id]



def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RAG Demo - Production Ready Script')
    
    parser.add_argument('--mode', choices=['build', 'query', 'interactive'], 
                       default='interactive', help='Operation mode')
    parser.add_argument('--db-name', help='Database name')
    parser.add_argument('--base-path', help='Base path for documents')
    parser.add_argument('--chunk-size', type=int, help='Chunk size for text splitting')
    parser.add_argument('--chunk-overlap', type=int, help='Chunk overlap for text splitting')
    parser.add_argument('--model-name', help='LLM model name')
    parser.add_argument('--temperature', type=float, help='LLM temperature')
    parser.add_argument('--excluded-dirs', help='Comma-separated list of directories to exclude')
    parser.add_argument('--excluded-patterns', help='Comma-separated list of file patterns to exclude')

    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = Config()
    
    # Override config with command line arguments
    if args.db_name:
        config.db_name = args.db_name
    if args.base_path:
        config.base_path = Path(args.base_path)
    if args.chunk_size:
        config.chunk_size = args.chunk_size
    if args.chunk_overlap:
        config.chunk_overlap = args.chunk_overlap
    if args.model_name:
        config.model_name = args.model_name
    if args.temperature:
        config.temperature = args.temperature
    if args.excluded_dirs:
        config.excluded_directories = {dir_name.strip() for dir_name in args.excluded_dirs.split(',') if dir_name.strip()}
    if args.excluded_patterns:
        config.excluded_patterns = [pattern.strip() for pattern in args.excluded_patterns.split(',') if pattern.strip()]
    

    
    # Validate configuration
    if not config.validate(args.mode):
        print("❌ Configuration validation failed")
        return 1
    
    try:
        # Initialize RAG system
        rag_system = RAGSystem(config)
        if not rag_system.initialize():
            print("❌ Failed to initialize RAG system")
            return 1
        
        # Run based on mode
        if args.mode == 'build':
            print("🏗️  Starting database build...")
            
            # Check if database exists
            if os.path.exists(config.db_name):
                print(f"📚 Existing database found: {config.db_name}")
                # Perform incremental update
                if not rag_system.update_database():
                    print("❌ Failed to update database")
                    return 1
            else:
                print(f"🆕 Creating new database: {config.db_name}")
                # Create new database
                if not rag_system.initialize():
                    print("❌ Failed to create database")
                    return 1
            
            print("✅ Database build completed")
            return 0
        elif args.mode == 'query':
            # Simple query mode
            question = input("Enter your question: ")
            result = rag_system.query(question)
            if result:
                print(f"🤖 Answer: {result['answer']}")
            return 0
        elif args.mode == 'interactive':
            # Interactive mode with Gradio
            return run_interactive_mode(rag_system)
        else:
            print(f"❌ Unknown mode: {args.mode}")
            return 1
            
    except KeyboardInterrupt:
        print("👋 Interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

def run_interactive_mode(rag_system: RAGSystem):
    """Run interactive mode with Gradio"""
    try:
        def chat(message, history):
            """Chat function for Gradio"""
            chat_history = []
            for human, ai in history:
                if human:
                    chat_history.append(HumanMessage(content=human))
                if ai:
                    chat_history.append(AIMessage(content=ai))
            
            result = rag_system.conversation_chain.invoke({
                "question": message,
                "chat_history": chat_history
            })
            
            return result["answer"]
        
        # Launch Gradio interface
        view = gr.ChatInterface(chat).launch(inbrowser=True)
        return 0
        
    except Exception as e:
        print(f"❌ Failed to launch interactive mode: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

