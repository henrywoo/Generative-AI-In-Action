#!/usr/bin/env python3
"""
Clean low-quality HTML documents from ChromaDB
Removes SingleFile pages and other low-quality HTML content
"""

import os
import sys
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from rag_demo import RAGSystem, Config

def clean_html_documents(dry_run: bool = True, force: bool = False):
    """Clean low-quality HTML documents from the database"""
    
    print("🧹 HTML Document Cleanup Tool")
    print("=" * 50)
    
    if dry_run:
        print("🔍 DRY RUN MODE - No documents will be actually deleted")
        print("   Use --force to actually delete documents")
    else:
        if not force:
            print("⚠️  WARNING: This will permanently delete documents!")
            confirm = input("Type 'YES' to confirm: ")
            if confirm != 'YES':
                print("❌ Operation cancelled")
                return
        print("🗑️  ACTUAL DELETION MODE - Documents will be permanently removed")
    
    print()
    
    # Initialize configuration
    config = Config()
    if not config.validate('interactive'):
        print("❌ Configuration validation failed")
        return
    
    # Initialize RAG system
    print("🚀 Initializing RAG system...")
    rag_system = RAGSystem(config)
    if not rag_system.initialize():
        print("❌ Failed to initialize RAG system")
        return
    
    print("✅ RAG system initialized successfully")
    
    # Clean low-quality HTML documents
    print("\n🔍 Starting HTML document cleanup...")
    result = rag_system.delete_low_quality_html_documents(dry_run=dry_run)
    
    if not result:
        print("❌ Cleanup failed")
        return
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Cleanup Summary:")
    print(f"   Total HTML documents: {result.get('total_html', 0)}")
    print(f"   Low-quality documents: {result.get('low_quality_count', 0)}")
    
    if result.get('low_quality_sources'):
        print(f"   Low-quality percentage: {result['low_quality_count']/result['total_html']*100:.1f}%")
        
        if dry_run:
            print(f"\n💡 To actually delete these documents, run:")
            print(f"   python clean_html_docs.py --force")
        else:
            print(f"\n✅ Cleanup completed successfully!")
    else:
        print("   No low-quality documents found")
    
    # Show remaining database stats
    try:
        remaining_docs = rag_system.vectorstore._collection.count()
        print(f"\n📊 Remaining documents in database: {remaining_docs}")
    except Exception as e:
        print(f"⚠️  Could not get remaining document count: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Clean low-quality HTML documents from ChromaDB')
    parser.add_argument('--force', action='store_true', 
                       help='Actually delete documents (default is dry run)')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Show what would be deleted without actually deleting (default)')
    
    args = parser.parse_args()
    
    # If --force is specified, override --dry-run
    if args.force:
        dry_run = False
    else:
        dry_run = args.dry_run
    
    try:
        clean_html_documents(dry_run=dry_run, force=args.force)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
