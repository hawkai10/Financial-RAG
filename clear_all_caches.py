#!/usr/bin/env python3
"""
Complete cache clearing utility for the RAG system.
Clears all types of caches: memory, disk, and database.
"""

import os
import sqlite3
from rag_backend import clear_all_caches, db_pool
from feedback_database import EnhancedFeedbackDatabase

def clear_complete_cache_system():
    """Clear ALL caches in the system completely."""
    
    print("🧹 Starting complete cache clearing...")
    
    # 1. Clear optimization caches (memory + disk)
    print("   📦 Clearing chunk and embedding caches...")
    clear_all_caches()
    
    # 2. Clear query result cache in database
    print("   🗄️  Clearing database query cache...")
    try:
        feedback_db = EnhancedFeedbackDatabase()
        
        # Clear query cache table
        with db_pool.get_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM query_cache")
            deleted_rows = cursor.rowcount
            conn.commit()
            print(f"      ✅ Deleted {deleted_rows} cached query results")
            
        # Clear performance metrics cache
        with db_pool.get_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM performance_metrics WHERE metric_type LIKE '%cache%'")
            deleted_metrics = cursor.rowcount
            conn.commit()
            print(f"      ✅ Deleted {deleted_metrics} cache performance metrics")
            
    except Exception as e:
        print(f"      ❌ Database cache clearing failed: {e}")
    
    # 3. Clear disk embedding cache files
    print("   💾 Clearing disk embedding cache files...")
    try:
        cache_dir = "embedding_cache"
        if os.path.exists(cache_dir):
            files_deleted = 0
            for file in os.listdir(cache_dir):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(cache_dir, file))
                    files_deleted += 1
            print(f"      ✅ Deleted {files_deleted} embedding cache files")
        else:
            print("      ℹ️  No embedding cache directory found")
    except Exception as e:
        print(f"      ❌ Disk cache clearing failed: {e}")
    
    # 4. Reset cache statistics
    print("   📊 Resetting cache statistics...")
    try:
        from rag_backend import chunk_cache, embedding_cache
        chunk_cache.hits = 0
        chunk_cache.misses = 0
        embedding_cache.hits = 0
        embedding_cache.misses = 0
        print("      ✅ Cache statistics reset")
    except Exception as e:
        print(f"      ❌ Statistics reset failed: {e}")
    
    print("🎉 Complete cache clearing finished!")
    print("\n📈 Recommendation: Run a test query to rebuild essential caches.")

def clear_only_memory_caches():
    """Clear only in-memory caches, keep disk cache."""
    print("🧠 Clearing only memory caches...")
    
    from rag_backend import chunk_cache, embedding_cache
    
    # Clear memory caches but preserve disk
    chunk_cache.cache.clear()
    chunk_cache.access_times.clear()
    chunk_cache.file_timestamps.clear()
    
    embedding_cache.memory_cache.clear()
    
    print("   ✅ Memory caches cleared")
    print("   ℹ️  Disk caches preserved")

def get_cache_status():
    """Show current cache status before/after clearing."""
    try:
        from rag_backend import get_optimization_stats, get_cache_health
        
        stats = get_optimization_stats()
        health = get_cache_health()
        
        print("\n📊 Current Cache Status:")
        print(f"   Chunk Cache: {stats['chunk_cache']['cache_size']} items, {stats['chunk_cache']['hit_rate']} hit rate")
        print(f"   Embedding Cache: {stats['embedding_cache']['memory_cache_size']} memory + {stats['embedding_cache']['disk_cache_files']} disk")
        print(f"   Connection Pool: {stats['connection_pool']['utilization']} utilization")
        print(f"   Overall Health: {health['overall_status']}")
        
    except Exception as e:
        print(f"❌ Could not get cache status: {e}")

if __name__ == "__main__":
    print("🔧 RAG System Cache Management Utility")
    print("="*50)
    
    # Show current status
    print("\n📋 BEFORE CLEARING:")
    get_cache_status()
    
    # Ask user what to clear
    choice = input("\nChoose clearing option:\n1. Clear ALL caches (complete reset)\n2. Clear only memory caches\n3. Show status only\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        clear_complete_cache_system()
    elif choice == "2":
        clear_only_memory_caches()
    elif choice == "3":
        print("ℹ️  Status only - no changes made")
    else:
        print("❌ Invalid choice")
        exit(1)
    
    # Show status after clearing
    print("\n📋 AFTER CLEARING:")
    get_cache_status()
