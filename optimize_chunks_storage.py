#!/usr/bin/env python3
"""
Optimize chunk storage by removing redundant raw_chunk_text fields
when similarity with chunk_text is above threshold.
"""

import json
import os
from typing import Dict, Any, List
from difflib import SequenceMatcher
import argparse
from tqdm import tqdm


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings (0-1 scale)"""
    if not text1 or not text2:
        return 0.0
    
    return SequenceMatcher(None, text1.strip(), text2.strip()).ratio()


def has_important_formatting(text: str) -> bool:
    """Check if text contains important formatting that shouldn't be lost"""
    formatting_indicators = [
        '|',  # Table pipes
        '\n---',  # Table separators
        '\n\n',  # Multiple newlines
        '\t',  # Tabs
        '  ',  # Multiple spaces (might indicate formatting)
    ]
    
    return any(indicator in text for indicator in formatting_indicators)


def should_keep_raw_text(raw_text: str, chunk_text: str, threshold: float = 0.95) -> bool:
    """
    Determine if raw_chunk_text should be kept based on similarity and other factors
    
    Args:
        raw_text: Original raw text
        chunk_text: Processed chunk text  
        threshold: Similarity threshold above which raw text is considered redundant
    
    Returns:
        True if raw_text should be kept, False if it can be removed
    """
    if not raw_text or not chunk_text:
        return True  # Keep if either is missing
    
    # Calculate similarity
    similarity = calculate_similarity(raw_text, chunk_text)
    
    # Keep if similarity is below threshold
    if similarity < threshold:
        return True
    
    # Keep if raw text has important formatting
    if has_important_formatting(raw_text):
        return True
    
    # Keep if there's a significant length difference (might indicate truncation)
    length_diff_ratio = abs(len(raw_text) - len(chunk_text)) / max(len(raw_text), len(chunk_text))
    if length_diff_ratio > 0.1:  # 10% length difference
        return True
    
    return False  # Safe to remove


def optimize_chunk(chunk: Dict[str, Any], threshold: float, stats: Dict[str, int]) -> Dict[str, Any]:
    """
    Optimize a single chunk by potentially removing raw_chunk_text
    
    Args:
        chunk: Chunk dictionary
        threshold: Similarity threshold
        stats: Statistics tracking dictionary
    
    Returns:
        Optimized chunk dictionary
    """
    stats['total_chunks'] += 1
    
    raw_text = chunk.get('raw_chunk_text', '')
    chunk_text = chunk.get('chunk_text', '')
    
    if not raw_text:
        stats['no_raw_text'] += 1
        return chunk
    
    if should_keep_raw_text(raw_text, chunk_text, threshold):
        stats['kept_raw_text'] += 1
        return chunk
    else:
        # Remove raw_chunk_text
        optimized_chunk = chunk.copy()
        optimized_chunk.pop('raw_chunk_text', None)
        stats['removed_raw_text'] += 1
        stats['storage_saved'] += len(raw_text)
        return optimized_chunk


def optimize_chunks_file(input_file: str, output_file: str, threshold: float = 0.95, 
                        dry_run: bool = False) -> Dict[str, int]:
    """
    Optimize chunks file by removing redundant raw_chunk_text fields
    
    Args:
        input_file: Path to input chunks JSON file
        output_file: Path to output optimized JSON file
        threshold: Similarity threshold (0-1)
        dry_run: If True, only calculate stats without writing output
    
    Returns:
        Dictionary with optimization statistics
    """
    print(f"Loading chunks from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    stats = {
        'total_chunks': 0,
        'kept_raw_text': 0,
        'removed_raw_text': 0,
        'no_raw_text': 0,
        'storage_saved': 0
    }
    
    # Process chunks
    optimized_chunks = []
    for chunk in tqdm(chunks, desc="Optimizing chunks"):
        optimized_chunk = optimize_chunk(chunk, threshold, stats)
        optimized_chunks.append(optimized_chunk)
    
    # Calculate file size statistics
    original_size = os.path.getsize(input_file)
    
    if not dry_run:
        print(f"Writing optimized chunks to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(optimized_chunks, f, indent=2, ensure_ascii=False)
        
        optimized_size = os.path.getsize(output_file)
        stats['file_size_original'] = original_size
        stats['file_size_optimized'] = optimized_size
        stats['file_size_saved'] = original_size - optimized_size
        stats['file_size_reduction_percent'] = (stats['file_size_saved'] / original_size) * 100
    
    return stats, optimized_chunks if dry_run else None


def print_statistics(stats: Dict[str, int], threshold: float):
    """Print optimization statistics"""
    print("\n" + "="*50)
    print("CHUNK OPTIMIZATION STATISTICS")
    print("="*50)
    print(f"Similarity threshold: {threshold:.1%}")
    print(f"Total chunks processed: {stats['total_chunks']:,}")
    print(f"Chunks with raw_text kept: {stats['kept_raw_text']:,}")
    print(f"Chunks with raw_text removed: {stats['removed_raw_text']:,}")
    print(f"Chunks without raw_text: {stats['no_raw_text']:,}")
    
    if stats['total_chunks'] > 0:
        removal_rate = (stats['removed_raw_text'] / stats['total_chunks']) * 100
        print(f"Raw text removal rate: {removal_rate:.1f}%")
    
    if 'storage_saved' in stats:
        print(f"Text storage saved: {stats['storage_saved']:,} characters")
    
    if 'file_size_original' in stats:
        print(f"Original file size: {stats['file_size_original']:,} bytes")
        print(f"Optimized file size: {stats['file_size_optimized']:,} bytes")
        print(f"File size reduction: {stats['file_size_saved']:,} bytes ({stats['file_size_reduction_percent']:.1f}%)")


def analyze_sample_removals(chunks: List[Dict], threshold: float = 0.95, num_samples: int = 5):
    """Analyze sample chunks that would be removed to validate the approach"""
    print(f"\n{'='*50}")
    print("SAMPLE ANALYSIS: Chunks where raw_text would be removed")
    print("="*50)
    
    samples_shown = 0
    for i, chunk in enumerate(chunks):
        raw_text = chunk.get('raw_chunk_text', '')
        chunk_text = chunk.get('chunk_text', '')
        
        if raw_text and not should_keep_raw_text(raw_text, chunk_text, threshold):
            similarity = calculate_similarity(raw_text, chunk_text)
            print(f"\nSample {samples_shown + 1}:")
            print(f"Document: {chunk.get('document_name', 'Unknown')}")
            print(f"Chunk ID: {chunk.get('chunk_id', 'Unknown')}")
            print(f"Similarity: {similarity:.3f}")
            print(f"Raw text preview: {raw_text[:100]}...")
            print(f"Chunk text preview: {chunk_text[:100]}...")
            
            samples_shown += 1
            if samples_shown >= num_samples:
                break
    
    if samples_shown == 0:
        print("No chunks would have raw_text removed with current threshold.")


def main():
    parser = argparse.ArgumentParser(description="Optimize chunk storage by removing redundant raw_chunk_text")
    parser.add_argument('input_file', help='Input chunks JSON file')
    parser.add_argument('-o', '--output', help='Output optimized JSON file')
    parser.add_argument('-t', '--threshold', type=float, default=0.95, 
                       help='Similarity threshold (0-1) for removing raw_text (default: 0.95)')
    parser.add_argument('-d', '--dry-run', action='store_true', 
                       help='Analyze only, do not create output file')
    parser.add_argument('-s', '--samples', type=int, default=5,
                       help='Number of sample removals to show (default: 5)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return 1
    
    if not args.dry_run and not args.output:
        args.output = args.input_file.replace('.json', '_optimized.json')
    
    try:
        # Run optimization
        stats, sample_chunks = optimize_chunks_file(
            args.input_file, 
            args.output or '', 
            args.threshold, 
            args.dry_run
        )
        
        # Print statistics
        print_statistics(stats, args.threshold)
        
        # Show sample analysis if dry run
        if args.dry_run and sample_chunks:
            analyze_sample_removals(sample_chunks, args.threshold, args.samples)
        
        if args.dry_run:
            print(f"\nDry run completed. No files were modified.")
        else:
            print(f"\nOptimization completed. Output saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
