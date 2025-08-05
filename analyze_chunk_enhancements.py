#!/usr/bin/env python3
"""
Script to analyze the differences between raw_chunk_text and chunk_text
after implementing Option 2 text enhancements.
"""

import json
import os
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import argparse


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings (0-1 scale)"""
    if not text1 or not text2:
        return 0.0
    
    return SequenceMatcher(None, text1.strip(), text2.strip()).ratio()


def calculate_readability_improvement(raw_text: str, enhanced_text: str) -> float:
    """Calculate a simple readability improvement score"""
    
    # Factors that indicate improvement
    raw_score = 0
    enhanced_score = 0
    
    # Check for proper formatting indicators
    formatting_indicators = [
        ('|', 2),      # Table formatting
        ('\n\n', 1),   # Paragraph breaks
        ('Rs.', 1),    # Proper currency formatting
        ('**', 1),     # Bold formatting
        ('•', 1),      # Bullet points
        ('---', 1),    # Section dividers
    ]
    
    for indicator, weight in formatting_indicators:
        raw_score += raw_text.count(indicator) * weight
        enhanced_score += enhanced_text.count(indicator) * weight
    
    # Normalize by text length
    raw_score = raw_score / max(len(raw_text), 1)
    enhanced_score = enhanced_score / max(len(enhanced_text), 1)
    
    # Return improvement ratio
    if raw_score == 0:
        return enhanced_score if enhanced_score > 0 else 1.0
    else:
        return enhanced_score / raw_score


def analyze_chunk_differences(chunks_file: str = "contextualized_chunks.json") -> Dict[str, any]:
    """Analyze differences between raw_chunk_text and chunk_text"""
    
    if not os.path.exists(chunks_file):
        print(f"Error: File '{chunks_file}' not found")
        return {}
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    stats = {
        'total_chunks': len(chunks),
        'identical_chunks': 0,
        'enhanced_chunks': 0,
        'table_chunks': 0,
        'text_chunks': 0,
        'average_similarity': 0.0,
        'average_improvement_ratio': 0.0,
        'enhancement_examples': []
    }
    
    similarities = []
    improvement_ratios = []
    
    for chunk in chunks:
        raw_text = chunk.get('raw_chunk_text', '')
        enhanced_text = chunk.get('chunk_text', '')
        is_table = chunk.get('is_table', False)
        
        if is_table:
            stats['table_chunks'] += 1
        else:
            stats['text_chunks'] += 1
        
        # Calculate similarity
        similarity = calculate_similarity(raw_text, enhanced_text)
        similarities.append(similarity)
        
        if similarity >= 0.99:  # Essentially identical
            stats['identical_chunks'] += 1
        else:
            stats['enhanced_chunks'] += 1
            
            # Calculate improvement ratio (readability score)
            improvement_ratio = calculate_readability_improvement(raw_text, enhanced_text)
            improvement_ratios.append(improvement_ratio)
            
            # Store example if significant improvement
            if improvement_ratio > 1.2:  # 20% improvement
                stats['enhancement_examples'].append({
                    'chunk_id': chunk.get('chunk_id', 'Unknown'),
                    'document_name': chunk.get('document_name', 'Unknown'),
                    'is_table': is_table,
                    'similarity': similarity,
                    'improvement_ratio': improvement_ratio,
                    'raw_text_preview': raw_text[:100],
                    'enhanced_text_preview': enhanced_text[:100]
                })
    
    # Calculate averages
    if similarities:
        stats['average_similarity'] = sum(similarities) / len(similarities)
    
    if improvement_ratios:
        stats['average_improvement_ratio'] = sum(improvement_ratios) / len(improvement_ratios)
    
    # Sort examples by improvement ratio
    stats['enhancement_examples'].sort(key=lambda x: x['improvement_ratio'], reverse=True)
    
    return stats


def show_sample_improvements(chunks_file: str = "contextualized_chunks.json", num_samples: int = 3):
    """Show sample improvements for verification"""
    
    if not os.path.exists(chunks_file):
        print(f"Error: File '{chunks_file}' not found")
        return
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    improved_chunks = [
        chunk for chunk in chunks 
        if calculate_similarity(chunk.get('raw_chunk_text', ''), chunk.get('chunk_text', '')) < 0.99
    ]
    
    print(f"\n{'='*70}")
    print(f"SAMPLE TEXT ENHANCEMENTS ({min(num_samples, len(improved_chunks))} examples)")
    print(f"{'='*70}\n")
    
    for i, chunk in enumerate(improved_chunks[:num_samples]):
        raw_text = chunk.get('raw_chunk_text', '')
        enhanced_text = chunk.get('chunk_text', '')
        similarity = calculate_similarity(raw_text, enhanced_text)
        improvement = calculate_readability_improvement(raw_text, enhanced_text)
        
        print(f"--- Sample {i+1} ---")
        print(f"Document: {chunk.get('document_name', 'Unknown')}")
        print(f"Chunk ID: {chunk.get('chunk_id', 'Unknown')}")
        print(f"Is Table: {chunk.get('is_table', False)}")
        print(f"Similarity: {similarity:.3f}")
        print(f"Improvement Ratio: {improvement:.2f}x")
        
        print(f"\nRAW TEXT:")
        print(f"{raw_text[:200]}{'...' if len(raw_text) > 200 else ''}")
        
        print(f"\nENHANCED TEXT:")
        print(f"{enhanced_text[:200]}{'...' if len(enhanced_text) > 200 else ''}")
        
        print("\n" + "="*70 + "\n")
    
    if len(improved_chunks) == 0:
        print("No enhanced chunks found. All chunks are identical.")


def compare_before_after(old_file: str, new_file: str):
    """Compare enhancement stats between old and new chunk files"""
    
    print(f"\n{'='*70}")
    print("BEFORE vs AFTER ENHANCEMENT COMPARISON")
    print(f"{'='*70}")
    
    if os.path.exists(old_file):
        old_stats = analyze_chunk_differences(old_file)
        print(f"\nBEFORE (Old file: {old_file}):")
        print_stats_summary(old_stats)
    else:
        print(f"\nOld file '{old_file}' not found - skipping comparison")
    
    if os.path.exists(new_file):
        new_stats = analyze_chunk_differences(new_file)
        print(f"\nAFTER (New file: {new_file}):")
        print_stats_summary(new_stats)
    else:
        print(f"\nNew file '{new_file}' not found")


def print_stats_summary(stats: Dict[str, any]):
    """Print a summary of enhancement statistics"""
    if not stats:
        print("No data available")
        return
    
    total = stats.get('total_chunks', 0)
    enhanced = stats.get('enhanced_chunks', 0)
    identical = stats.get('identical_chunks', 0)
    
    print(f"  Total chunks: {total:,}")
    print(f"  Enhanced chunks: {enhanced:,}")
    print(f"  Identical chunks: {identical:,}")
    print(f"  Table chunks: {stats.get('table_chunks', 0):,}")
    print(f"  Text chunks: {stats.get('text_chunks', 0):,}")
    
    if total > 0:
        enhancement_rate = (enhanced / total) * 100
        print(f"  Enhancement rate: {enhancement_rate:.1f}%")
    
    print(f"  Average similarity: {stats.get('average_similarity', 0):.3f}")
    print(f"  Average improvement ratio: {stats.get('average_improvement_ratio', 0):.2f}x")
    
    examples = stats.get('enhancement_examples', [])
    if examples:
        print(f"  Best improvements: {len(examples)} examples found")


def main():
    parser = argparse.ArgumentParser(description="Analyze chunk text enhancements")
    parser.add_argument('chunks_file', nargs='?', default='contextualized_chunks.json',
                       help='Chunks JSON file to analyze (default: contextualized_chunks.json)')
    parser.add_argument('-s', '--samples', type=int, default=3,
                       help='Number of sample enhancements to show (default: 3)')
    parser.add_argument('-c', '--compare', type=str,
                       help='Compare with old chunks file (provide old file path)')
    parser.add_argument('--examples-only', action='store_true',
                       help='Only show examples, skip detailed analysis')
    
    args = parser.parse_args()
    
    if args.examples_only:
        show_sample_improvements(args.chunks_file, args.samples)
        return 0
    
    # Run full analysis
    print(f"{'='*70}")
    print("TEXT ENHANCEMENT ANALYSIS")
    print(f"{'='*70}")
    print(f"Analyzing file: {args.chunks_file}")
    
    stats = analyze_chunk_differences(args.chunks_file)
    
    if not stats:
        return 1
    
    print_stats_summary(stats)
    
    # Show comparison if requested
    if args.compare:
        compare_before_after(args.compare, args.chunks_file)
    
    # Show sample improvements
    show_sample_improvements(args.chunks_file, args.samples)
    
    # Show top enhancement examples
    examples = stats.get('enhancement_examples', [])
    if examples:
        print(f"\n{'='*70}")
        print(f"TOP ENHANCEMENT EXAMPLES")
        print(f"{'='*70}")
        
        for i, example in enumerate(examples[:3]):
            print(f"\n{i+1}. {example['chunk_id']}")
            print(f"   Document: {example['document_name']}")
            print(f"   Table: {example['is_table']}")
            print(f"   Improvement: {example['improvement_ratio']:.2f}x")
            print(f"   Similarity: {example['similarity']:.3f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
