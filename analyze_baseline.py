#!/usr/bin/env python3
"""
vLLM + GoT Baseline Analysis Script
Analyzes a specific Graph of Thoughts experiment run
"""

import json
import glob
import os
import sys
from typing import Dict, List
import statistics

def extract_results(experiment_name: str) -> List[Dict]:
    """
    Extract results from a specific experiment run.
    
    :param experiment_name: Name of experiment (e.g., 'vllm_got_2025-09-07_15-55-48')
    :return: List of result dictionaries
    """
    results = []
    
    # Build path to experiment results
    experiment_path = os.path.join("results", experiment_name, "got")
    
    if not os.path.exists(experiment_path):
        print(f"ERROR: Experiment path not found: {experiment_path}")
        return []
    
    # Find all JSON result files in the got directory
    json_files = glob.glob(os.path.join(experiment_path, "*.json"))
    
    if not json_files:
        print(f"ERROR: No JSON files found in: {experiment_path}")
        return []
    
    print(f"Found {len(json_files)} result files in {experiment_name}")
    
    for file_path in sorted(json_files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract sample ID from filename
            sample_id = int(os.path.basename(file_path).replace('.json', ''))
            
            # Find the ground_truth_evaluator operation
            ground_truth_op = None
            for op in data:
                if isinstance(op, dict) and op.get("operation") == "ground_truth_evaluator":
                    ground_truth_op = op
                    break
            
            if not ground_truth_op:
                print(f"WARNING: No ground_truth_evaluator found in {file_path}")
                continue
                
            # Extract final thought state
            final_thought = ground_truth_op["thoughts"][0]
            
            # Find token usage (last entry in JSON)
            token_info = data[-1]
            
            result = {
                'sample_id': sample_id,
                'file_path': file_path,
                'original': final_thought["original"],
                'model_output': final_thought["current"],
                'solved': final_thought.get("problem_solved", [False])[0],
                'error_score': ground_truth_op["scores"][0],
                'prompt_tokens': token_info.get("prompt_tokens", 0),
                'completion_tokens': token_info.get("completion_tokens", 0),
                'total_tokens': token_info.get("prompt_tokens", 0) + token_info.get("completion_tokens", 0)
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"ERROR processing {file_path}: {e}")
            continue
    
    return results

def analyze_results(results: List[Dict]) -> Dict:
    """
    Analyze extracted results and calculate key metrics.
    
    :param results: List of result dictionaries
    :return: Analysis summary
    """
    if not results:
        return {}
    
    total_samples = len(results)
    solved_count = sum(1 for r in results if r['solved'])
    error_scores = [r['error_score'] for r in results]
    prompt_tokens = [r['prompt_tokens'] for r in results]
    completion_tokens = [r['completion_tokens'] for r in results]
    total_tokens = [r['total_tokens'] for r in results]
    
    analysis = {
        'total_samples': total_samples,
        'solved_count': solved_count,
        'success_rate': solved_count / total_samples,
        'avg_error_score': statistics.mean(error_scores),
        'error_distribution': {i: error_scores.count(i) for i in range(max(error_scores) + 1)},
        'avg_prompt_tokens': statistics.mean(prompt_tokens),
        'avg_completion_tokens': statistics.mean(completion_tokens),
        'avg_total_tokens': statistics.mean(total_tokens),
        'total_tokens_used': sum(total_tokens),
        'min_error': min(error_scores),
        'max_error': max(error_scores)
    }
    
    return analysis

def print_baseline_report(analysis: Dict, results: List[Dict], experiment_name: str):
    """
    Print a comprehensive baseline report.
    
    :param analysis: Analysis summary
    :param results: Raw results for detailed breakdown
    :param experiment_name: Name of the experiment being analyzed
    """
    print("\n" + "=" * 60)
    print("vLLM + GoT BASELINE EVALUATION REPORT")
    print("=" * 60)
    # Auto-detect task size from experiment name or data
    task_size = "32"
    if "064" in experiment_name:
        task_size = "64"
    elif results and len(results) > 0:
        # Try to detect from actual data
        original_data = results[0].get('original', '')
        if original_data:
            try:
                # Count elements in the original list string
                import re
                numbers = re.findall(r'\d+', original_data)
                if len(numbers) >= 60:
                    task_size = "64"
            except:
                pass
    
    print(f"Experiment: {experiment_name}")
    print(f"Model: Qwen2-7B-Instruct")
    print(f"Method: Graph of Thoughts (GoT)")
    print(f"Task: {task_size}-element list sorting")
    print()
    
    # Core Performance Metrics
    print("PERFORMANCE METRICS:")
    print(f"   Success Rate: {analysis['success_rate']:.1%} ({analysis['solved_count']}/{analysis['total_samples']})")
    print(f"   Average Error Score: {analysis['avg_error_score']:.2f}")
    print(f"   Error Range: {analysis['min_error']} - {analysis['max_error']}")
    print()
    
    # Error Distribution
    print("ERROR DISTRIBUTION:")
    for error_score, count in analysis['error_distribution'].items():
        percentage = count / analysis['total_samples'] * 100
        print(f"   Score {error_score}: {count} samples ({percentage:.1f}%)")
    print()
    
    # Token Usage
    print("TOKEN USAGE:")
    print(f"   Avg Prompt Tokens: {analysis['avg_prompt_tokens']:.0f}")
    print(f"   Avg Completion Tokens: {analysis['avg_completion_tokens']:.0f}")
    print(f"   Avg Total Tokens/Sample: {analysis['avg_total_tokens']:.0f}")
    print(f"   Total Tokens Used: {analysis['total_tokens_used']:,}")
    print()
    
    # Sample-by-Sample Breakdown
    print("DETAILED BREAKDOWN:")
    results_sorted = sorted(results, key=lambda x: x['sample_id'])
    for r in results_sorted:
        status = "SOLVED" if r['solved'] else "FAILED"
        print(f"   Sample {r['sample_id']}: {status} (Error Score: {r['error_score']}, Tokens: {r['total_tokens']})")
    
    print("\n" + "=" * 60)

def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python analyze_baseline.py <experiment_name>")
        print("Example: python analyze_baseline.py vllm_got_2025-09-07_15-55-48")
        return
    
    experiment_name = sys.argv[1]
    print(f"Analyzing experiment: {experiment_name}")
    
    # Extract results
    results = extract_results(experiment_name)
    
    if not results:
        print("ERROR: No results to analyze. Check the experiment name and path.")
        return
    
    # Analyze results  
    analysis = analyze_results(results)
    
    # Print report
    print_baseline_report(analysis, results, experiment_name)
    
    # Save raw data for further analysis
    output_file = f"{experiment_name}_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'experiment_name': experiment_name,
            'analysis': analysis,
            'raw_results': results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()