#!/usr/bin/env python3

"""
Test script to verify vLLM server integration with Graph of Thoughts framework.

Usage:
1. Start your vLLM server:
   python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf --port 8000

2. Run this test:
   python test_vllm_integration.py

3. For GoT sorting example:
   python test_vllm_integration.py --run-got-example
"""

import sys
import argparse
import time
import logging
import os

# Add current directory and examples to Python path
sys.path.insert(0, '.')  # Current directory (we're now inside graph-of-thoughts)
sys.path.insert(0, 'examples/sorting')

from graph_of_thoughts import language_models, controller, operations
from sorting_032 import SortingPrompter, SortingParser, got, utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_connection():
    """Test basic connection to vLLM server."""
    print("Testing basic vLLM server connection...")
    
    try:
        # Create vLLM client (using small model for quick testing)
        lm = language_models.vLLMClient("vllm_config.json", model_name="vllm_small")
        
        # Simple test query
        test_prompt = "Hello! Please respond with 'Connection successful!'"
        
        print(f"Sending test query: {test_prompt}")
        response = lm.query(test_prompt, num_responses=1)
        
        # Extract response text
        response_texts = lm.get_response_texts(response)
        print(f"Response: {response_texts[0]}")
        
        print("✓ Basic connection test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False


def test_got_sorting():
    """Test GoT sorting example with vLLM."""
    print("\nTesting Graph of Thoughts sorting example with vLLM...")
    print("NOTE: Using small model (1.1B) for quick testing - accuracy may be limited")
    
    try:
        # Create vLLM client (using small model for quick testing)
        lm = language_models.vLLMClient("vllm_config.json", model_name="vllm_small")
        
        # Test problem - small list for quick testing
        test_list = "[3, 1, 4, 1, 5, 9, 2, 6]"
        print(f"Test problem: Sort {test_list}")
        
        # Create GoT graph (using simplified version for testing)
        gop = operations.GraphOfOperations()
        gop.append_operation(operations.Generate(1, 1))  # Generate 1 solution
        gop.append_operation(operations.Score(1, False, utils.num_errors))
        gop.append_operation(operations.GroundTruth(utils.test_sorting))
        
        # Create controller
        ctrl = controller.Controller(
            lm,
            gop,
            SortingPrompter(),
            SortingParser(),
            {
                "original": test_list,
                "current": "",
                "method": "cot",  # Use chain-of-thought for simple test
                "phase": 0
            }
        )
        
        # Run the controller
        print("Running GoT sorting...")
        start_time = time.time()
        ctrl.run()
        end_time = time.time()
        
        # Get results
        final_thoughts = ctrl.get_final_thoughts()
        
        print(f"✓ GoT sorting completed in {end_time - start_time:.2f} seconds")
        print(f"Final thoughts: {len(final_thoughts)} operations completed")
        
        # Print some basic metrics
        print(f"Total prompt tokens: {lm.prompt_tokens}")
        print(f"Total completion tokens: {lm.completion_tokens}")
        print(f"Estimated cost: ${lm.cost:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ GoT sorting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test vLLM integration with GoT")
    parser.add_argument("--run-got-example", action="store_true", 
                       help="Run full GoT sorting example")
    parser.add_argument("--config", default="vllm_config.json",
                       help="Path to vLLM config file")
    
    args = parser.parse_args()
    
    print("vLLM Integration Test")
    print("=" * 40)
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"✗ Config file {args.config} not found!")
        print("Make sure vllm_config.json exists in the current directory.")
        return 1
    
    # Test basic connection first
    if not test_basic_connection():
        print("\n❌ Basic connection failed. Is your vLLM server running?")
        print("Start server with: python -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8000")
        print("(Or use --model meta-llama/Llama-2-7b-chat-hf for better quality)")
        return 1
    
    # Run GoT example if requested
    if args.run_got_example:
        if test_got_sorting():
            print("\n🎉 All tests passed! vLLM integration is working correctly.")
        else:
            print("\n❌ GoT sorting test failed.")
            return 1
    else:
        print("\n✓ Basic test passed. Use --run-got-example to test full GoT integration.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())