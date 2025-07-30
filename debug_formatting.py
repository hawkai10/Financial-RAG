#!/usr/bin/env python3
"""
Debug script to test AI response formatting
"""

from api_server import format_ai_response
from utils import logger

def test_response_formatting():
    """Test the AI response formatting with sample problematic responses"""
    
    # Sample problematic response like what you're seeing
    sample_response = """1. Total Count/Summary: Bhartiya Enterprises issued credit notes to 4 different companies.

2. Detailed Breakdown:
| Credit Note No. | Consignee/Buyer | Amount Chargeable (in Rupees) | Tax Amount (in Rupees) | Invoice Date (Original) |
|---|---|---|---|---|
| 11 | Krishna Prabhash Agro Oil Pvt Ltd | 42,102 | 2,085 | 16-Jun-19 |
| CN/24 | Vijay Agro Products Pvt Ltd | 254,434 | 12,115.90 | 30-Jul-22 |
| 10 | Sri Venkta Srinivasa Oils Pvt Ltd | 51,953 | 2,473.95 | 21-Oct-19 |
| CN/6 | Seetharama Oil Industies (P) Ltd | 249,959 | 11,902.80 | 2-Jun-22 |

3. List of Relevant Items:
• Krishna Prabhash Agro Oil Pvt Ltd: Credit Note #11
• Vijay Agro Products Pvt Ltd: Credit Note # CN/24
• Sri Venkta Srinivasa Oils Pvt Ltd: Credit Note #10
• Seetharama Oil Industies (P) Ltd: Credit Note # CN/6"""

    logger.info("Testing AI response formatting...")
    logger.info(f"Input response length: {len(sample_response)}")
    logger.info(f"Input response preview: {sample_response[:200]}...")
    
    # Test the formatting function
    formatted = format_ai_response(sample_response)
    
    logger.info("\n=== FORMATTED RESULT ===")
    logger.info(f"Summary: {formatted['summary']}")
    logger.info(f"Items count: {len(formatted['items'])}")
    
    for i, item in enumerate(formatted['items']):
        logger.info(f"\nItem {i+1}:")
        logger.info(f"  Title: {item['title']}")
        logger.info(f"  Text length: {len(item['text'])}")
        logger.info(f"  Text preview: {item['text'][:200]}...")

if __name__ == "__main__":
    test_response_formatting()
