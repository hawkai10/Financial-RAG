"""
Full Agent Framework for Complex Multi-Step Reasoning
Handles analytical queries that require planning and multi-step execution
"""

import json
import re
from typing import Dict, List, Any, Optional
from utils import logger
from chunk_manager import ChunkManager
from progressive_retrieval import ProgressiveRetriever

class FullAgent:
    """Full Agent for complex reasoning and multi-step analysis."""
    
    def __init__(self, chunk_manager: ChunkManager, progressive_retrieval: ProgressiveRetriever, gemini_api_func):
        self.chunk_manager = chunk_manager
        self.progressive_retrieval = progressive_retrieval
        self.call_gemini = gemini_api_func
        
        # Templates for different agent operations
        self.plan_generation_template = """
You are a business analyst planning how to answer a complex question. Break down this question into logical steps.

QUESTION: {query}

Create a step-by-step plan to answer this question thoroughly. Each step should be a specific sub-question that can be researched independently.

Response format (JSON only):
{{
    "plan_steps": [
        {{"step": 1, "description": "Find Q1 revenue data", "sub_query": "Q1 revenue financial results"}},
        {{"step": 2, "description": "Find Q2 revenue data", "sub_query": "Q2 revenue financial results"}},
        {{"step": 3, "description": "Compare and analyze differences", "sub_query": "revenue trends analysis"}}
    ],
    "complexity_assessment": "moderate|complex",
    "estimated_steps": 3,
    "reasoning": "Brief explanation of the approach"
}}

Return valid JSON only.
"""
        
        self.synthesis_template = """
You are analyzing business documents to provide a comprehensive answer.

ORIGINAL QUESTION: {query}

RESEARCH PLAN EXECUTED:
{plan_summary}

GATHERED INFORMATION:
{accumulated_context}

Based on the above research and information, provide a comprehensive, well-reasoned answer to the original question. 

Requirements:
1. Use only the information provided in the gathered research
2. Clearly explain your reasoning and analysis
3. If comparing data, show specific numbers and differences
4. If analyzing trends, explain the patterns and implications
5. Structure your answer logically with clear sections
6. If information is insufficient, clearly state what's missing

Provide a detailed, analytical response:
"""
        
        self.verification_template = """
Review this answer for accuracy and grounding in the provided source material.

QUESTION: {query}
ANSWER: {answer}
SOURCE MATERIAL: {context}

Check if the answer is:
1. Factually grounded in the source material
2. Free from hallucinations or assumptions
3. Logically consistent

Response format (JSON only):
{{
    "is_grounded": true|false,
    "confidence_score": 0.85,
    "issues_found": ["list of any issues"],
    "suggestions": ["suggestions for improvement"],
    "verified": true|false
}}

Return valid JSON only.
"""
        
        logger.info("Full Agent initialized")
    
    async def process_complex_query(self, query: str, complexity_level: str = "moderate", **kwargs) -> Dict[str, Any]:
        """Process complex analytical query with multi-step reasoning."""
        
        try:
            logger.info(f"Full Agent processing complex query (complexity: {complexity_level})")
            
            # Step 1: Generate execution plan
            plan = await self._generate_plan(query)
            if not plan or not plan.get('plan_steps'):
                # Fallback to single-step analysis
                return await self._single_step_analysis(query)
            
            # Step 2: Execute plan steps
            accumulated_context = await self._execute_plan_steps(plan['plan_steps'], query)
            
            # Step 3: Synthesize final answer
            final_answer = await self._synthesize_answer(query, plan, accumulated_context)
            
            # Step 4: Verify answer quality
            verification = await self._verify_answer(query, final_answer, accumulated_context)
            
            return self._format_complex_response(
                query, final_answer, plan, accumulated_context, verification, complexity_level
            )
            
        except Exception as e:
            logger.error(f"Full Agent processing failed: {e}")
            return self._create_error_response(query, str(e))
    
    async def _generate_plan(self, query: str) -> Optional[Dict[str, Any]]:
        """Generate execution plan for complex query."""
        
        try:
            prompt = self.plan_generation_template.format(query=query)
            response = await self.call_gemini(prompt)
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(0))
                logger.info(f"Generated plan with {len(plan_data.get('plan_steps', []))} steps")
                return plan_data
            
            return None
            
        except Exception as e:
            logger.warning(f"Plan generation failed: {e}")
            return None
    
    async def _execute_plan_steps(self, plan_steps: List[Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
        """Execute each step of the plan."""
        
        accumulated_context = []
        
        for step in plan_steps:
            try:
                step_num = step.get('step', 0)
                sub_query = step.get('sub_query', '')
                description = step.get('description', '')
                
                logger.info(f"Executing step {step_num}: {description}")
                
                # Retrieve relevant chunks for this step
                chunks = await self._retrieve_chunks_for_step(sub_query)
                
                # Store step results
                step_result = {
                    'step_number': step_num,
                    'description': description,
                    'sub_query': sub_query,
                    'chunks_found': len(chunks),
                    'chunks': chunks[:10],  # Limit to top 10 chunks per step
                    'success': len(chunks) > 0
                }
                
                accumulated_context.append(step_result)
                
            except Exception as e:
                logger.warning(f"Step {step_num} failed: {e}")
                accumulated_context.append({
                    'step_number': step_num,
                    'description': description,
                    'error': str(e),
                    'success': False
                })
        
        return accumulated_context
    
    async def _retrieve_chunks_for_step(self, sub_query: str) -> List[Dict[str, Any]]:
        """Retrieve chunks for a specific plan step."""
        
        try:
            chunks, retrieval_info = await self.progressive_retrieval.retrieve_progressively(
                queries=[sub_query],
                strategy="Analyse",
                confidence=0.4
            )
            
            return chunks
            
            
        except Exception as e:
            logger.error(f"Chunk retrieval for step failed: {e}")
            return []
    
    async def _synthesize_answer(self, query: str, plan: Dict[str, Any], context: List[Dict[str, Any]]) -> str:
        """Synthesize final answer from gathered context."""
        
        try:
            # Prepare plan summary
            plan_summary = []
            for step in plan.get('plan_steps', []):
                plan_summary.append(f"Step {step['step']}: {step['description']}")
            plan_text = "\n".join(plan_summary)
            
            # Prepare context summary
            context_summary = []
            for step_result in context:
                if step_result.get('success') and step_result.get('chunks'):
                    context_summary.append(f"\n--- Step {step_result['step_number']}: {step_result['description']} ---")
                    for chunk in step_result['chunks']:
                        context_summary.append(chunk.get('chunk_text', '')[:500] + "...")
            
            context_text = "\n".join(context_summary)
            
            # Generate synthesis prompt
            prompt = self.synthesis_template.format(
                query=query,
                plan_summary=plan_text,
                accumulated_context=context_text
            )
            
            # Get final answer
            answer = await self.call_gemini(prompt)
            return answer
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return f"Failed to synthesize comprehensive answer: {e}"
    
    async def _verify_answer(self, query: str, answer: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify answer quality and grounding."""
        
        try:
            # Prepare context for verification
            context_text = ""
            for step_result in context:
                if step_result.get('chunks'):
                    for chunk in step_result['chunks'][:3]:  # Sample chunks
                        context_text += chunk.get('chunk_text', '')[:200] + "... "
            
            prompt = self.verification_template.format(
                query=query,
                answer=answer,
                context=context_text
            )
            
            response = await self.call_gemini(prompt)
            
            # Parse verification response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            return {"verified": False, "confidence_score": 0.5}
            
        except Exception as e:
            logger.warning(f"Answer verification failed: {e}")
            return {"verified": False, "confidence_score": 0.3, "error": str(e)}
    
    async def _single_step_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback to single-step analysis when planning fails."""
        
        try:
            # Retrieve chunks for the query
            chunks, retrieval_info = await self.progressive_retrieval.retrieve_progressively(
                queries=[query],
                strategy="Analyse",
                confidence=0.4
            )
            
            if not chunks:
                return self._create_error_response(query, "No relevant information found")
            
            # Create context from chunks
            context_text = "\n\n".join([
                chunk.get('chunk_text', '') for chunk in chunks[:15]
            ])
            
            # Generate answer
            prompt = f"""
Based on the following business documents, provide a comprehensive analysis for this question:

QUESTION: {query}

RELEVANT INFORMATION:
{context_text}

Provide a detailed, analytical response based solely on the provided information:
"""
            
            answer = await self.call_gemini(prompt)
            
            return {
                'answer': answer,
                'strategy_used': 'Full-Agent-SingleStep',
                'complexity_level': 'moderate',
                'chunks_used': len(chunks),
                'plan_steps': 1,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Single-step analysis failed: {e}")
            return self._create_error_response(query, str(e))
    
    def _format_complex_response(self, query: str, answer: str, plan: Dict[str, Any], 
                                context: List[Dict[str, Any]], verification: Dict[str, Any],
                                complexity_level: str) -> Dict[str, Any]:
        """Format the complex analysis response."""
        
        # Calculate success metrics
        successful_steps = sum(1 for step in context if step.get('success', False))
        total_chunks = sum(step.get('chunks_found', 0) for step in context)
        
        return {
            'answer': answer,
            'strategy_used': 'Full-Agent',
            'complexity_level': complexity_level,
            'plan_executed': {
                'total_steps': len(plan.get('plan_steps', [])),
                'successful_steps': successful_steps,
                'step_details': [
                    {
                        'step': step.get('step_number'),
                        'description': step.get('description'),
                        'success': step.get('success', False),
                        'chunks_found': step.get('chunks_found', 0)
                    }
                    for step in context
                ]
            },
            'verification': verification,
            'total_chunks_processed': total_chunks,
            'confidence_score': verification.get('confidence_score', 0.7),
            'success': successful_steps > 0
        }
    
    def _create_error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Create error response."""
        
        return {
            'answer': f"Unable to process complex query due to error: {error}",
            'strategy_used': 'Full-Agent',
            'success': False,
            'error': error,
            'should_fallback': True
        }

# Global instance - will be initialized in rag_backend.py
full_agent = None

def initialize_full_agent(chunk_manager: ChunkManager, progressive_retrieval: ProgressiveRetriever, gemini_api_func):
    """Initialize the global full-agent instance."""
    global full_agent
    full_agent = FullAgent(chunk_manager, progressive_retrieval, gemini_api_func)
    return full_agent

__all__ = ['FullAgent', 'initialize_full_agent', 'full_agent']
