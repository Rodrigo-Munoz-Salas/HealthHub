"""
Health RAG Tool for Google ADK Agent
Tool definition for integrating RAG pipeline with ADK agent
"""

import asyncio
import logging
from typing import Dict, Any, List
from google.adk import Tool, ToolCall
from google.adk.types import Message

# Local imports
from mobile_rag_system import MobileHealthRAG

logger = logging.getLogger(__name__)

class HealthRAGTool:
    """Tool for health emergency RAG pipeline integration with ADK agent"""
    
    def __init__(self, mobile_rag: MobileHealthRAG):
        self.mobile_rag = mobile_rag
        self.tool = Tool(
            name="health_emergency_rag",
            description="Process health emergency queries using RAG pipeline with TinyLlama and vector database",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Health emergency query or description"
                    },
                    "emergency_type": {
                        "type": "string",
                        "description": "Type of emergency (chest_pain, fainting, burn, choking, stroke, general)",
                        "enum": ["chest_pain", "fainting", "burn", "choking", "stroke", "general"]
                    }
                },
                "required": ["query"]
            }
        )
        
        logger.info("🔧 Health RAG Tool initialized")
    
    async def call(self, tool_call: ToolCall) -> str:
        """Execute health emergency RAG tool call"""
        try:
            # Extract parameters from tool call
            query = tool_call.parameters.get("query", "")
            emergency_type = tool_call.parameters.get("emergency_type", "general")
            
            logger.info(f"🔍 Processing health emergency query: {query[:50]}...")
            logger.info(f"🚨 Emergency type: {emergency_type}")
            
            # Process through mobile RAG system
            rag_response = self.mobile_rag.query_emergency(query)
            
            # Format response for ADK agent
            formatted_response = self._format_rag_response(rag_response, emergency_type)
            
            logger.info(f"✅ Health emergency response generated")
            return formatted_response
            
        except Exception as e:
            logger.error(f"❌ Error in health RAG tool: {e}")
            return self._get_fallback_response()
    
    def _format_rag_response(self, rag_response: Dict[str, Any], emergency_type: str) -> str:
        """Format RAG response for ADK agent"""
        try:
            # Extract key information
            emergency_type_detected = rag_response.get("emergency_type", "general")
            immediate_actions = rag_response.get("immediate_actions", [])
            call_911 = rag_response.get("call_911", False)
            confidence = rag_response.get("confidence", 0.5)
            response_time = rag_response.get("query_time", 0)
            
            # Format response
            response_parts = []
            
            # Emergency type and urgency
            if call_911:
                response_parts.append("🚨 **CALL 911 IMMEDIATELY** 🚨")
            else:
                response_parts.append("🏥 **Health Emergency Guidance**")
            
            # Immediate actions
            if immediate_actions:
                response_parts.append("\n**Immediate Actions:**")
                for i, action in enumerate(immediate_actions, 1):
                    response_parts.append(f"{i}. {action}")
            
            # Additional guidance
            if emergency_type_detected != "general":
                response_parts.append(f"\n**Emergency Type:** {emergency_type_detected.replace('_', ' ').title()}")
            
            # Confidence and response time
            response_parts.append(f"\n**Response Time:** {response_time:.2f} seconds")
            response_parts.append(f"**Confidence:** {confidence:.1%}")
            
            # Safety reminder
            response_parts.append("\n⚠️ **Important:** This is AI guidance. Always call 911 for life-threatening emergencies.")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"❌ Error formatting RAG response: {e}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """Get fallback response when RAG system fails"""
        return """🚨 **Emergency Response Unavailable**

I'm unable to process your health emergency query at the moment.

**Immediate Actions:**
1. Call 911 immediately for life-threatening emergencies
2. Seek immediate medical attention
3. Do not delay emergency care

**Safety First:** Always prioritize calling 911 for medical emergencies."""
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information for ADK agent"""
        return {
            "name": self.tool.name,
            "description": self.tool.description,
            "parameters": self.tool.parameters,
            "mobile_rag_status": self.mobile_rag.get_system_status()
        }

# Tool function for ADK agent
async def health_emergency_rag_tool(query: str, emergency_type: str = "general") -> str:
    """Tool function for health emergency RAG processing"""
    try:
        # This would be called by the ADK agent
        # In practice, this would be integrated with the mobile RAG system
        return f"Health emergency response for: {query} (Type: {emergency_type})"
    except Exception as e:
        return f"Error processing health emergency query: {e}"
