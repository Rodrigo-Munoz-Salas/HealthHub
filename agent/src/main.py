"""
Google ADK Agent for Mobile Health Emergency RAG
Main entry point for the ADK agent integration
"""

import asyncio
import logging
from typing import Dict, Any
from pathlib import Path

# Google ADK imports
from google.adk import Agent, Tool, ToolCall
from google.adk.types import Message, UserMessage, AssistantMessage

# Local imports
from health_rag_tool import HealthRAGTool
from mobile_rag_system import MobileHealthRAG
from agent_config import get_agent_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthEmergencyAgent:
    """Google ADK Agent for Health Emergency RAG"""
    
    def __init__(self):
        self.agent_config = get_agent_config()
        self.mobile_rag = MobileHealthRAG()
        self.health_rag_tool = HealthRAGTool(self.mobile_rag)
        
        # Initialize ADK agent
        self.agent = Agent(
            name=self.agent_config["name"],
            description=self.agent_config["description"],
            tools=[self.health_rag_tool],
            model=self.agent_config["model"],
            system_prompt=self.agent_config["system_prompt"]
        )
        
        logger.info("🏥 Health Emergency ADK Agent initialized")
        logger.info(f"📱 Mobile RAG system loaded: {self.mobile_rag.get_system_status()}")
    
    async def process_user_input(self, user_input: str) -> str:
        """Process user input through ADK agent and RAG pipeline"""
        try:
            logger.info(f"🔍 Processing user input: {user_input[:50]}...")
            
            # Create user message
            user_message = UserMessage(content=user_input)
            
            # Process through ADK agent
            response = await self.agent.process_message(user_message)
            
            logger.info(f"✅ Response generated: {response.content[:50]}...")
            return response.content
            
        except Exception as e:
            logger.error(f"❌ Error processing user input: {e}")
            return "I'm sorry, I cannot process your health emergency request at the moment. Please try again or call 911 for immediate assistance."
    
    async def handle_emergency_query(self, query: str) -> Dict[str, Any]:
        """Handle health emergency queries with detailed response"""
        try:
            logger.info(f"🚨 Handling emergency query: {query[:50]}...")
            
            # Process through RAG pipeline
            rag_response = self.mobile_rag.query_emergency(query)
            
            # Format response for ADK agent
            formatted_response = self._format_emergency_response(rag_response)
            
            logger.info(f"✅ Emergency response generated: {formatted_response['summary'][:50]}...")
            return formatted_response
            
        except Exception as e:
            logger.error(f"❌ Error handling emergency query: {e}")
            return {
                "error": "Unable to process emergency query",
                "fallback": "Call 911 immediately for medical emergencies",
                "timestamp": asyncio.get_event_loop().time()
            }
    
    def _format_emergency_response(self, rag_response: Dict[str, Any]) -> Dict[str, Any]:
        """Format RAG response for ADK agent"""
        return {
            "summary": rag_response.get("response", "Emergency guidance not available"),
            "emergency_type": rag_response.get("emergency_type", "unknown"),
            "immediate_actions": rag_response.get("immediate_actions", []),
            "call_911": rag_response.get("call_911", False),
            "confidence": rag_response.get("confidence", 0.5),
            "response_time": rag_response.get("query_time", 0),
            "timestamp": rag_response.get("timestamp", 0)
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status and system information"""
        return {
            "agent_name": self.agent_config["name"],
            "agent_description": self.agent_config["description"],
            "mobile_rag_status": self.mobile_rag.get_system_status(),
            "tools_available": [tool.name for tool in self.agent.tools],
            "model": self.agent_config["model"],
            "system_ready": True
        }

async def main():
    """Main function to run the ADK agent"""
    print("🏥 Google ADK Health Emergency RAG Agent")
    print("=" * 60)
    print("Initializing ADK agent with mobile RAG system...")
    print()
    
    try:
        # Initialize health emergency agent
        health_agent = HealthEmergencyAgent()
        
        # Display agent status
        status = health_agent.get_agent_status()
        print("🤖 Agent Status:")
        print(f"   Name: {status['agent_name']}")
        print(f"   Description: {status['agent_description']}")
        print(f"   Model: {status['model']}")
        print(f"   Tools: {', '.join(status['tools_available'])}")
        print()
        
        # Display mobile RAG status
        rag_status = status['mobile_rag_status']
        print("📱 Mobile RAG Status:")
        print(f"   Guidelines: {rag_status['guidelines_loaded']}")
        print(f"   Vector DB: {rag_status['vector_db_loaded']}")
        print(f"   Emergency Protocols: {rag_status['emergency_protocols']}")
        print(f"   System Ready: {rag_status['system_ready']}")
        print()
        
        # Interactive mode
        print("💬 Interactive Mode - Enter health emergency queries:")
        print("   Type 'quit' to exit")
        print("   Type 'status' to check system status")
        print("   Type 'help' for available commands")
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("🏥 Health Emergency Query: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye! Stay safe!")
                    break
                elif user_input.lower() == 'status':
                    status = health_agent.get_agent_status()
                    print(f"📊 System Status: {status['mobile_rag_status']}")
                    continue
                elif user_input.lower() == 'help':
                    print("📋 Available Commands:")
                    print("   - Enter health emergency description")
                    print("   - 'status' - Check system status")
                    print("   - 'quit' - Exit application")
                    continue
                elif not user_input:
                    continue
                
                # Process user input
                print("🔍 Processing your health emergency query...")
                response = await health_agent.process_user_input(user_input)
                
                print(f"🏥 Health Emergency Response:")
                print(f"   {response}")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Stay safe!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 Please try again or call 911 for immediate assistance")
                print()
    
    except Exception as e:
        print(f"❌ Failed to initialize ADK agent: {e}")
        print("💡 Check your mobile RAG system setup")

if __name__ == "__main__":
    asyncio.run(main())
