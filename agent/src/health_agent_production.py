#!/usr/bin/env python3
"""
Health Agent Production System
Google Agent ADK wrapper for RAG-based health emergency detection
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthAgent:
    """
    Google Agent ADK implementation for health emergency detection
    Provides concise, actionable health advice in 2-3 sentences max
    """
    
    def __init__(self, rag_system=None):
        """Initialize the Health Agent with RAG system"""
        self.rag_system = rag_system
        self.agent_id = "health-emergency-agent"
        self.version = "1.0.0"
        
        # Agent capabilities
        self.capabilities = [
            "emergency_detection",
            "symptom_analysis", 
            "medical_advice",
            "global_health_guidance"
        ]
        
        # Response templates for consistent formatting
        self.response_templates = {
            "emergency": "Emergency: Yes. Action: {action}. {symptom_info}",
            "non_emergency": "Emergency: No. Action: {action}. {symptom_info}",
            "uncertain": "Emergency: Uncertain. Action: {action}. {symptom_info}"
        }
    
    def process_health_query(self, query: str) -> Dict[str, Any]:
        """
        Process health query through RAG system with agent wrapper
        
        Args:
            query: User's health-related query
            
        Returns:
            Dict containing agent response and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"🏥 Health Agent processing query: {query[:100]}...")
            
            # Get RAG system response
            if self.rag_system:
                rag_result = self.rag_system.query_health_emergency(query)
            else:
                # Fallback if no RAG system
                rag_result = self._fallback_response(query)
            
            # Process through agent logic
            agent_response = self._process_agent_response(query, rag_result)
            
            # Format final response
            final_response = self._format_agent_response(agent_response, query)
            
            processing_time = time.time() - start_time
            
            return {
                "agent_id": self.agent_id,
                "query": query,
                "response": final_response,
                "emergency_detected": agent_response.get("is_emergency", False),
                "confidence": agent_response.get("confidence", 0.5),
                "processing_time": processing_time,
                "timestamp": time.time(),
                "metadata": {
                    "rag_result": rag_result,
                    "agent_version": self.version,
                    "capabilities_used": self._get_used_capabilities(agent_response)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Health Agent error: {e}")
            return self._error_response(query, str(e))
    
    def _process_agent_response(self, query: str, rag_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process RAG result through agent logic"""
        
        # Extract key information
        emergency_type = rag_result.get('emergency_type', 'unknown')
        ai_response = rag_result.get('ai_response', '')
        natural_response = rag_result.get('natural_response', '')
        
        # Determine if emergency
        is_emergency = self._determine_emergency_status(emergency_type, ai_response, natural_response)
        
        # Extract symptoms
        symptoms = self._extract_symptoms(query)
        
        # Generate appropriate action
        action = self._generate_action(is_emergency, symptoms, emergency_type)
        
        # Calculate confidence
        confidence = self._calculate_confidence(ai_response, natural_response, is_emergency)
        
        return {
            "is_emergency": is_emergency,
            "action": action,
            "symptoms": symptoms,
            "confidence": confidence,
            "emergency_type": emergency_type
        }
    
    def _determine_emergency_status(self, emergency_type: str, ai_response: str, natural_response: str) -> bool:
        """Determine if situation is emergency based on multiple signals"""
        
        # Check emergency type
        if emergency_type in ['chest_pain', 'stroke', 'severe_injury', 'poisoning', 'choking']:
            return True
        
        # Check AI response for emergency indicators
        if ai_response:
            emergency_indicators = ['emergency: yes', 'call 911', 'go to er', 'seek hospital', 'immediate']
            if any(indicator in ai_response.lower() for indicator in emergency_indicators):
                return True
        
        # Check natural response
        if natural_response:
            if 'emergency' in natural_response.lower() and 'yes' in natural_response.lower():
                return True
        
        return False
    
    def _extract_symptoms(self, query: str) -> str:
        """Extract key symptoms from query"""
        symptoms = []
        query_lower = query.lower()
        
        # Common emergency symptoms
        symptom_mapping = {
            'chest pain': ['chest pain', 'chest discomfort', 'chest tightness'],
            'breathing difficulty': ['shortness of breath', 'can\'t breathe', 'breathing', 'wheezing'],
            'severe headache': ['severe headache', 'head pain', 'migraine'],
            'dizziness': ['dizzy', 'dizziness', 'faint', 'lightheaded'],
            'fever': ['fever', 'high temperature', 'hot'],
            'nausea': ['nausea', 'vomiting', 'sick', 'nauseous'],
            'head trauma': ['head injury', 'hit head', 'fell', 'concussion'],
            'abdominal pain': ['stomach pain', 'abdominal pain', 'belly ache'],
            'back pain': ['back pain', 'spine', 'back injury']
        }
        
        for symptom, keywords in symptom_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                symptoms.append(symptom)
        
        return ", ".join(symptoms) if symptoms else "general symptoms"
    
    def _generate_action(self, is_emergency: bool, symptoms: str, emergency_type: str) -> str:
        """Generate appropriate action based on emergency status"""
        
        if is_emergency:
            if emergency_type in ['chest_pain', 'stroke']:
                return "Seek emergency medical care immediately"
            elif emergency_type in ['severe_injury', 'poisoning']:
                return "Go to nearest hospital emergency department"
            else:
                return "Seek immediate medical attention"
        else:
            if symptoms:
                return f"Monitor {symptoms} and see doctor if symptoms worsen"
            else:
                return "Schedule appointment with healthcare provider"
    
    def _calculate_confidence(self, ai_response: str, natural_response: str, is_emergency: bool) -> float:
        """Calculate confidence score for the assessment"""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence if AI response is available and clear
        if ai_response and ai_response != "Model not available for text generation.":
            confidence += 0.2
            
            # Higher confidence for clear emergency indicators
            if is_emergency and any(word in ai_response.lower() for word in ['emergency', 'immediate', 'urgent']):
                confidence += 0.2
            elif not is_emergency and any(word in ai_response.lower() for word in ['no', 'not emergency']):
                confidence += 0.2
        
        # Increase confidence if natural response is clear
        if natural_response and len(natural_response) > 10:
            confidence += 0.1
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _format_agent_response(self, agent_response: Dict[str, Any], query: str) -> str:
        """Format final agent response in 2-3 sentences max"""
        
        is_emergency = agent_response.get("is_emergency", False)
        action = agent_response.get("action", "Seek medical advice")
        symptoms = agent_response.get("symptoms", "")
        
        # Choose appropriate template
        if is_emergency:
            template = self.response_templates["emergency"]
        else:
            template = self.response_templates["non_emergency"]
        
        # Format response
        if symptoms and symptoms != "general symptoms":
            symptom_info = f"Symptoms: {symptoms}."
        else:
            symptom_info = ""
        
        response = template.format(
            action=action,
            symptom_info=symptom_info
        )
        
        # Ensure response is concise (2-3 sentences max)
        sentences = response.split('. ')
        if len(sentences) > 3:
            response = '. '.join(sentences[:3]) + '.'
        
        return response.strip()
    
    def _get_used_capabilities(self, agent_response: Dict[str, Any]) -> List[str]:
        """Determine which agent capabilities were used"""
        capabilities = ["emergency_detection"]
        
        if agent_response.get("symptoms"):
            capabilities.append("symptom_analysis")
        
        if agent_response.get("is_emergency"):
            capabilities.append("medical_advice")
        
        capabilities.append("global_health_guidance")
        
        return capabilities
    
    def _fallback_response(self, query: str) -> Dict[str, Any]:
        """Fallback response when RAG system is not available"""
        return {
            "emergency_type": "unknown",
            "ai_response": "System unavailable",
            "natural_response": "Please seek medical advice from a healthcare professional."
        }
    
    def _error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Error response when processing fails"""
        return {
            "agent_id": self.agent_id,
            "query": query,
            "response": "Emergency: Uncertain. Action: Seek medical advice immediately. System error occurred.",
            "emergency_detected": True,  # Default to emergency for safety
            "confidence": 0.1,
            "processing_time": 0.0,
            "timestamp": time.time(),
            "error": error,
            "metadata": {
                "error": True,
                "agent_version": self.version
            }
        }


def run_health_agent_pipeline(query: str, rag_system=None) -> Dict[str, Any]:
    """
    Main function to run the health agent pipeline
    
    Args:
        query: Health-related query from user
        rag_system: Optional RAG system instance
        
    Returns:
        Agent response with health advice
    """
    try:
        # Initialize agent
        agent = HealthAgent(rag_system)
        
        # Process query
        result = agent.process_health_query(query)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Health Agent Pipeline error: {e}")
        return {
            "agent_id": "health-emergency-agent",
            "query": query,
            "response": "Emergency: Uncertain. Action: Seek medical advice immediately. System error occurred.",
            "emergency_detected": True,
            "confidence": 0.1,
            "processing_time": 0.0,
            "timestamp": time.time(),
            "error": str(e)
        }


def test_health_agent():
    """Test the health agent with sample queries"""
    
    print("🏥 Testing Health Agent Production System")
    print("=" * 60)
    
    # Initialize RAG system
    try:
        from windows_rag_system import WindowsRAGSystem
        rag_system = WindowsRAGSystem()
        print("✅ RAG System initialized")
    except Exception as e:
        print(f"⚠️ RAG System not available: {e}")
        rag_system = None
    
    # Test scenarios
    test_queries = [
        "I have severe chest pain and shortness of breath",
        "I fell down the stairs and hit my head",
        "I have a high fever with body aches",
        "I feel dizzy and nauseous",
        "I have a minor headache"
    ]
    
    print(f"\n🔍 Testing {len(test_queries)} scenarios:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {query} ---")
        
        result = run_health_agent_pipeline(query, rag_system)
        
        print(f"Agent Response: {result['response']}")
        print(f"Emergency Detected: {result['emergency_detected']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Processing Time: {result['processing_time']:.3f}s")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
    
    print(f"\n✅ Health Agent testing completed")


if __name__ == "__main__":
    test_health_agent()

