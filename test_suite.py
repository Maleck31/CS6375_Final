import unittest
import json
import sys
import os
sys.path.append('.')  # Ensure modules can be found

from semantic_parser_implementation import SemanticParser
from evaluation_framework import SemanticParserEvaluator

class TestSemanticParser(unittest.TestCase):
    """Test cases for the Semantic Parser implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.parser = SemanticParser()
        self.evaluator = SemanticParserEvaluator(self.parser)
    
    def test_basic_commands(self):
        """Test basic command parsing"""
        # Test open command
        result = self.parser.parse("Open the email application")
        self.assertEqual(result["type"], "open")
        self.assertIn("email", result["entity"].lower())
        
        # Test close command
        result = self.parser.parse("Close the browser")
        self.assertEqual(result["type"], "close")
        self.assertIn("browser", result["entity"].lower())
        
        # Test retrieve info command
        result = self.parser.parse("Find all emails from Sarah")
        self.assertEqual(result["type"], "retrieve_info")
        self.assertIn("email", result["entity"].lower())
    
    def test_scheduling_commands(self):
        """Test scheduling command parsing"""
        result = self.parser.parse("Schedule a meeting for tomorrow at 10 AM")
        self.assertEqual(result["type"], "schedule")
        self.assertIn("meeting", result["event_type"].lower())
        self.assertIn("tomorrow", result["time"].lower())
        
        result = self.parser.parse("Book a doctor's appointment for next Friday")
        self.assertEqual(result["type"], "schedule")
        self.assertIn("doctor", result["event_type"].lower())
        self.assertIn("friday", result["time"].lower())
    
    def test_conditional_commands(self):
        """Test conditional command parsing"""
        result = self.parser.parse("If there are no meetings this afternoon, schedule a call with marketing")
        self.assertEqual(result["type"], "check_condition")
        self.assertIn("meetings", result["condition"].lower())
        self.assertIn("schedule", result.get("then", "").lower())
        
        result = self.parser.parse("Check the weather for tomorrow, and if it's sunny, schedule a picnic")
        self.assertEqual(result["type"], "check_condition")
        self.assertIn("weather", result["condition"].lower())
        self.assertIn("schedule", result.get("then", "").lower())
    
    def test_memory_feature(self):
        """Test memory component"""
        # First command
        self.parser.parse("Find the Q1 budget report")
        
        # Reference previous command
        result = self.parser.parse("Share it with the finance team")
        
        # Should have maintained context from previous command
        self.assertNotEqual(result["type"], "unknown")
    
    def test_intent_detection(self):
        """Test intent detection"""
        intent, _ = self.parser.detect_intent("Open the file manager")
        self.assertEqual(intent, "open")
        
        intent, _ = self.parser.detect_intent("What's the weather like today?")
        self.assertEqual(intent, "retrieve_info")
        
        intent, _ = self.parser.detect_intent("Schedule a meeting for tomorrow")
        self.assertEqual(intent, "schedule")
    
    def test_entity_extraction(self):
        """Test entity extraction"""
        doc = self.parser.nlp("Send an email to John about the project")
        entities = self.parser.extract_entities(doc)
        
        # Should have extracted "John" and possibly "project"
        entity_texts = list(entities.keys())
        self.assertTrue(any("John" in text for text in entity_texts))
    
    def test_attribute_extraction(self):
        """Test attribute extraction"""
        doc = self.parser.nlp("Schedule a team meeting for tomorrow at 3 PM")
        intent, _ = self.parser.detect_intent("Schedule a team meeting for tomorrow at 3 PM")
        attributes = self.parser.extract_attributes(doc, intent)
        
        # Should have extracted time
        self.assertIn("time", attributes) 
    
    def test_complex_parsing(self):
        """Test parsing of complex commands"""
        text = "Check the weather for tomorrow at 3 PM, and if it's sunny, schedule a meeting at 2 PM"
        result = self.parser.parse(text)
        
        # Should be a conditional command
        self.assertEqual(result["type"], "check_condition")
        self.assertIn("weather", result["condition"].lower())
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation"""
        # Simple test data
        test_data = [
            ("Open the file manager", {"type": "open", "entity": "file manager"})
        ]
        
        # Run evaluation
        results = self.evaluator.evaluate(test_data)
        
        # Basic checks on metrics
        self.assertIn("average", results)
        self.assertIn("bleu", results["average"])
        self.assertIn("rouge-1", results["average"])
        self.assertIn("intent_accuracy", results["average"])
        
        # Intent accuracy should be at least reasonable
        self.assertGreaterEqual(results["average"]["intent_accuracy"], 0.5)

class TestEndToEnd(unittest.TestCase):
    """End-to-end tests of the Semantic Parser"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.parser = SemanticParser()
    
    def test_end_to_end_workflow(self):
        """Test an end-to-end workflow with multiple commands"""
        # Initial command
        cmd1 = self.parser.parse("Check my calendar for tomorrow")
        self.assertEqual(cmd1["type"], "retrieve_info")
        
        # Command with context from previous
        cmd2 = self.parser.parse("If there are no meetings in the afternoon, schedule a team call at 3 PM")
        self.assertEqual(cmd2["type"], "check_condition")
        
        # Another command in sequence
        cmd3 = self.parser.parse("Send an email to the team about the call")
        # Should maintain context about the team call
        self.assertNotEqual(cmd3["type"], "unknown")


if __name__ == "__main__":
    unittest.main()
