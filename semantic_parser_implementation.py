import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import json
import re
from collections import defaultdict, deque

class SemanticParser:
    def __init__(self, max_memory_size=10):
        """
        Initialize the Semantic Parser with required NLP components
        
        Args:
            max_memory_size: Maximum number of past interactions to remember
        """
        # Load spaCy model for dependency parsing and NER
        self.nlp = spacy.load("en_core_web_lg")
        
        # Initialize intent detection pipeline
        self.intent_classifier = pipeline("zero-shot-classification", 
                                          model="facebook/bart-large-mnli")
        
        # Initialize memory component
        self.memory = deque(maxlen=max_memory_size)
        
        # Command templates for different intents
        self.command_templates = {
            "retrieve_info": "retrieve({entity}, {attribute})",
            "create": "create({entity}, {attributes})",
            "update": "update({entity}, {id}, {attributes})",
            "delete": "delete({entity}, {id})",
            "schedule": "schedule({event_type}, {time}, {attributes})",
            "check_condition": "check_if({condition}, then={action_true}, else={action_false})",
            "navigate": "navigate_to({destination})",
            "search": "search({query}, {filter})",
            "open": "open({entity})",
            "close": "close({entity})"
        }
        
        # Default intent labels for zero-shot classification
        self.intent_labels = [
            "retrieve_info", "create", "update", "delete", 
            "schedule", "check_condition", "navigate", "search",
            "open", "close", "other"
        ]
        
        # TF-IDF vectorizer for memory retrieval
        self.tfidf = TfidfVectorizer()
        self.memory_vectors = None
        
        # Known entity types that we'll be looking for
        self.entity_types = {
            "email": ["send", "email", "message", "mail", "inbox"],
            "document": ["document", "file", "report", "paper", "spreadsheet"],
            "app": ["application", "app", "program", "software", "browser", "window"],
            "meeting": ["meeting", "appointment", "event", "conference", "call", "session"],
            "person": ["team", "department", "person", "people", "colleague", "boss", "manager", "director", "staff"]
        }
        
    def add_to_memory(self, text, parsed_command=None):
        """Add interaction to memory"""
        memory_item = {
            "text": text,
            "parsed_command": parsed_command,
            "timestamp": len(self.memory)  # Simple timestamp for ordering
        }
        self.memory.append(memory_item)
        
        # Update memory vectors
        if len(self.memory) > 0:
            memory_texts = [item["text"] for item in self.memory]
            self.memory_vectors = self.tfidf.fit_transform(memory_texts)
    
    def retrieve_from_memory(self, query, top_k=3):
        """Retrieve relevant past interactions based on query"""
        if not self.memory or len(self.memory) == 0:
            return []
        
        # Convert query to vector
        query_vector = self.tfidf.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.memory_vectors)[0]
        
        # Get top-k most similar items
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.memory[i] for i in top_indices]
    
    def detect_intent(self, text):
        """Detect the intent of the text using zero-shot classification"""
        results = self.intent_classifier(text, candidate_labels=self.intent_labels)
        
        # Better intent detection with keyword matching
        text_lower = text.lower()
        
        # Direct mapping of keywords to intents
        intent_keywords = {
            "retrieve_info": ["find", "get", "show", "display", "list", "search for"],
            "open": ["open", "launch", "start", "run", "access"],
            "close": ["close", "exit", "quit", "shutdown", "end"],
            "schedule": ["schedule", "book", "plan", "arrange", "set up"],
            "check_condition": ["check if", "verify if", "see if", "determine if", "if"],
            "create": ["create", "make", "generate", "compose", "write"],
            "update": ["update", "modify", "change", "edit", "revise", "attach"],
            "delete": ["delete", "remove", "erase", "clear", "eliminate"]
        }
        
        # Check for keyword matches
        for intent, keywords in intent_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                # If there's a direct keyword match, prioritize this over zero-shot classification
                return intent, 0.9
        
        return results["labels"][0], results["scores"][0]
    
    def extract_entities(self, doc, intent):
        """Extract named entities and their relationships"""
        entities = {}
        
        # First pass: use spaCy's built-in NER
        for ent in doc.ents:
            entities[ent.text] = {
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
            
        # Second pass: use noun chunks to identify potential entities
        # This helps with non-NER entities like "email application"
        for chunk in doc.noun_chunks:
            # Look for specific entity types based on the chunk text
            chunk_text = chunk.text.lower()
            
            # Try to classify the entity type based on our known types
            for entity_type, keywords in self.entity_types.items():
                if any(keyword in chunk_text for keyword in keywords):
                    entities[chunk.text] = {
                        "label": entity_type,
                        "start": chunk.start_char,
                        "end": chunk.end_char
                    }
                    break
        
        # Third pass: intent-specific entity extraction
        text = doc.text.lower()
        
        if intent == "open" or intent == "close":
            # For open/close commands, extract what's being opened/closed
            verb_pos = text.find("open") if intent == "open" else text.find("close")
            if verb_pos >= 0:
                # Look for the entity after the verb
                rest_of_text = text[verb_pos + 4:].strip()
                
                # Remove common words like "the", "a", etc.
                rest_of_text = re.sub(r'^(the|a|an)\s+', '', rest_of_text)
                
                # The rest of the text is likely the entity to be opened/closed
                if rest_of_text:
                    entities[rest_of_text] = {
                        "label": "app",
                        "start": verb_pos + 4,
                        "end": len(text)
                    }
        
        elif intent == "retrieve_info":
            # For retrieval commands, extract what info is being requested
            for pattern in ["find", "get", "show", "display", "search", "find all", "get all"]:
                pattern_pos = text.find(pattern)
                if pattern_pos >= 0:
                    # Look for entities after the pattern
                    rest_of_text = text[pattern_pos + len(pattern):].strip()
                    
                    # Extract the main entity being searched/retrieved
                    for entity_type, keywords in self.entity_types.items():
                        for keyword in keywords:
                            if keyword in rest_of_text:
                                # Extract the entity around the keyword
                                start_pos = rest_of_text.find(keyword)
                                entity_text = rest_of_text[max(0, start_pos-5):min(len(rest_of_text), start_pos+len(keyword)+15)]
                                
                                # Clean up the entity text
                                entity_text = re.sub(r'^(the|a|an)\s+', '', entity_text.strip())
                                entity_text = re.sub(r'\s+(from|with|about|for)\s+.*$', '', entity_text)
                                
                                if entity_text:
                                    entities[entity_text] = {
                                        "label": entity_type,
                                        "start": pattern_pos + len(pattern),
                                        "end": pattern_pos + len(pattern) + len(entity_text)
                                    }
                                
                                # Now look for additional info after the entity
                                # For example "from John" in "find all emails from John"
                                if "from" in rest_of_text:
                                    from_pos = rest_of_text.find("from")
                                    attribute_text = rest_of_text[from_pos:].strip()
                                    attribute_text = re.sub(r'^from\s+', '', attribute_text)
                                    if attribute_text:
                                        entities[f"from {attribute_text}"] = {
                                            "label": "attribute",
                                            "start": pattern_pos + len(pattern) + from_pos,
                                            "end": pattern_pos + len(pattern) + from_pos + len(attribute_text) + 5
                                        }
        
        elif intent == "schedule":
            # For scheduling commands, extract event type and time
            # Extract event type
            for pattern in ["schedule", "book", "arrange", "set up"]:
                pattern_pos = text.find(pattern)
                if pattern_pos >= 0:
                    # Look for what's being scheduled
                    rest_of_text = text[pattern_pos + len(pattern):].strip()
                    
                    # Remove common words like "the", "a", etc.
                    rest_of_text = re.sub(r'^(the|a|an)\s+', '', rest_of_text)
                    
                    # Extract the event type
                    event_match = re.search(r'^([a-z\s]+?)(?=\s+(?:for|at|on|tomorrow|next|in))', rest_of_text)
                    if event_match:
                        event_type = event_match.group(1).strip()
                        entities[event_type] = {
                            "label": "event_type",
                            "start": pattern_pos + len(pattern),
                            "end": pattern_pos + len(pattern) + len(event_type)
                        }
                    
                    # Extract time
                    time_patterns = [
                        r'(tomorrow\s+at\s+\d+(?::\d+)?\s*(?:am|pm|AM|PM)?)',
                        r'(at\s+\d+(?::\d+)?\s*(?:am|pm|AM|PM)?)',
                        r'(on\s+\w+\s+\d+(?:st|nd|rd|th)?)',
                        r'(next\s+\w+)',
                        r'(in\s+\d+\s+\w+)'
                    ]
                    
                    for time_pattern in time_patterns:
                        time_match = re.search(time_pattern, rest_of_text)
                        if time_match:
                            time_text = time_match.group(1).strip()
                            entities[time_text] = {
                                "label": "time",
                                "start": pattern_pos + len(pattern) + rest_of_text.find(time_text),
                                "end": pattern_pos + len(pattern) + rest_of_text.find(time_text) + len(time_text)
                            }
                            break
        
        return entities
    
    def extract_attributes(self, doc, intent, entities):
        """Extract attributes based on dependency parsing and entities"""
        attributes = defaultdict(dict)
        
        # Basic extraction based on dependency parsing
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                attributes["subject"] = token.text
            elif token.dep_ in ("dobj", "pobj"):
                if token.head.pos_ == "VERB":
                    action = token.head.text
                    attributes["objects"][token.text] = {"action": action}
            
            # Extract time-related information
            if token.ent_type_ in ("DATE", "TIME"):
                attributes["time"] = token.text
                
            # Extract conditions for check_condition intent
            if token.text in ("if", "when", "unless"):
                condition_span = None
                for child in token.children:
                    if child.dep_ == "advcl":
                        condition_span = doc[child.left_edge.i:child.right_edge.i+1].text
                if condition_span:
                    attributes["condition"] = condition_span
        
        # Intent-specific attribute extraction
        if intent == "retrieve_info":
            # Find the entity being searched
            entity = None
            attribute = None
            
            # Extract from entities
            for e, props in entities.items():
                if props.get("label") in ["email", "document", "app", "meeting", "person"]:
                    entity = e
                elif props.get("label") == "attribute":
                    attribute = e
            
            if entity:
                attributes["entity"] = entity
            if attribute:
                attributes["attribute"] = attribute
                
            # If we didn't find an attribute yet, look for common patterns
            if not attribute:
                text = doc.text.lower()
                
                # Look for patterns like "from John", "about budget", etc.
                prep_patterns = [
                    (r'from\s+(\w+(?:\s+\w+)*)', "from"),
                    (r'about\s+(\w+(?:\s+\w+)*)', "about"),
                    (r'with\s+(\w+(?:\s+\w+)*)', "with"),
                    (r'related\s+to\s+(\w+(?:\s+\w+)*)', "related to")
                ]
                
                for pattern, prep in prep_patterns:
                    match = re.search(pattern, text)
                    if match:
                        attributes["attribute"] = f"{prep} {match.group(1)}"
                        break
        
        elif intent == "open" or intent == "close":
            # For open/close, the entity is what's being opened/closed
            for e, props in entities.items():
                attributes["entity"] = e
                break
        
        elif intent == "schedule":
            # For scheduling, extract the event type and time
            for e, props in entities.items():
                if props.get("label") == "event_type":
                    attributes["event_type"] = e
                elif props.get("label") == "time":
                    attributes["time"] = e
        
        elif intent == "check_condition":
            # For check_condition, extract condition, then, and else
            text = doc.text.lower()
            
            # Extract condition
            if "if" in text:
                if_pos = text.find("if")
                condition_text = text[if_pos + 2:].strip()
                
                # Look for comma or "then" or "else" to end the condition
                end_markers = [",", " then ", " else "]
                end_pos = len(condition_text)
                
                for marker in end_markers:
                    marker_pos = condition_text.find(marker)
                    if marker_pos >= 0:
                        end_pos = min(end_pos, marker_pos)
                
                condition = condition_text[:end_pos].strip()
                attributes["condition"] = condition
                
                # Look for then clause
                if "then" in condition_text:
                    then_pos = condition_text.find("then")
                    then_text = condition_text[then_pos + 4:].strip()
                    
                    # Check if there's an else clause
                    if "else" in then_text:
                        else_pos = then_text.find("else")
                        then_action = then_text[:else_pos].strip()
                        else_action = then_text[else_pos + 4:].strip()
                        
                        attributes["then"] = then_action
                        attributes["else"] = else_action
                    else:
                        attributes["then"] = then_text
                
                # If no explicit then, but there's a comma, assume what follows is the then
                elif "," in condition_text:
                    comma_pos = condition_text.find(",")
                    then_text = condition_text[comma_pos + 1:].strip()
                    attributes["then"] = then_text
                    
        return dict(attributes)
    
    def build_command(self, intent, entities, attributes, doc):
        """Build a structured command based on intent, entities, and attributes"""
        if intent not in self.command_templates:
            return {"type": "unknown", "raw_text": doc.text}
        
        command = {"type": intent, "raw_text": doc.text}
        
        # Fill in the command based on intent type
        if intent == "retrieve_info":
            entity = attributes.get("entity", "")
            attribute = attributes.get("attribute", "")
            
            # If we didn't find these in attributes, try to extract from entities
            if not entity and entities:
                entity_keys = [e for e, props in entities.items() 
                              if props.get("label") in ["email", "document", "app", "meeting", "person"]]
                if entity_keys:
                    entity = entity_keys[0]
            
            # Default entity if still missing
            if not entity:
                if "email" in doc.text.lower():
                    entity = "emails"
                elif "document" in doc.text.lower() or "file" in doc.text.lower():
                    entity = "documents"
                else:
                    entity = "information"
            
            command["entity"] = entity
            command["attribute"] = attribute
            
        elif intent == "schedule":
            event_type = attributes.get("event_type", "")
            time = attributes.get("time", "")
            
            # If we didn't find these in attributes, try to extract from entities
            if not event_type:
                for e, props in entities.items():
                    if props.get("label") == "event_type":
                        event_type = e
                        break
            
            if not time:
                for e, props in entities.items():
                    if props.get("label") == "time":
                        time = e
                        break
            
            # Default if still missing
            if not event_type:
                if "meeting" in doc.text.lower():
                    event_type = "team meeting"
                elif "appointment" in doc.text.lower():
                    event_type = "appointment"
                else:
                    event_type = "meeting"
            
            if not time and "tomorrow" in doc.text.lower():
                time = "tomorrow"
            
            command["event_type"] = event_type
            command["time"] = time
            
        elif intent == "check_condition":
            condition = attributes.get("condition", "")
            then_action = attributes.get("then", "")
            else_action = attributes.get("else", "")
            
            # If we didn't find these in attributes, try to extract from text
            text = doc.text.lower()
            
            if not condition and "if" in text:
                if_pos = text.find("if")
                end_pos = text.find(",", if_pos) if "," in text[if_pos:] else len(text)
                condition = text[if_pos + 2:end_pos].strip()
            
            command["condition"] = condition
            command["then"] = then_action
            if else_action:
                command["else"] = else_action
            
        elif intent == "open" or intent == "close":
            entity = attributes.get("entity", "")
            
            # If we didn't find it in attributes, try to extract from entities
            if not entity and entities:
                entity_keys = list(entities.keys())
                if entity_keys:
                    entity = entity_keys[0]
            
            command["entity"] = entity
            
        else:
            # Generic command building for other intents
            for key, value in attributes.items():
                if key not in ["raw_text"] and value:
                    command[key] = value
            
            # If we have important entities not in attributes, add them
            for entity, props in entities.items():
                if entity not in attributes.values():
                    if props.get("label") in ["email", "document", "app", "meeting", "person"]:
                        command["entity"] = entity
                        break
        
        return command
    
    def parse(self, text):
        """
        Main parsing function to convert text to structured command
        
        Args:
            text: Text to parse
            
        Returns:
            Structured command object
        """
        # Process with spaCy
        doc = self.nlp(text)
        
        # Check memory for similar past interactions
        relevant_memories = self.retrieve_from_memory(text)
        
        # Detect intent
        intent, confidence = self.detect_intent(text)
        
        # Extract entities and attributes
        entities = self.extract_entities(doc, intent)
        attributes = self.extract_attributes(doc, intent, entities)
        
        # Enhance parsing with memory if available
        if relevant_memories:
            # Use past successful parses to improve current parsing
            for mem in relevant_memories:
                if mem["parsed_command"] and mem["parsed_command"]["type"] == intent:
                    # Inherit missing attributes from memory if current parsing is incomplete
                    for key, value in mem["parsed_command"].items():
                        if key not in ["type", "raw_text"] and key not in attributes:
                            attributes[key] = value
        
        # Build the structured command
        command = self.build_command(intent, entities, attributes, doc)
        
        # Split complex sentences into multiple commands if needed
        commands = [command]
        
        # Look for coordinating conjunctions indicating multiple commands
        conjunctions = ["and", "then", "after", "before", "while"]
        
        # Check if sentence has multiple verbs with a conjunction
        if any(conj in text.lower() for conj in conjunctions):
            
            # Look for patterns like "do X and do Y"
            for conj in conjunctions:
                if conj in text.lower():
                    parts = text.lower().split(conj)
                    
                    # Only split if both parts have verbs
                    if len(parts) == 2:
                        doc1 = self.nlp(parts[0])
                        doc2 = self.nlp(parts[1])
                        
                        has_verb1 = any(token.pos_ == "VERB" for token in doc1)
                        has_verb2 = any(token.pos_ == "VERB" for token in doc2)
                        
                        if has_verb1 and has_verb2:
                            # Parse each part as a separate command
                            sub_intent2, _ = self.detect_intent(parts[1])
                            sub_entities2 = self.extract_entities(doc2, sub_intent2)
                            sub_attributes2 = self.extract_attributes(doc2, sub_intent2, sub_entities2)
                            sub_command2 = self.build_command(sub_intent2, sub_entities2, sub_attributes2, doc2)
                            
                            # Add second command to the list if it's different
                            if sub_command2["type"] != "unknown" and sub_command2["type"] != command["type"]:
                                commands.append(sub_command2)
                                break
        
        # Add to memory
        self.add_to_memory(text, command)
        
        return commands if len(commands) > 1 else commands[0]
    
    def evaluate(self, test_data):
        """
        Evaluate the parser on test data
        
        Args:
            test_data: List of (text, expected_command) tuples
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            "accuracy": 0,
            "intent_accuracy": 0,
            "entity_recall": 0,
            "entity_precision": 0
        }
        
        correct = 0
        correct_intents = 0
        total_expected_entities = 0
        total_predicted_entities = 0
        total_correct_entities = 0
        
        for text, expected in test_data:
            parsed = self.parse(text)
            
            # Check intent accuracy
            if parsed["type"] == expected["type"]:
                correct_intents += 1
            
            # Check entity extraction
            expected_entities = set()
            predicted_entities = set()
            
            for key, value in expected.items():
                if key not in ["type", "raw_text"] and value:
                    if isinstance(value, dict):
                        for subkey in value:
                            expected_entities.add(f"{key}:{subkey}")
                    else:
                        expected_entities.add(key)
            
            for key, value in parsed.items():
                if key not in ["type", "raw_text"] and value:
                    if isinstance(value, dict):
                        for subkey in value:
                            predicted_entities.add(f"{key}:{subkey}")
                    else:
                        predicted_entities.add(key)
            
            correct_entities = expected_entities.intersection(predicted_entities)
            
            total_expected_entities += len(expected_entities)
            total_predicted_entities += len(predicted_entities)
            total_correct_entities += len(correct_entities)
            
            # Check overall command correctness
            if parsed["type"] == expected["type"] and len(correct_entities) == len(expected_entities):
                correct += 1
        
        # Calculate metrics
        n = len(test_data)
        results["accuracy"] = correct / n
        results["intent_accuracy"] = correct_intents / n
        results["entity_recall"] = total_correct_entities / total_expected_entities if total_expected_entities > 0 else 0
        results["entity_precision"] = total_correct_entities / total_predicted_entities if total_predicted_entities > 0 else 0
        results["f1_score"] = 2 * (results["entity_precision"] * results["entity_recall"]) / (results["entity_precision"] + results["entity_recall"]) if (results["entity_precision"] + results["entity_recall"]) > 0 else 0
        
        return results


# Usage examples
if __name__ == "__main__":
    parser = SemanticParser()
    
    # Example 1: Simple command
    text1 = "Open the email application"
    command1 = parser.parse(text1)
    print(f"Example 1: {text1}")
    print(f"Parsed command: {json.dumps(command1, indent=2)}")
    print()
    
    # Example 2: Command with condition
    text2 = "Check the weather for tomorrow at 3 PM, and if it's sunny, schedule a meeting at 2 PM"
    command2 = parser.parse(text2)
    print(f"Example 2: {text2}")
    print(f"Parsed command: {json.dumps(command2, indent=2)}")
    print()
    
    # Example 3: Command with context from memory
    text3 = "Send it to John"  # This relies on context from memory
    command3 = parser.parse(text3)
    print(f"Example 3: {text3}")
    print(f"Parsed command: {json.dumps(command3, indent=2)}")
    print()
    
    # Evaluation
    test_data = [
        ("Open the file manager", {"type": "open", "entity": "file manager"}),
        ("Schedule a meeting for tomorrow at 10 AM", {"type": "schedule", "event_type": "meeting", "time": "tomorrow at 10 AM"}),
        ("Check if there are any emails from Sarah", {"type": "retrieve_info", "entity": "emails", "attribute": "from Sarah"})
    ]
    
    eval_results = parser.evaluate(test_data)
    print("Evaluation Results:")
    print(json.dumps(eval_results, indent=2))