import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from semantic_parser_implementation import SemanticParser

class SemanticParserEvaluator:
    def __init__(self, parser=None):
        """
        Initialize the evaluator
        
        Args:
            parser: Instance of SemanticParser. If None, a new one will be created.
        """
        self.parser = parser if parser else SemanticParser()
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        
    def command_to_string(self, command):
        """Convert a command object to a string representation for text-based metrics"""
        if isinstance(command, list):
            return "; ".join([self.command_to_string(c) for c in command])
            
        cmd_type = command.get("type", "unknown")
        
        if cmd_type == "retrieve_info":
            entity = command.get("entity", "")
            attribute = command.get("attribute", "")
            return f"retrieve {entity} {attribute}".strip()
            
        elif cmd_type == "schedule":
            event = command.get("event_type", "event")
            time = command.get("time", "")
            return f"schedule {event} at {time}".strip()
            
        elif cmd_type == "check_condition":
            condition = command.get("condition", "")
            then_action = command.get("then", "")
            else_action = command.get("else", "")
            result = f"check if {condition}"
            if then_action:
                result += f" then {then_action}"
            if else_action:
                result += f" else {else_action}"
            return result
            
        elif cmd_type == "open":
            entity = command.get("entity", "")
            return f"open {entity}".strip()
            
        elif cmd_type == "close":
            entity = command.get("entity", "")
            return f"close {entity}".strip()
            
        else:
            # Generic string representation
            parts = []
            for key, value in command.items():
                if key not in ["type", "raw_text"]:
                    if isinstance(value, dict):
                        value_str = ", ".join([f"{k}={v}" for k, v in value.items()])
                        parts.append(f"{key}=({value_str})")
                    else:
                        parts.append(f"{key}={value}")
            return f"{cmd_type} {' '.join(parts)}".strip()
    
    def calculate_bleu(self, reference, candidate):
        """Calculate BLEU score between reference and candidate strings"""
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=self.smooth)
    
    def calculate_rouge(self, reference, candidate):
        """Calculate ROUGE scores between reference and candidate strings"""
        try:
            scores = self.rouge.get_scores(candidate, reference)[0]
            return {
                "rouge-1": scores["rouge-1"]["f"],
                "rouge-2": scores["rouge-2"]["f"],
                "rouge-l": scores["rouge-l"]["f"]
            }
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            return {
                "rouge-1": 0,
                "rouge-2": 0,
                "rouge-l": 0
            }
    
    def calculate_intent_accuracy(self, true_commands, pred_commands):
        """Calculate accuracy of intent classification"""
        # Check if both are lists or both are dictionaries
        if isinstance(true_commands, list) and isinstance(pred_commands, list):
            # For multi-command scenarios
            true_intents = [cmd.get("type", "unknown") for cmd in true_commands]
            pred_intents = [cmd.get("type", "unknown") for cmd in pred_commands]
            
            # Make sure lists are same length
            max_len = max(len(true_intents), len(pred_intents))
            true_intents = true_intents + ["unknown"] * (max_len - len(true_intents))
            pred_intents = pred_intents + ["unknown"] * (max_len - len(pred_intents))
            
            return accuracy_score(true_intents, pred_intents)
        elif isinstance(true_commands, dict) and isinstance(pred_commands, dict):
            # Single command scenario
            return 1.0 if true_commands.get("type") == pred_commands.get("type") else 0.0
        else:
            # Handle mixed case (one is list, one is dict)
            print("Warning: Mismatched command types (list vs dict)")
            return 0.0  # Return zero accuracy for mismatched types
    
    def calculate_entity_metrics(self, true_commands, pred_commands):
        """Calculate precision, recall, and F1 for entity extraction"""
        def extract_entities(command):
            entities = []
            if isinstance(command, list):
                for cmd in command:
                    entities.extend(extract_entities(cmd))
                return entities
                
            for key, value in command.items():
                if key not in ["type", "raw_text"]:
                    if isinstance(value, dict):
                        for k, v in value.items():
                            entities.append(f"{key}:{k}={v}")
                    else:
                        entities.append(f"{key}={value}")
            return entities
        
        true_entities = set(extract_entities(true_commands))
        pred_entities = set(extract_entities(pred_commands))
        
        # Calculate precision, recall, F1
        tp = len(true_entities.intersection(pred_entities))
        fp = len(pred_entities - true_entities)
        fn = len(true_entities - pred_entities)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def evaluate(self, test_data):
        """
        Comprehensive evaluation of the parser
        
        Args:
            test_data: List of tuples (input_text, expected_command)
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            "bleu": [],
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "intent_accuracy": [],
        }
        
        for text, expected in test_data:
            # Parse the input
            parsed = self.parser.parse(text)
            
            # Convert commands to string for text-based metrics
            expected_str = self.command_to_string(expected)
            parsed_str = self.command_to_string(parsed)
            
            # Calculate BLEU
            bleu = self.calculate_bleu(expected_str, parsed_str)
            results["bleu"].append(bleu)
            
            # Calculate ROUGE
            rouge_scores = self.calculate_rouge(expected_str, parsed_str)
            results["rouge-1"].append(rouge_scores["rouge-1"])
            results["rouge-2"].append(rouge_scores["rouge-2"])
            results["rouge-l"].append(rouge_scores["rouge-l"])
            
            # Calculate intent accuracy
            intent_acc = self.calculate_intent_accuracy(expected, parsed)
            results["intent_accuracy"].append(intent_acc)
        
        # Calculate averages
        avg_results = {k: np.mean(v) for k, v in results.items()}
        
        return {
            "detailed": results,
            "average": avg_results
        }
    
    def visualize_results(self, results):
        """
        Visualize evaluation results
        
        Args:
            results: Dictionary with evaluation results from evaluate()
        """
        avg_results = results["average"]
        
        # Create bar chart
        metrics = list(avg_results.keys())
        values = list(avg_results.values())
        
        plt.figure(figsize=(12, 6))
        plt.bar(metrics, values)
        plt.title('Semantic Parser Evaluation Results')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.savefig('evaluation_results.png')
        plt.close()
        
        return 'evaluation_results.png'

# Example usage
if __name__ == "__main__":
    # Create parser
    parser = SemanticParser()
    
    # Create evaluator
    evaluator = SemanticParserEvaluator(parser)
    
    # Example test data
    test_data = [
        # ── simple single-command sentences ──────────────────────────
        ("Open the email application",
        {"type": "open", "entity": "email application"}),

        ("Close the browser window",
        {"type": "close", "entity": "browser window"}),

        ("Find all emails from John",
        {"type": "retrieve_info", "entity": "emails", "attribute": "from John"}),

        # ── scheduling (still one command) ───────────────────────────
        ("Schedule a team meeting tomorrow at 2 PM",
        {"type": "schedule", "event_type": "team meeting", "time": "tomorrow at 2 PM"}),

        # ── conditional statements (one command) ─────────────────────
        ("Check the weather for tomorrow, and if it's sunny, schedule a picnic",
        {"type": "check_condition",
        "condition": "weather for tomorrow",
        "then": "schedule a picnic"}),

        # ── really contains two separate actions → list of dicts ─────
        ("Send an email to the team about the project deadline and attach the latest report",
        [
            {"type": "create",
            "entity": "email",
            "attributes": {"recipient": "team",
                            "subject": "project deadline"}},
            {"type": "update",
            "entity": "email",
            "attributes": {"attachment": "latest report"}}
        ]),

        ("Find documents related to Q1 budget and share them with the finance department",
        [
            {"type": "retrieve_info",
            "entity": "documents",
            "attribute": "related to Q1 budget"},
            {"type": "create",
            "entity": "share",
            "attributes": {"object": "documents",
                            "recipient": "finance department"}}
        ])
    ]

    
    # Run evaluation
    eval_results = evaluator.evaluate(test_data)
    
    # Print results
    print("Evaluation Results:")
    print(json.dumps(eval_results["average"], indent=2))
    
    # Visualize
    visualization_path = evaluator.visualize_results(eval_results)
    print(f"Visualization saved to: {visualization_path}")