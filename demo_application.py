import os
import sys
import json
import argparse
import time
import requests
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.markdown import Markdown
from rich import print as rprint
import pyautogui
import nltk
from rich.prompt import Prompt
import threading

# Download necessary NLTK data
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

# Import our semantic parser
sys.path.append('.')  # Ensure modules can be found
from semantic_parser_implementation import SemanticParser

# Setup rich console
console = Console()

class LLMAutomationSystem:
    def __init__(self, llm_api_key=None, llm_endpoint=None):
        """
        Initialize the LLM Automation System
        
        Args:
            llm_api_key: API key for LLM service (optional)
            llm_endpoint: Endpoint URL for LLM service (optional)
        """
        self.parser = SemanticParser(max_memory_size=20)
        self.llm_api_key = llm_api_key or os.environ.get("LLM_API_KEY")
        self.llm_endpoint = llm_endpoint or os.environ.get("LLM_ENDPOINT")
        
        # Initialize command executor registry
        self.command_executors = {
            "open": self.execute_open,
            "close": self.execute_close,
            "retrieve_info": self.execute_retrieve_info,
            "create": self.execute_create,
            "update": self.execute_update,
            "delete": self.execute_delete,
            "schedule": self.execute_schedule,
            "check_condition": self.execute_check_condition,
            "navigate": self.execute_navigate,
            "search": self.execute_search
        }
        
        # Conversation history
        self.conversation_history = []
        
    def get_llm_response(self, user_input):
        """
        Get response from LLM API
        
        Args:
            user_input: User query or instruction
            
        Returns:
            LLM response text
        """
        # For demo purposes, we'll simulate LLM responses
        # In a real implementation, this would call an actual LLM API
        
        if self.llm_api_key and self.llm_endpoint:
            # Example of how to call an actual LLM API
            try:
                headers = {
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "messages": self.conversation_history + [{"role": "user", "content": user_input}],
                    "max_tokens": 500
                }
                
                response = requests.post(
                    self.llm_endpoint,
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    console.print(f"[red]Error from LLM API: {response.status_code}[/red]")
                    return "I'm sorry, I encountered an error processing your request."
                    
            except Exception as e:
                console.print(f"[red]Exception when calling LLM API: {str(e)}[/red]")
                return "I'm sorry, I encountered an error processing your request."
        
        # Simulated responses for demo purposes
        responses = {
            "hello": "Hello! How can I assist you today?",
            "help": "I can help you with various tasks. Try asking me to schedule meetings, find information, or automate UI actions.",
            "schedule": "I'll schedule that for you. What time would you prefer?",
            "find": "I'll look that up for you right away.",
            "open": "Opening that application for you now.",
            "weather": "The weather forecast shows sunny conditions with a high of 75°F.",
            "email": "I'll help you compose that email. What would you like to include in it?",
            "search": "I'll search for that information right away."
        }
        
        # Find the most relevant canned response
        for key, response in responses.items():
            if key.lower() in user_input.lower():
                return response
                
        # Default response if no match found
        if "?" in user_input:
            return "That's a good question. Based on the information I have, I would recommend checking your system settings."
        else:
            return "I'll help you with that. What specific action would you like me to take?"
    
    def parse_llm_response(self, llm_response):
        """
        Parse LLM response into structured commands
        
        Args:
            llm_response: Response text from LLM
            
        Returns:
            Structured command(s)
        """
        return self.parser.parse(llm_response)
    
    def execute_command(self, command):
        """
        Execute a parsed command
        
        Args:
            command: Structured command object or list of commands
            
        Returns:
            Execution result dictionary
        """
        if isinstance(command, list):
            results = []
            for cmd in command:
                results.append(self.execute_command(cmd))
            return results
        
        cmd_type = command.get("type", "unknown")
        executor = self.command_executors.get(cmd_type)
        
        if executor:
            try:
                return executor(command)
            except Exception as e:
                console.print(f"[red]Error executing command: {str(e)}[/red]")
                return {"status": "error", "message": str(e)}
        else:
            return {"status": "error", "message": f"Unknown command type: {cmd_type}"}
    
    def execute_open(self, command):
        """Execute open command"""
        entity = command.get("entity", "")
        console.print(f"[green]Opening {entity}...[/green]")
        
        # Simulate opening application
        # In a real implementation, this would use platform-specific APIs
        time.sleep(1)
        
        # Example of how to use pyautogui to automate UI interaction
        # pyautogui.hotkey('win')  # Windows key
        # pyautogui.typewrite(entity)
        # pyautogui.press('enter')
        
        return {"status": "success", "message": f"Opened {entity}"}
    
    def execute_close(self, command):
        """Execute close command"""
        entity = command.get("entity", "")
        console.print(f"[green]Closing {entity}...[/green]")
        
        # Simulate closing application
        # In a real implementation, this would use platform-specific APIs
        time.sleep(1)
        
        # Example of how to use pyautogui to automate UI interaction
        # pyautogui.hotkey('alt', 'f4')
        
        return {"status": "success", "message": f"Closed {entity}"}
    
    def execute_retrieve_info(self, command):
        """Execute retrieve_info command"""
        entity = command.get("entity", "")
        attribute = command.get("attribute", "")
        console.print(f"[green]Retrieving {attribute} from {entity}...[/green]")
        
        # Simulate retrieving information
        # In a real implementation, this would query databases or APIs
        time.sleep(1.5)
        
        # Demo data for different entities
        demo_data = {
            "emails": ["New project proposal", "Meeting reminder", "Weekly report"],
            "meetings": ["9:00 AM - Team Standup", "2:00 PM - Client Call", "4:30 PM - Project Review"],
            "files": ["report.docx", "presentation.pptx", "budget.xlsx"],
            "weather": "Partly cloudy, 72°F"
        }
        
        if entity in demo_data:
            return {"status": "success", "data": demo_data[entity], "message": f"Retrieved {attribute} from {entity}"}
        else:
            return {"status": "success", "data": f"Sample data for {entity}", "message": f"Retrieved {attribute} from {entity}"}
    
    def execute_create(self, command):
        """Execute create command"""
        entity = command.get("entity", "")
        attributes = command.get("attributes", {})
        console.print(f"[green]Creating {entity} with attributes: {attributes}...[/green]")
        
        # Simulate creating an entity
        # In a real implementation, this would call APIs or create files
        time.sleep(1)
        
        return {"status": "success", "message": f"Created {entity}", "id": "123456"}
    
    def execute_update(self, command):
        """Execute update command"""
        entity = command.get("entity", "")
        entity_id = command.get("id", "")
        attributes = command.get("attributes", {})
        console.print(f"[green]Updating {entity} {entity_id} with attributes: {attributes}...[/green]")
        
        # Simulate updating an entity
        # In a real implementation, this would call APIs or update files
        time.sleep(1)
        
        return {"status": "success", "message": f"Updated {entity} {entity_id}"}
    
    def execute_delete(self, command):
        """Execute delete command"""
        entity = command.get("entity", "")
        entity_id = command.get("id", "")
        console.print(f"[green]Deleting {entity} {entity_id}...[/green]")
        
        # Simulate deleting an entity
        # In a real implementation, this would call APIs or delete files
        time.sleep(1)
        
        return {"status": "success", "message": f"Deleted {entity} {entity_id}"}
    
    def execute_schedule(self, command):
        """Execute schedule command"""
        event_type = command.get("event_type", "meeting")
        time_str = command.get("time", "")
        attributes = command.get("attributes", {})
        console.print(f"[green]Scheduling {event_type} at {time_str} with attributes: {attributes}...[/green]")
        
        # Simulate scheduling an event
        # In a real implementation, this would call calendar APIs
        time.sleep(1.5)
        
        return {"status": "success", "message": f"Scheduled {event_type} at {time_str}", "id": "evt-123456"}
    
    def execute_check_condition(self, command):
        """Execute check_condition command"""
        condition = command.get("condition", "")
        then_action = command.get("then", "")
        else_action = command.get("else", "")
        
        console.print(f"[green]Checking condition: {condition}...[/green]")
        
        # Simulate checking a condition
        # In a real implementation, this would evaluate the condition
        time.sleep(1)
        
        # For demo purposes, randomly determine if condition is true or false
        import random
        condition_result = random.choice([True, False])
        
        if condition_result:
            console.print(f"[green]Condition is true, executing: {then_action}[/green]")
            # Parse and execute the then_action
            if then_action:
                then_cmd = self.parser.parse(then_action)
                return self.execute_command(then_cmd)
            return {"status": "success", "message": f"Condition {condition} is true"}
        else:
            console.print(f"[green]Condition is false, executing: {else_action}[/green]")
            # Parse and execute the else_action
            if else_action:
                else_cmd = self.parser.parse(else_action)
                return self.execute_command(else_cmd)
            return {"status": "success", "message": f"Condition {condition} is false"}
    
    def execute_navigate(self, command):
        """Execute navigate command"""
        destination = command.get("destination", "")
        console.print(f"[green]Navigating to {destination}...[/green]")
        
        # Simulate navigation
        # In a real implementation, this would navigate to a URL or location
        time.sleep(1)
        
        return {"status": "success", "message": f"Navigated to {destination}"}
    
    def execute_search(self, command):
        """Execute search command"""
        query = command.get("query", "")
        filter_str = command.get("filter", "")
        console.print(f"[green]Searching for '{query}' with filter: {filter_str}...[/green]")
        
        # Simulate search
        # In a real implementation, this would call search APIs
        time.sleep(1.5)
        
        # Demo search results
        results = [
            {"title": f"Result 1 for {query}", "url": "https://example.com/1"},
            {"title": f"Result 2 for {query}", "url": "https://example.com/2"},
            {"title": f"Result 3 for {query}", "url": "https://example.com/3"}
        ]
        
        return {"status": "success", "message": f"Search completed for '{query}'", "results": results}
    
    def process_user_input(self, user_input):
        """
        Process user input through the entire pipeline
        
        Args:
            user_input: User query or instruction
            
        Returns:
            Dictionary with process results
        """
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Step 1: Get response from LLM
        llm_response = self.get_llm_response(user_input)
        
        # Add LLM response to conversation history
        self.conversation_history.append({"role": "assistant", "content": llm_response})
        
        # Step 2: Parse LLM response into structured commands
        structured_command = self.parse_llm_response(llm_response)
        
        # Step 3: Execute the structured command
        execution_result = self.execute_command(structured_command)
        
        return {
            "user_input": user_input,
            "llm_response": llm_response,
            "structured_command": structured_command,
            "execution_result": execution_result
        }
    
    def run_cli(self):
        """Run the system in CLI mode"""
        console.print(Panel.fit(
            "[bold blue]Semantic Parser for LLM Responses[/bold blue]\n"
            "[green]A system that translates LLM responses into structured commands[/green]",
            title="UI Automation Demo"
        ))
        
        console.print("\nType 'exit' or 'quit' to end the session.\n")
        
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You")
                
                if user_input.lower() in ["exit", "quit"]:
                    console.print("[bold yellow]Exiting session. Goodbye![/bold yellow]")
                    break
                
                # Show thinking animation
                with console.status("[bold green]Processing..."):
                    result = self.process_user_input(user_input)
                
                # Display LLM response
                console.print(Panel(
                    result["llm_response"],
                    title="[bold purple]LLM Response",
                    border_style="purple"
                ))
                
                # Display parsed command
                command_json = json.dumps(result["structured_command"], indent=2)
                console.print(Panel(
                    f"[bold white]{command_json}[/bold white]",
                    title="[bold blue]Parsed Command",
                    border_style="blue"
                ))
                
                # Display execution result
                execution_json = json.dumps(result["execution_result"], indent=2)
                console.print(Panel(
                    f"[bold white]{execution_json}[/bold white]",
                    title="[bold green]Execution Result",
                    border_style="green"
                ))
                
                console.print("\n" + "-" * 80 + "\n")
                
            except KeyboardInterrupt:
                console.print("[bold yellow]Session interrupted. Exiting...[/bold yellow]")
                break
            except Exception as e:
                console.print(f"[bold red]Error: {str(e)}[/bold red]")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Semantic Parser for LLM Responses")
    
    parser.add_argument("--api-key", type=str, help="API key for LLM service")
    parser.add_argument("--api-endpoint", type=str, help="Endpoint URL for LLM service")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Initialize the system
    system = LLMAutomationSystem(
        llm_api_key=args.api_key,
        llm_endpoint=args.api_endpoint
    )
    
    # Run the CLI interface
    system.run_cli()


if __name__ == "__main__":
    main()
        