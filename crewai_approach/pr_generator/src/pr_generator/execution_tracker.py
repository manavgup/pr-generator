"""
Tracking and visualization for CrewAI agent execution.
"""
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

class AgentState:
    """Represents a state snapshot of an agent during execution."""
    def __init__(self, agent_name: str, task_name: str, state: str, content: str, timestamp: Optional[datetime] = None):
        self.agent_name = agent_name
        self.task_name = task_name
        self.state = state  # e.g., "started", "thinking", "tool_use", "completed"
        self.content = content
        self.timestamp = timestamp or datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "task_name": self.task_name,
            "state": self.state,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }

class ExecutionTracker:
    """Tracks and visualizes CrewAI agent execution."""
    
    def __init__(self, output_dir: str = "execution_logs"):
        """
        Initialize the execution tracker.
        
        Args:
            output_dir: Directory to store execution logs and visualizations
        """
        self.states: List[AgentState] = []
        self.output_dir = output_dir
        self.execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.message_patterns = {
            "thinking": [
                "I need to think about", "Let me analyze", "I'll consider", 
                "Thinking through", "Analyzing"
            ],
            "tool_use": [
                "Using tool:", "I'll use the", "Executing tool", 
                "Calling tool", "Tool Input:"
            ],
            "conclusion": [
                "Therefore,", "In conclusion", "To summarize", 
                "My analysis shows", "Final Answer:", "Based on my analysis"
            ]
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def log_state(self, agent_name: str, task_name: str, state: str, content: str) -> None:
        """
        Log a state of an agent.
        
        Args:
            agent_name: Name of the agent
            task_name: Name of the task
            state: Current state (e.g., "started", "thinking", "tool_use")
            content: Content or message related to the state
        """
        state = AgentState(agent_name, task_name, state, content)
        self.states.append(state)
        logger.debug(f"Tracked state: {agent_name} - {state.state}")
        
    def detect_states_from_output(self, agent_name: str, task_name: str, output: str) -> None:
        """
        Analyze raw agent output to detect and log states.
        
        Args:
            agent_name: Name of the agent
            task_name: Name of the task
            output: Raw output from the agent
        """
        # Split output into lines
        lines = output.split('\n')
        
        for line in lines:
            # Check for thinking patterns
            for pattern in self.message_patterns["thinking"]:
                if pattern in line:
                    self.log_state(agent_name, task_name, "thinking", line.strip())
                    break
                    
            # Check for tool use patterns
            for pattern in self.message_patterns["tool_use"]:
                if pattern in line:
                    self.log_state(agent_name, task_name, "tool_use", line.strip())
                    break
                    
            # Check for conclusion patterns
            for pattern in self.message_patterns["conclusion"]:
                if pattern in line:
                    self.log_state(agent_name, task_name, "conclusion", line.strip())
                    break
    
    def get_hook_callbacks(self, agent_name: str, task_name: str) -> Dict[str, Callable]:
        """
        Get callback functions to use as hooks in CrewAI tasks.
        
        Args:
            agent_name: Name of the agent
            task_name: Name of the task
            
        Returns:
            Dictionary of hook callbacks for starting, in-progress, and completion
        """
        def on_start(agent, task):
            self.log_state(agent_name, task_name, "started", f"Starting task: {task_name}")
            
        def on_message(agent, task, message):
            self.detect_states_from_output(agent_name, task_name, message)
            
        def on_end(agent, task, output):
            self.log_state(agent_name, task_name, "completed", f"Completed task: {task_name}")
            
        return {
            "on_task_start": on_start,
            "on_agent_message": on_message,
            "on_task_end": on_end
        }
    
    def save_execution_log(self) -> str:
        """
        Save the execution log to a JSON file.
        
        Returns:
            Path to the saved file
        """
        log_file = os.path.join(self.output_dir, f"execution_log_{self.execution_id}.json")
        
        with open(log_file, 'w') as f:
            json.dump([state.to_dict() for state in self.states], f, indent=2)
            
        logger.info(f"Saved execution log to {log_file}")
        return log_file
    
    def generate_html_timeline(self) -> str:
        """
        Generate an HTML timeline visualization.
        
        Returns:
            Path to the generated HTML file
        """
        html_file = os.path.join(self.output_dir, f"timeline_{self.execution_id}.html")
        
        # Generate basic timeline HTML
        agent_colors = {}
        agents = set(state.agent_name for state in self.states)
        
        # Generate colors for agents
        import random
        colors = ["#ff7675", "#74b9ff", "#55efc4", "#ffeaa7", "#a29bfe", "#fd79a8", "#81ecec"]
        for i, agent in enumerate(agents):
            agent_colors[agent] = colors[i % len(colors)]
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CrewAI Execution Timeline</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .timeline {{ display: flex; flex-direction: column; }}
                .agent-row {{ display: flex; margin-bottom: 15px; }}
                .agent-name {{ width: 150px; padding: 5px; text-align: right; font-weight: bold; }}
                .timeline-events {{ display: flex; flex-grow: 1; }}
                .event {{ 
                    margin: 2px; padding: 5px; border-radius: 4px; 
                    min-width: 100px; position: relative; 
                }}
                .event-content {{ display: none; position: absolute; background: white; 
                               border: 1px solid #ddd; padding: 10px; width: 300px;
                               z-index: 100; top: 25px; left: 0; }}
                .event:hover .event-content {{ display: block; }}
                .event.started {{ background-color: #81ecec; }}
                .event.thinking {{ background-color: #74b9ff; }}
                .event.tool_use {{ background-color: #a29bfe; }}
                .event.conclusion {{ background-color: #55efc4; }}
                .event.completed {{ background-color: #ffeaa7; }}
                .legend {{ display: flex; margin-bottom: 20px; }}
                .legend-item {{ margin-right: 15px; display: flex; align-items: center; }}
                .legend-color {{ width: 20px; height: 20px; margin-right: 5px; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <h1>CrewAI Execution Timeline</h1>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #81ecec;"></div>
                    <span>Started</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #74b9ff;"></div>
                    <span>Thinking</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #a29bfe;"></div>
                    <span>Tool Use</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #55efc4;"></div>
                    <span>Conclusion</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #ffeaa7;"></div>
                    <span>Completed</span>
                </div>
            </div>
            <div class="timeline">
        """
        
        # Group states by agent
        agents_states = {}
        for state in self.states:
            if state.agent_name not in agents_states:
                agents_states[state.agent_name] = []
            agents_states[state.agent_name].append(state)
        
        # Generate timeline for each agent
        for agent, states in agents_states.items():
            html_content += f"""
                <div class="agent-row">
                    <div class="agent-name" style="color: {agent_colors.get(agent, '#000')};">{agent}</div>
                    <div class="timeline-events">
            """
            
            for state in states:
                truncated_content = state.content[:50] + "..." if len(state.content) > 50 else state.content
                time_str = state.timestamp.strftime("%H:%M:%S")
                
                html_content += f"""
                        <div class="event {state.state}" title="{time_str}">
                            {time_str} - {truncated_content}
                            <div class="event-content">{state.content}</div>
                        </div>
                """
            
            html_content += """
                    </div>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(html_file, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Generated HTML timeline at {html_file}")
        return html_file
    
    def generate_mermaid_diagram(self) -> str:
        """
        Generate a Mermaid sequence diagram of agent interactions.
        
        Returns:
            Path to the generated Mermaid diagram file
        """
        mermaid_file = os.path.join(self.output_dir, f"sequence_{self.execution_id}.md")
        
        mermaid_content = """
```mermaid
sequenceDiagram
    participant User
"""
        
        # Add participants
        agents = set(state.agent_name for state in self.states)
        for agent in agents:
            mermaid_content += f"    participant {agent.replace(' ', '_')}\n"
        
        # Add sequence
        last_agent = "User"
        for state in self.states:
            agent = state.agent_name.replace(' ', '_')
            if state.state == "started":
                mermaid_content += f"    User->>+{agent}: Run task {state.task_name}\n"
                last_agent = agent
            elif state.state == "tool_use":
                tool_name = state.content.split("Using tool:")[1].strip() if "Using tool:" in state.content else "tool"
                mermaid_content += f"    {agent}->>+{agent}: Uses {tool_name}\n"
            elif state.state == "completed":
                mermaid_content += f"    {agent}-->>-User: Return results\n"
        
        mermaid_content += "```"
        
        with open(mermaid_file, 'w') as f:
            f.write(mermaid_content)
            
        logger.info(f"Generated Mermaid diagram at {mermaid_file}")
        return mermaid_file

    def visualize_execution(self) -> Dict[str, str]:
        """
        Generate all visualizations and save logs.
        
        Returns:
            Dictionary mapping visualization type to file path
        """
        results = {}
        
        # Save execution log
        results["log"] = self.save_execution_log()
        
        # Generate HTML timeline
        results["timeline"] = self.generate_html_timeline()
        
        # Generate Mermaid diagram
        results["sequence"] = self.generate_mermaid_diagram()
        
        logger.info(f"Generated all visualizations: {results}")
        return results