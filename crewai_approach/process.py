# crewai_approach/process.py

import argparse
from typing import Any, Dict

class CrewBase:
    """Base class for all CrewAI agents."""
    pass

def Task(name: str = None):
    """Decorator to mark a method as a task."""
    def decorator(fn):
        return fn
    return decorator

def before_task(task_name: str):
    """Decorator to run setup before a given task."""
    def decorator(fn):
        return fn
    return decorator

class Process:
    """Minimal stub for Process orchestration."""
    class Parser(argparse.ArgumentParser):
        """Wraps argparse.ArgumentParser for CLI parsing."""
        pass

    @staticmethod
    def hierarchical(agent: CrewBase, kickoff_task: str, **kwargs) -> Any:
        """
        Return an object with a kickoff() method.
        In the real SDK this wires up staged execution;
        here it simply returns the agent itself.
        """
        return agent

    def kickoff(self, inputs: Dict[str, Any]) -> None:
        """
        Stub kickoff: just print inputs so you can verify invocation.
        """
        print("Running hierarchical workflow with inputs:")
        for k, v in inputs.items():
            print(f"  {k}: {v}")