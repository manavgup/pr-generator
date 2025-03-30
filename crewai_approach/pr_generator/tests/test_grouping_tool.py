from crewai_approach.pr_generator.src.pr_generator.tools.git_tools import GroupingTool 
def test_grouping_tool():
    """
    Test the GroupingTool with sample input
    """
    sample_analysis_result = {
        "description": "Code changes across multiple components",
        "groups": [
            {
                "title": "Authentication Improvements",
                "files": ["auth/main.py", "auth/middleware.py"],
                "rationale": "Enhance authentication mechanisms",
                "suggested_branch_name": "feat/auth-improvements"
            },
            {
                "title": "Error Handling Refactor",
                "files": ["core/error_handler.py", "utils/logging.py"],
                "rationale": "Improve error handling and logging",
                "suggested_branch_name": "refactor/error-handling"
            }
        ]
    }
    
    tool = GroupingTool()
    result = tool.process_changes(sample_analysis_result)
    print(result)