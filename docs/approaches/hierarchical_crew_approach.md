We will refactor the approach to use `Process.hierarchical` correctly. This is a much cleaner and more robust design within the CrewAI framework.

**Core Concepts for Hierarchical Refactoring:**

1.  **Manager Agent:** We need a dedicated agent to act as the manager (or use `manager_llm`). This agent oversees the workflow, delegates tasks, and synthesizes results. Our previous "Supervisor" concept fits perfectly here.
2.  **Worker Agents:** Agents responsible for specific capabilities (analysis, grouping, validating, merging).
3.  **Task Definition:** Tasks should be defined clearly. The manager will decide which agent performs which task based on the overall goal and agent capabilities.
4.  **Delegation:** Agents (especially the manager) need `allow_delegation=True` to pass tasks to other agents.
5.  **No Crew-in-Tool:** The `BatchDelegationTool` is completely removed. The logic of iterating through batches and assigning work will be handled by the Manager Agent's reasoning, guided by a specific coordination task.

**Revised Plan:**

1.  **Define Agents:**
    *   `PRManagerAgent`: Oversees the entire process. High-level goal setting, delegation, final review. Needs `allow_delegation=True`. This agent will be assigned as the `manager_agent` in the `Crew`.
    *   `AnalysisAgent`: Performs initial analysis (repo, metrics, patterns), strategy selection, and batch splitting. Needs relevant tools. Might not need delegation itself.
    *   `BatchProcessorAgent`: Groups, validates, and refines *a single batch* of files. Needs `FileGrouperTool`, `GroupValidatorTool`, `GroupRefiner`. Does not need delegation.
    *   `MergingAgent`: Merges results from batches, performs final validation and refinement. Needs `GroupMergingTool`, `GroupValidatorTool`, `GroupRefiner`. Might be the same agent as the Manager or a separate specialist. Let's start by having the Manager handle this via tools or delegate to a specialist if needed.

2.  **Define Tools:** Keep all the tools we defined previously *except* `BatchDelegationTool`.
    *   `RepoAnalyzerTool`, `RepositoryMetricsCalculator`, `PatternAnalyzerTool`, `GroupingStrategySelector`, `BatchSplitterTool` (for AnalysisAgent)
    *   `FileGrouperTool`, `GroupValidatorTool`, `GroupRefiner` (for BatchProcessorAgent, *and* for Manager/MergingAgent for final steps)
    *   `GroupMergingTool` (for Manager/MergingAgent)

3.  **Define Tasks:**
    *   **Supervisor/High-Level Tasks:**
        *   `initial_analysis`: Analyze repo. (Input: repo_path, max_files. Output: RepositoryAnalysis JSON) -> `AnalysisAgent`
        *   `calculate_global_metrics`: Calc metrics. (Input: RepositoryAnalysis JSON. Output: RepositoryMetrics JSON) -> `AnalysisAgent`
        *   `analyze_global_patterns`: Find patterns. (Input: RepositoryAnalysis JSON. Output: PatternAnalysisResult JSON) -> `AnalysisAgent`
        *   `select_grouping_strategy`: Choose strategy. (Input: Analysis, Metrics, Patterns JSONs. Output: GroupingStrategyDecision JSON) -> `AnalysisAgent`
        *   `split_into_batches`: Create batches. (Input: RepositoryAnalysis JSON, strategy decision?. Output: BatchSplitterOutput JSON) -> `AnalysisAgent`
        *   `coordinate_batch_processing` (NEW): **This is key.** Instructs the Manager Agent to iterate through the batches from `split_into_batches`. For *each* batch, it should delegate the `process_single_batch` task (see below) to the `BatchProcessorAgent`. It needs to collect the results (refined `PRGroupingStrategy` JSON for each batch). (Input: BatchSplitterOutput JSON, GroupingStrategyDecision JSON. Output: List[str] of worker results JSONs). -> `PRManagerAgent`
        *   `merge_batch_results`: Merge collected results. (Input: List of worker results JSONs, original RepositoryAnalysis JSON. Output: Merged PRGroupingStrategy JSON). -> `PRManagerAgent` (using `GroupMergingTool`)
        *   `final_validation`: Validate merged groups. (Input: Merged PRGroupingStrategy JSON, original RepositoryAnalysis JSON. Output: PRValidationResult JSON). -> `PRManagerAgent` (using `GroupValidatorTool` in final mode)
        *   `final_refinement`: Refine merged groups. (Input: Merged Strategy JSON, Validation Result JSON, Original Analysis JSON. Output: Final PRGroupingStrategy JSON). -> `PRManagerAgent` (using `GroupRefiner` in final mode)
    *   **Delegatable/Worker Task:**
        *   `process_single_batch`: This task represents the work done *per batch*. Its description tells the agent (the `BatchProcessorAgent`) to take a single batch's context (file IDs, global strategy), generate groups, validate them, and refine them. (Input: WorkerBatchContext JSON. Output: Refined PRGroupingStrategy JSON for the batch). -> `BatchProcessorAgent`. This task is *not* in the main crew sequence but is called by the manager during `coordinate_batch_processing`.

4.  **Update `crew.py`:**
    *   Instantiate all agents and tools.
    *   Assign tools correctly to agents.
    *   Define all tasks (including `process_single_batch`, even though it's not in the main sequence).
    *   Define the main sequence of tasks for the `Crew` object (excluding `process_single_batch`).
    *   Instantiate the `Crew` with `process=Process.hierarchical` and `manager_agent=pr_manager_agent`.

5.  **Update YAML Configs:**
    *   `agents.yaml`: Define `pr_manager_agent`, `analysis_agent`, `batch_processor_agent`. Ensure `pr_manager_agent` has `allow_delegation: true`.
    *   `tasks.yaml`: Define all tasks, including the crucial `coordinate_batch_processing` and the delegate target `process_single_batch`.
