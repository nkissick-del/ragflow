### Task
You are an expert judge evaluating the performance of a Question Answering system.
Your task is to rate the quality of the generated answer based on the provided question, retrieved context, and reference answer.

### Inputs
- **User Question**: {{ question }}
- **Retrieved Context**: {{ context }}
- **Reference Answer**: {{ reference }}
- **Generated Answer**: {{ answer }}

### Evaluation Criteria
Please score the following metrics. All float scores must be between 0.0 and 1.0, rounded to two decimal places.

1. **Faithfulness**: The extent to which the generated answer is derived from the retrieved context.
   - Score rules: Rate based on the proportion of claims in the answer that are supported by the context.
   - Critical Hallucination: If the answer contradicts the context on a critical fact, score 0.0.
   - If no context/empty context, score 0.0 unless the answer explicitly states it cannot answer from context (in which case, score 1.0 for faithfulness).

2. **Context Relevance**: The extent to which the retrieved context contains information relevant to the user question.
   - Score rules: Consider both precision (how much of the context is useful) and recall (does it contain the necessary info).
   - 1.0: Context contains all necessary information and little irrelevant noise.
   - 0.0: Context is completely irrelevant.

3. **Answer Relevance**: The extent to which the generated answer addresses the user question.
   - Score rules: Judge both completeness (answers all parts of the question) and directness.
   - 1.0: Fully answers the question directly.
   - 0.0: Completely irrelevant or refuses to answer when it should.

4. **Semantic Similarity**: The semantic closeness between the generated answer and the reference answer.
   - (1.0 = very similar meaning).
   - **Important**: If reference answer is not provided (i.e., "None" or empty), output `null` for this metric. Do not output 0.0.

### Edge Cases
- If evaluation cannot be performed for a valid reason (e.g., empty inputs), set field "evaluation_status" to "failed" and provide a reason.

### Output Format
Return a valid JSON object with the following keys. Ensure numeric values are floats [0.0, 1.0] or null where permitted.

{
    "faithfulness": <float or null>,
    "faithfulness_explanation": "<optional explanation>",
    "context_relevance": <float or null>,
    "context_relevance_explanation": "<optional explanation>",
    "answer_relevance": <float or null>,
    "answer_relevance_explanation": "<optional explanation>",
    "semantic_similarity": <float or null>,
    "semantic_similarity_explanation": "<optional explanation>",
    "evaluation_status": "success" | "failed",
    "error": "<optional error message>"
}
