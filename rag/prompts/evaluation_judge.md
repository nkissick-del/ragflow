### Task
You are an expert judge evaluating the performance of a Question Answering system.
Your task is to rate the quality of the generated answer based on the provided question, retrieved context, and reference answer.

### Inputs
- **User Question**: {{ question }}
- **Retrieved Context**: {{ context }}
- **Reference Answer**: {{ reference }}
- **Generated Answer**: {{ answer }}

### Evaluation Criteria
Please score the following metrics on a scale from 0.0 to 1.0:

1. **Faithfulness**: The extent to which the generated answer is derived from the retrieved context. (1.0 = fully supported by context).
2. **Context Relevance**: The extent to which the retrieved context contains information relevant to the user question. (1.0 = highly relevant).
3. **Answer Relevance**: The extent to which the generated answer addresses the user question. (1.0 = fully answers the question).
4. **Semantic Similarity**: The semantic closeness between the generated answer and the reference answer. (1.0 = very similar meaning). If reference answer is not provided (i.e., "None" or empty), output 0.0 for this metric.

### Output Format
Return a valid JSON object with the following keys:
{
    "faithfulness": <float>,
    "context_relevance": <float>,
    "answer_relevance": <float>,
    "semantic_similarity": <float>
}
