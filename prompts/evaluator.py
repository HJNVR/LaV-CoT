evaluate_prompt = """
You are a vision-language assistant tasked with evaluating a specific Chain-of-Thought (CoT) step.
Input: Image, Target Language: {Target language}, Question: {Question}, CoT Step to Evaluate: {s_i}, Final Answer: {Final Answer}
Goal:
Identify the specific part(s) of {s_i} that are incorrect or unsupported by the image/text evidence.
Tasks:
1. Text Extraction & Summarization:
Detect visible text in the image, summarize in {Target language}, and include bounding boxes:
```json[{{"summary": "<text in target language>", "bbox": [x_min, y_min, x_max, y_max]}}]```
2.Language Identification:
Identify main text language (\lang{{}})
3.Spatial Image Caption:
Describe objects, spatial positions, relationships, and link them with extracted text; count total objects (\obj{{}})
4.Step Evaluation:
Analyze {s_i} using extracted text and image context.
Assign a correctness score between 0 and 1.
If the step is not fully correct, locate erroneous part(s) {s_error}.

Output:
{“score”: <float between 0 and 1>,
“s_error”: “<text of erroneous part, empty if fully correct>”}
"""