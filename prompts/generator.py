vanilla_cot_prompt = """You are a vision-language assistant. 
Input: 
Image, Target Language: {Target language},  Question: {Question},  Final Answer: {Final Answer}
Goal: 
Extract text & visual context from the image, explain step-by-step how the final answer is derived, using only evidence from the image.
Tasks:
1. Text Extraction & Summarization: 
Detect visible text, summarize in {Target language}, include bounding boxes:
```json[{{"summary": "<text in target language>", "bbox": [x_min, y_min, x_max, y_max]}}]```
2. Language Identification: 
Identify main text language in \lang{{}}
3. Spatial Image Caption: 
Describe objects, spatial positions, relationships, link with extracted texts, count total objects in \obj{{}}
4. Step-by-step Reasoning: 
Break down the question, reference extracted text + objects + caption, explain logically how evidence supports the {Final Answer}.
5. Language Consistency: Use {Target language} throughout all steps.

Output:
<think></think>: Full reasoning steps (text, caption, reasoning, linked evidence).
<answer></answer>: Final answer in {Target language}.
"""

correct_cot_prompt = """You are a vision-language assistant tasked with correcting a specific erroneous CoT step.
Input: Image, Target Language: {Target language}, Question: {Question}, Previous CoT Step (optional context): {s_prev}, Current Erroneous Step: {s_error}, Final Answer: {Final Answer}
Goal:
- Generate a corrected version of {s_error} that is logically consistent with the Final Answer.
Tasks:
1. Text Extraction & Summarization:
Detect all visible text in the image, summarize in {Target language}, and include bounding boxes:
```json[{{"summary": "<text in target language>", "bbox": [x_min, y_min, x_max, y_max]}}]```
2. Language Identification:
Identify the main language of the extracted text (\lang{{}})
3. Spatial Image Caption:
Describe objects, spatial positions, relationships, and link them with extracted text; count total objects (\obj{{}})
4. Correct Step Generation:
Using {s_error} and optionally {s_prev}, produce the corrected CoT step.
5. Language Consistency:
Ensure all outputs use {Target language}.

Output:
Corrected_step
"""