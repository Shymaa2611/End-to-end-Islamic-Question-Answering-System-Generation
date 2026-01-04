# End-to-End Islamic Question Answering Generation System

This repository presents an end-to-end Arabic Islamic Question Answering (QA) system designed to answer religious questions using authoritative Islamic sources, namely the Holy Qur’an and Hadith.
The system integrates retrieval-augmented generation with in-context learning (ICL), where carefully selected demonstrations guide the answer generation process to improve accuracy, relevance, and faithfulness to source texts.

![Proposed End-to-End Islamic Question Answering Generation System](images/framework.png)

## Methodology

### Retrieval

Given an input question, the system retrieves the top-K relevant passages from a curated collection of Qur’anic verses and Hadith texts.
Both the question and candidate passages are encoded using a sentence embedding model, and cosine similarity is computed to rank passages based on semantic relevance.

### Reranking
To refine the retrieved results, we employ NAMAA Space GATE Reranker V1 (GATE), a cross-encoder model fine-tuned by NUR for the IslamicEval 2025 Shared Task.
GATE is built upon AraBERT and utilizes the Arabic Triplet Matryoshka training strategy, achieving strong performance while remaining computationally efficient.

For zero-answer detection, a threshold-based decision mechanism is applied. If all reranked passages score below a predefined threshold, the question is classified as having no valid answer, and the system returns −1.
###  In-Context Learning
In-Context Learning (ICL) enables a language model to perform a task by conditioning on a small set of input–output demonstrations included directly in the prompt, without updating model parameters.
While effective, ICL performance is highly sensitive to demonstration selection.
To address this, we adopt a retrieval–reranking-based demonstration selection strategy, inspired by our passage retrieval pipeline:
-For a given input question, candidate passages are retrieved from the training set using a dense 
 retriever.
- These candidates are reranked using the GATE cross-encoder to compute fine-grained relevance scores.
- The top-ranked passages are selected, and their associated question–answer pairs are extracted from the  
  training data.
- These pairs are used as in-context demonstrations.
This approach ensures that demonstrations are not only semantically similar to the input question, but also highly relevant at the interaction level, resulting in more effective and informative ICL.
###  Answer Generation

Final answer generation is performed using a large language model (LLM) from Mistral AI.
The model is explicitly constrained to generate answers only from the retrieved and reranked evidence, ensuring grounded and reliable outputs.
Answer generation is formulated as an answer extraction task using a structured prompt with two components:

- In-Context Demonstrations ,Each demonstration follows a fixed structure:
Context: Retrieved passages
Question: Demonstration query
Answer: Ground-truth response

- Test Instance
Question: User input
Context: Top-K reranked passages
Answer: Left empty for model generation
By enforcing a consistent context–question–answer format, the model is guided to produce answers that are faithful to the retrieved evidence, significantly reducing hallucinations and improving answer quality.

## Installation & Requirements

### Requirements
- Python 3.10
- MistralAI API
- Additional libraries:

``` bash
!pip install huggingface_hub==0.13.4
!pip install -U sentence-transformers
!pip install faiss-cpu
!pip install -U google-generativeai
!pip install -U google-genai

```

### Installation
1- Clone the repository:

```bash

git clone https://github.com/Shymaa2611/End-to-end-Islamic-Question-Answering-System-Generation.git
cd End-to-end-Islamic-Question-Answering-System-Generation

```

2- Create and activate a virtual environment (optional but recommended):
```bash

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

```
3- Install dependencies:
```bash

pip install -r requirements.txt

```

5- Run System:
 - To Run Demonstrations Selection
 ```bash

!python src/in_Context_learning/run.py

```

 - To Run Islamic Question Answering System With In-context Learning
 ```bash

!python src/in_context_learning/QA.py

```




