import os
import csv

from rag_pipeline import (
    load_environment,
    load_and_split_documents,
    create_vectorstore,
    build_qa_chain,
)
from langchain.evaluation.qa import QAEvalChain, QAGenerateChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# === Configuration ===
TRANSCRIPT_PATH = 'transcript.txt'  # Path to the full transcript text file
INDEX_DIR = 'faiss_index'           # Directory for the FAISS index
OUTPUT_CSV = 'evaluation_results.csv'  # Output CSV file for evaluation results
NUM_EXAMPLES = 10                   # Number of QA examples to generate


def load_pipeline():
    """
    Load environment variables, prepare documents, create the vectorstore,
    and initialize the QA chain.

    Returns:
        qa_chain: The RetrievalQA chain for answering queries.
        chat_model: The LLM model used for generation and evaluation.
    """
    api_key, chat_model = load_environment()
    docs = load_and_split_documents(TRANSCRIPT_PATH)
    vectorstore = create_vectorstore(
        docs,
        api_key,
        store_type='faiss',
        persist_dir=INDEX_DIR
    )
    qa_chain = build_qa_chain(vectorstore, chat_model,False)
    return qa_chain, chat_model


def generate_examples(chat_model, num_examples: int = NUM_EXAMPLES) -> list[dict]:
    """
    Use QAGenerateChain to create QA examples from the transcript.

    Args:
        chat_model: The LLM instance for generation.
        num_examples: Maximum number of QA pairs to generate.

    Returns:
        List of dicts, each containing 'query' and 'answer'.
    """
    # Read the transcript text
    with open(TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    # Initialize QA generation chain
    gen_chain = QAGenerateChain.from_llm(chat_model)
    # Generate QA examples
    qa_examples = gen_chain.generate(
        text=text,
        max_questions=num_examples
    )
    return qa_examples


def generate_predictions(qa_chain, examples: list[dict]) -> list[dict]:
    """
    Run the QA chain on each generated question to get model predictions.

    Args:
        qa_chain: The RetrievalQA chain for answering queries.
        examples: List of QA example dicts with 'query' keys.

    Returns:
        List of dicts, each containing 'prediction'.
    """
    predictions = []
    for ex in examples:
        resp = qa_chain({'query': ex['query']})
        predictions.append({'prediction': resp.get('result', '').strip()})
    return predictions


def run_evaluation(chat_model, examples: list[dict], predictions: list[dict]) -> list[dict]:
    """
    Initialize QAEvalChain and evaluate predictions against reference answers.

    Args:
        chat_model: The LLM instance used for evaluation.
        examples: List of QA example dicts with 'query' and 'answer'.
        predictions: List of prediction dicts with 'prediction'.

    Returns:
        List of evaluation result dicts, each containing 'score' and optional 'reasoning'.
    """
    eval_chain = QAEvalChain.from_llm(chat_model)
    results = eval_chain.evaluate(
        examples=examples,
        predictions=predictions,
        question_key='query',
        prediction_key='prediction',
        answer_key='answer'
    )
    return results


def write_results(examples: list[dict], predictions: list[dict], results: list[dict]):
    """
    Write detailed evaluation results to a CSV file.

    Columns: question, ground_truth, prediction, score, reasoning.

    Args:
        examples: List of QA examples.
        predictions: List of model predictions.
        results: List of evaluation outputs.
    """
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['question', 'ground_truth', 'prediction', 'score', 'reasoning'])
        for ex, pred, res in zip(examples, predictions, results):
            writer.writerow([
                ex['query'],
                ex['answer'],
                pred['prediction'],
                res.get('score') or res.get('results'),
                res.get('reasoning', '')
            ])


def compute_accuracy(results: list[dict]) -> float:
    """
    Compute accuracy as the percentage of correct results.

    A correct result has score == 1 or results == 'CORRECT'.

    Args:
        results: List of evaluation result dicts.

    Returns:
        Accuracy percentage (0-100).
    """
    total = len(results)
    correct = 0
    for res in results:
        score = res.get('score') or (1 if res.get('results') == 'CORRECT' else 0)
        if score == 1:
            correct += 1
    return (correct / total * 100) if total > 0 else 0


def evaluate_test_set():
    """
    End-to-end evaluation: generate examples, predict, evaluate, and save results.

    Returns:
        Accuracy percentage (0-100).
    """
    qa_chain, chat_model = load_pipeline()
    examples = generate_examples(chat_model)
    predictions = generate_predictions(qa_chain, examples)
    results = run_evaluation(chat_model, examples, predictions)
    write_results(examples, predictions, results)
    return compute_accuracy(results)

def check_faithfulness(examples: list[dict], qa_chain, llm) -> list[dict]:
    """
    For each example, run the QA chain to get an answer and sources,
    then assess whether the answer is fully supported by the sources.

    Returns a list of dicts with keys:
      - 'query'
      - 'answer'
      - 'context' (concatenated source documents)
      - 'faithfulness' (YES/NO + explanation)
    """
    # Define the faithfulness prompt
    faithfulness_prompt = """
                    Given the following retrieved documents and the answer, is the answer fully supported by the documents? Answer YES or NO and explain.

                    Retrieved Documents:
                    {context}

                    Answer:
                    {answer}
                    """
    # Build the LLM chain for faithfulness checking
    faithfulness_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(faithfulness_prompt)
    )

    results = []
    for ex in examples:
        out = qa_chain({'query': ex['query']})
        context = "".join(
            doc.page_content for doc in out['source_documents']
        )
        answer = out['result']
        # Run faithfulness check
        raw_faith = faithfulness_chain.run(
            context=context,
            answer=answer
        ).strip()

        # Parse raw_faith into a dict with label and explanation
        # Assume response starts with YES or NO
        label = None
        explanation = raw_faith
        lower = raw_faith.lower()
        if lower.startswith('yes'):
            label = 'YES'
            explanation = raw_faith[len(label):].strip(" :.-")
        elif lower.startswith('no'):
            label = 'NO'
            explanation = raw_faith[len(label):].strip(" :.-")
        faithfulness = {'label': label, 'explanation': explanation}
        results.append({
            'query': ex['query'],
            'answer': answer,
            'context': context,
            'faithfulness': faithfulness
        })
    return results


def main():
    accuracy = evaluate_test_set()
    total = len(generate_examples(load_pipeline()[1]))
    correct = int(total * accuracy / 100)
    print(f"Evaluation completed. Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Detailed results saved to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
