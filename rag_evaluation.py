import csv
# Import RAG pipeline utilities
from rag_pipeline import (
    load_environment,
    load_and_split_documents,
    create_vectorstore,
    build_qa_chain
)
from langchain.evaluation.qa import QAEvalChain, QAGenerateChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# === Configuration ===
TRANSCRIPT_PATH = './transcripts'  # Path to the full transcript text file
OUTPUT_CSV = './evaluation/evaluation_results.csv'  # Output CSV file for evaluation results
NUM_EXAMPLES = 50                   # Number of QA examples to generate
TRANSCRIPT_FILE= './transcripts/How_to_get_a_Band_8_in_IELTS_listening.txt'

def load_pipeline(return_source_documents):
    """
    Load environment variables, prepare documents, create the vectorstore,
    and initialize the QA chain.

    Returns:
        qa_chain: The RetrievalQA chain for answering queries.
        chat_model: The LLM model used for generation and evaluation.
    """
    api_key, chat_model , _= load_environment()
    docs = load_and_split_documents(TRANSCRIPT_PATH)
    vectorstore = create_vectorstore(
        docs=docs,
        api_key=api_key
    )
    qa_chain = build_qa_chain(vectorstore, chat_model, return_source_documents)
    return qa_chain, chat_model


def generate_examples(chat_model, num_examples: int = NUM_EXAMPLES) -> list[dict]:
    """
    Use QAGenerateChain to create exactly `num_examples` QA pairs from the transcript,
    by calling the chain in a loop.
    """
    # 1. Load the full transcript text
    with open(TRANSCRIPT_FILE, 'r', encoding='utf-8') as f:
        transcript = f.read()

    # 2. Initialize the generation chain
    gen_chain = QAGenerateChain.from_llm(chat_model)

    # 3. Loop to generate one QA pair per iteration
    inputs = [{"doc": transcript} for _ in range(NUM_EXAMPLES)]
    examples = gen_chain.generate(inputs)
    examples = [{"query": q, "answer": a} for (q, a) in examples]
    return examples


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
    results = []
    for example, prediction in zip(examples, predictions):
        # Defensive: Ensure keys exist
        if 'query' not in example or 'answer' not in example or 'prediction' not in prediction:
            raise ValueError(f"Missing required key in example or prediction: {example}, {prediction}")
        result = eval_chain.evaluate(
            examples=[example],  # single example
            predictions=[prediction],
            question_key='query',
            prediction_key='prediction',
            answer_key='answer'
        )
        results.append(result)
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
            if isinstance(res, list):
                    res = res[0] if res else {}
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
        res = res[0] if res else {}
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
    qa_chain, chat_model = load_pipeline(return_source_documents=False)
    examples = generate_examples(chat_model,30)
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


import csv

def results_to_csv(results, csv_path):
    """
    Converts a list of dictionaries (results) to a CSV file.

    Parameters:
    - results: List[Dict[str, Any]], where each dict represents a row.
    - csv_path: str, path to the output CSV file.
    """
    if not results:
        raise ValueError("The results list is empty.")

    # Determine CSV columns from the first result
    columns = list(results[0].keys())

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in results:
            # Ensure all values are stringifiable
            clean_row = {col: (row.get(col, "") if isinstance(row.get(col, ""), str) else str(row.get(col, "")))
                         for col in columns}
            writer.writerow(clean_row)
import csv

def results_to_csv_flat(results, csv_path):
    """
    Converts a list of dictionaries (results) to a CSV file,
    flattening the 'faithfulness' dict into separate 'label' and 'explanation' columns.
    
    Parameters:
    - results: List[Dict[str, Any]], where each dict represents a row and may include a 'faithfulness' dict.
    - csv_path: str, path to the output CSV file.
    """
    if not results:
        raise ValueError("The results list is empty.")
    
    # Flatten each row
    flattened = []
    for row in results:
        new_row = {}
        for key, value in row.items():
            if key == 'faithfulness' and isinstance(value, dict):
                new_row['label'] = value.get('label', '')
                new_row['explanation'] = value.get('explanation', '')
            else:
                # Ensure all other values are stringifiable
                new_row[key] = value if isinstance(value, str) else str(value)
        flattened.append(new_row)
    
    # Determine CSV columns from the first flattened row
    columns = list(flattened[0].keys())
    
    # Write to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for row in flattened:
            writer.writerow(row)

# Example usage:
# results_to_csv_flat(results, '/mnt/data/flattened_results.csv')
# print("CSV file saved to /mnt/data/flattened_results.csv")



def main():
    accuracy = evaluate_test_set()
    total = len(generate_examples(load_pipeline()[1]))
    correct = int(total * accuracy / 100)
    print(f"Evaluation completed. Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Detailed results saved to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
