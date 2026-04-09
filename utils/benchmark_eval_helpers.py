import json
import numpy as np
from tqdm import tqdm
import time
from typing import List, Union, Dict, Any, Optional


def load_json(
    path: str 
) -> List[Dict]:
    """
    Loads a JSON of JSONL file.

    args:
    - path (str): the path to the dataset file

    returns:
    - a list of items in the dataset (in Python dict format)
    """
    dataset = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                json_object = json.loads(line.strip())
                dataset.append(json_object)

        if not dataset:
            raise ValueError("The loaded dataset file is empty!")

    except FileNotFoundError:
        print("The file does not exist.")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    return dataset


def output_to_file(
        items: List[Dict],
        path: str
    ):
    """
    Writes a list of dictionaries to a .jsonl file.

    args:
    - items (List[Dict]): a list of Python dicts to be written to a file
    - path (str): path of output file
    """
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for i in range(len(items)):
                json.dump(items[i], f)
                f.write("\n")

    except Exception as e:
        print(f"An error occurred: {e}")


def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
    ) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.

    args:
    - num_samples (Union[int, List[int], np.ndarray]): number of samples generated per problem
    - num_correct (Union[List[int], np.ndarray]): number of correct samples per problem
    - k (int): the k value for pass@k metric

    returns:
    - an array of pass@k estimates for each problem (Np Array)
    """
    if isinstance(num_samples, int):
        num_samples_arr = np.full_like(num_correct, num_samples, dtype=np.float64)
    else:
        num_samples_arr = np.asarray(num_samples, dtype=np.float64)

    num_correct_arr = np.asarray(num_correct, dtype=np.float64)

    pass_at_k_values = np.zeros_like(num_samples_arr)

    for i in range(len(num_samples_arr)):
        n = num_samples_arr[i]
        c = num_correct_arr[i]
        incorrect = n - c

        if incorrect < k:
            pass_at_k_values[i] = 1.0   # not enough incorrect samples, must have at least one correct
        else:
            p_allwrong = 1.0
            for j in range(k):
                p_allwrong *= (n - c - j) / (n - j)

            pass_at_k_values[i] = 1 - p_allwrong

    return pass_at_k_values


def judge_answer(
        client,
        question: str,
        generated_ans: str,
        ground_truths: List[str]
    ) -> Dict[str, Any]:
    """
    Uses an LLM to judge if a generated answer matches ground truth.

    args:
    - client: LLM client to use
    - question (str): the question to judge answers for
    - generated_ans (str): the generated answer answer
    - ground_truths (List[str]): a list of acceptable ground truth answers

    returns:
    - a dictionary with 'correct' (bool), 'explanation' (str), and 'raw_response' (str)
    """
    ground_truths_str = " OR ".join(f'"{gt}"' for gt in ground_truths)

    prompt = f"""You are an evaluation judge. Determine if the generated answer is correct given the ground truth answer(s).

Question: {question}

Generated Answer: {generated_ans}

Ground Truth Answer(s): {ground_truths_str}

The generated answer should be considered CORRECT if it:
1. Contains the same factual information as any of the ground truth answers
2. Is semantically equivalent (e.g., "December 1972" matches "14 December 1972 UTC")
3. Minor formatting differences are acceptable (e.g., "James I" vs "James I.")

The generated answer should be considered INCORRECT if:
1. It contains wrong factual information
2. It's too vague or ambiguous
3. It contradicts the ground truth

Respond with ONLY "CORRECT" or "INCORRECT" followed by a brief explanation.

Format:
CORRECT/INCORRECT: <brief explanation>"""

    try:
        response = client.generate_response(user_prompt=prompt)

        result_text = response.get("answer", "")

        # parse result
        if result_text.startswith("CORRECT"):
            correct = True
            explanation = result_text.split(":", 1)[1].strip() if ":" in result_text else result_text
        elif result_text.startswith("INCORRECT"):
            correct = False
            explanation = result_text.split(":", 1)[1].strip() if ":" in result_text else result_text
        else:
            # default to incorrect if format is unexpected
            correct = False
            explanation = f"Unexpected format: {result_text}"

        return {
            "correct": correct,
            "explanation": explanation,
            "raw_response": result_text
        }

    except Exception as e:
        return {
            "correct": False,
            "explanation": f"Error during judgment: {str(e)}",
            "raw_response": ""
        }


def grade_with_llm_judge(
        responses: List[Dict[str, Any]],
        client,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
    """
    Grades generated responses using LLM-as-judge technique.

    args:
    - responses (List[Dict[str, Any]]): List of student response dicts
    - client: LLM client to use
    - output_file (Optiona;[str]): optional path to save detailed results to

    returns:
    - a dictionary with grading metrics and detailed results
    """
    results = []
    correct_count = 0
    total_count = 0

    print(f"\nGrading {len(responses)} generated responses using LLM judge...")

    for resp in tqdm(responses, desc="Grading"):
        question = resp['question']
        generated_answer = resp.get('llm_response', '')
        ground_truths = resp['answers']

        # use LLM to judge
        judgment = judge_answer(client, question, generated_answer, ground_truths)

        if judgment['correct']:
            correct_count += 1
        total_count += 1

        results.append({
            'id': resp.get('id', ''),
            'question': question,
            'student_answer': generated_answer,
            'ground_truths': ground_truths,
            'correct': judgment['correct'],
            'explanation': judgment['explanation'],
            'raw_judge_response': judgment['raw_response']
        })

        # rate limiting
        time.sleep(0.1)

    accuracy = correct_count / total_count if total_count > 0 else 0.0

    final_results = {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': total_count,
        'detailed_results': results
    }

    # save to file
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {output_file}")

    return final_results