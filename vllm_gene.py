from vllm import LLM, SamplingParams
import json
from typing import List
from tqdm import tqdm


def prepare_prompts(input_file: str, batch_size: int = 32) -> List[List[str]]:
    """Load prompts from file and create batches"""
    with open(input_file, 'r') as f:
        prompts = [line.strip() for line in f]
    
    # Create batches
    return [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

def generate_responses(prompt_batches: List[List[str]], output_file: str, n_responses: int = 3):
    """Generate multiple responses for batches of prompts"""
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        n=n_responses,  # Generate multiple responses per prompt
    )
    model = LLM(
        model="Qwen/Qwen2.5-3B-Instruct",
        tensor_parallel_size=4,
        trust_remote_code=True,
        dtype="float16",
    )
    with open(output_file, 'w') as f:
        for batch_idx, batch in enumerate(tqdm(prompt_batches)):
            outputs = model.generate(batch, sampling_params)
            
            # Process and save results
            for output in outputs:
                result = {
                    'prompt': output.prompt,
                    'generated_texts': [out.text for out in output.outputs],  # Save all responses
                }
                f.write(json.dumps(result) + '\n')


if __name__ == "__main__":
    input_file = "prompts.txt"
    output_file = "responses.jsonl"
    batch_size = 32
    n_responses = 3  # Number of responses to generate per prompt
    prompt_batches = prepare_prompts(input_file, batch_size)
    generate_responses(prompt_batches, output_file, n_responses)