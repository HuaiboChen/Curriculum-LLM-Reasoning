from vllm import LLM, SamplingParams
import json
from typing import List
from tqdm import tqdm
from datasets import load_dataset
import hydra
from omegaconf import DictConfig


def prepare_prompts(input_file: str, batch_size: int = 32) -> List[List[str]]:
    """Load prompts from file and create batches"""
    dataset = load_dataset('json', data_files=input_file)['train']
    prompts = [item['question'] for item in dataset]
    return [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

@hydra.main(version_base=None, config_path="config", config_name="generation")
def main(cfg: DictConfig):
    input_file = cfg.data.input_file
    output_file = cfg.data.results_file
    batch_size = cfg.data.batch_size
    n_responses = cfg.generation.n_responses
    
    prompt_batches = prepare_prompts(input_file, batch_size)
    
    sampling_params = SamplingParams(
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        max_tokens=cfg.generation.max_tokens,
        n=n_responses,
    )
    
    model = LLM(
        model=cfg.model.name,
        tensor_parallel_size=cfg.model.tensor_parallel_size,
        trust_remote_code=cfg.model.trust_remote_code,
        dtype=cfg.model.dtype,
    )
    
    with open(output_file, 'w') as f:
        for batch_idx, batch in enumerate(tqdm(prompt_batches)):
            outputs = model.generate(batch, sampling_params)
            
            # Process and save results
            for output in outputs:
                result = {
                    'prompt': output.prompt,
                    'generated_texts': [out.text for out in output.outputs],
                }
                f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    main()