#!/usr/bin/env python3
import os
import json
import requests
import time
from typing import List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm import tqdm  # progress bar

# load environment variables
load_dotenv()

class LLMAnnotator:
    def __init__(self, input_file: str, output_file: str, max_workers: int = 5):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.api_key = os.getenv("V3_API_KEY")
        self.api_url = "https://api.gpt.ge/v1/chat/completions"
        self.model = "gpt-5-mini-minimal"
        self.max_workers = max_workers

    def _construct_prompt(self, title: str, abstract: str) -> List[Dict]:
        """Construct the prompt to send to the LLM, embedding annotation guidelines."""
        system_content = """
        You are a professional psychology terminology annotation expert. Your task is to extract specific psychology terms from the given research paper abstracts.
        
        Please follow these guidelines:
        1. Extraction targets: Psychology theory concepts (e.g., cognitive dissonance), constructs (self-efficacy), experimental methods (Stroop test), psychological phenomena (memory consolidation).
        2. Exclusion targets: Generic academic vocabulary (study, research), common English words, non-psychology-specific statistical terms.
        3. Format requirements: Strictly output JSON format with a single key "extracted_terms" containing a string list. Do not include any other explanatory text.
        
        Example output:
        {"extracted_terms": ["cognitive dissonance", "attitude change", "motivation"]}
        """

        
        user_content = f"Title: {title}\nAbstract: {abstract}\n\nPlease extract the psychology terms:"
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def annotate_single_doc(self, doc: Dict) -> Dict:
        """Process a single document and return annotated result."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        messages = self._construct_prompt(doc.get('title', ''), doc.get('abstract', ''))
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1, # low temperature for deterministic outputs
            "max_tokens": 1000,
            "response_format": {"type": "json_object"} # force JSON format when supported
        }

        try:
            # Simple retry mechanism
            for attempt in range(3):
                try:
                    response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    
                    content = result['choices'][0]['message']['content']
                    
                    # Parse JSON string
                    try:
                        parsed_json = json.loads(content)
                        terms = parsed_json.get("extracted_terms", [])
                    except json.JSONDecodeError:
                        # If the model returned a Markdown code block, try to clean it
                        clean_content = content.replace("```json", "").replace("```", "").strip()
                        parsed_json = json.loads(clean_content)
                        terms = parsed_json.get("extracted_terms", [])

                    # Write the result into a document copy
                    doc_result = doc.copy()
                    doc_result['llm_annotated_terms'] = terms
                    return doc_result

                except (requests.RequestException, json.JSONDecodeError) as e:
                    if attempt == 2:
                        print(f"Document {doc.get('id')} failed: {e}")
                        doc_result = doc.copy()
                        doc_result['llm_annotated_terms'] = []
                        doc_result['error'] = str(e)
                        return doc_result
                    time.sleep(1) # wait before retry
                    
        except Exception as e:
            print(f"Unexpected error: {e}")
            return doc

    def run(self):
        print(f"Loading data: {self.input_file}")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Starting parallel annotation for {len(data)} documents (workers: {self.max_workers})...")
        annotated_data = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_doc = {executor.submit(self.annotate_single_doc, doc): doc for doc in data}
            
            # Use tqdm to display a progress bar
            for future in tqdm(as_completed(future_to_doc), total=len(data)):
                try:
                    result = future.result()
                    annotated_data.append(result)
                except Exception as exc:
                    print(f"Task raised an exception: {exc}")

        # Save results
        print(f"Saving annotated results to: {self.output_file}")
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(annotated_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Input file path (ensure this is the DEV set you split)
    INPUT_FILE = "data/abstracts/DEV_set_to_annotate.json"
    # Output file path
    OUTPUT_FILE = "data/abstracts/DEV_set_annotated_LLM.json"

    annotator = LLMAnnotator(INPUT_FILE, OUTPUT_FILE, max_workers=8)
    annotator.run()