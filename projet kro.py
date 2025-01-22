from datasets import load_dataset
from pyserini.search.lucene import LuceneSearcher
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import json
import os
from tqdm import tqdm
import tempfile
import shutil
#yeah

class QuestionAnsweringSystem:
    def __init__(self, index_path='nq_index'):
        # Initialize BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
        
        # Load dataset for document lookup
        self.dataset = load_dataset("nq_open")
        
        # Initialize Pyserini searcher
        self.index_path = os.path.abspath(index_path)
        try:
            self.searcher = LuceneSearcher(self.index_path)
        except Exception as e:
            print(f"Index not found at {self.index_path}. Creating new index...")
            self._create_index()
            self.searcher = LuceneSearcher(self.index_path)

    def _create_index(self):
        """Create a Pyserini index from the dataset"""
        # Create temporary directory for documents
        temp_dir = tempfile.mkdtemp()
        docs_dir = os.path.join(temp_dir, "docs")
        os.makedirs(docs_dir, exist_ok=True)
        
        # Create documents for indexing
        print("Creating document files...")
        for i, example in enumerate(tqdm(self.dataset['train'])):
            doc = {
                'id': str(i),
                'contents': f"{example['question']} {' '.join(example['answer'])}",
                'raw': json.dumps({
                    'question': example['question'],
                    'answer': example['answer']
                })
            }
            
            # Write document to a jsonl file
            with open(os.path.join(docs_dir, "docs.jsonl"), 'a', encoding='utf-8') as f:
                f.write(json.dumps(doc) + '\n')
        
        # Create fresh index directory
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        os.makedirs(self.index_path)
        
        # Build indexing command
        cmd = (
            f"python -m pyserini.index.lucene "
            f"--collection JsonCollection "
            f"--input {docs_dir} "
            f"--index {self.index_path} "
            f"--generator DefaultLuceneDocumentGenerator "
            f"--threads 1 "
            f"--storePositions --storeDocvectors --storeContents"
        )
        
        # Execute indexing command
        print("Indexing documents...")
        os.system(cmd)
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print("Indexing complete!")

    def bm25_search(self, query, k=10):
        """Perform BM25 search using Pyserini"""
        try:
            hits = self.searcher.search(query, k)
            results = []
            for hit in hits:
                doc_id = int(hit.docid)
                if doc_id < len(self.dataset['train']):
                    example = self.dataset['train'][doc_id]
                    results.append({
                        'question': example['question'],
                        'answer': example['answer']
                    })
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def generate_answer(self, question, context):
        """Generate answer using BERT"""
        if not context:
            return "Sorry, I couldn't find relevant information to answer this question."
            
        inputs = self.tokenizer(
            question, 
            context, 
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][answer_start:answer_end]
            )
        )
        
        return answer.strip() or "Sorry, I couldn't generate a good answer from the context."

    def question_answering(self, question):
        """Complete question answering pipeline"""
        try:
            # BM25 search
            search_results = self.bm25_search(question)
            
            # Extract contexts
            contexts = []
            for result in search_results:
                context = f"Question: {result['question']} Answer: {' '.join(result['answer'])}"
                contexts.append(context)
            
            # Concatenate contexts with proper spacing
            full_context = " ".join(contexts)
            
            # Generate answer
            answer = self.generate_answer(question, full_context)
            
            return answer
        except Exception as e:
            print(f"Error during question answering: {e}")
            return "Sorry, an error occurred while processing your question."

    def evaluate(self, eval_dataset, num_examples=None):
        """Evaluate the system on a dataset"""
        correct = 0
        total = 0
        
        examples = eval_dataset if num_examples is None else eval_dataset.select(range(num_examples))
        
        for example in tqdm(examples, desc="Evaluating"):
            try:
                question = example["question"]
                ground_truth = example["answer"]
                
                prediction = self.question_answering(question)
                
                if self._exact_match(prediction, ground_truth):
                    correct += 1
                total += 1
            except Exception as e:
                print(f"Error evaluating example: {e}")
                continue
            
        accuracy = correct / total if total > 0 else 0
        return accuracy

    def _exact_match(self, prediction, ground_truth):
        """Check if prediction matches any of the ground truth answers"""
        if not prediction or not ground_truth:
            return False
        prediction = prediction.lower().strip()
        return any(answer.lower().strip() == prediction for answer in ground_truth)

# Usage example
if __name__ == "__main__":
    try:
        # Load dataset
        print("Loading dataset...")
        dataset = load_dataset("nq_open")
        
        # Initialize QA system
        print("Initializing QA system...")
        qa_system = QuestionAnsweringSystem()
        
        # Evaluate on a small subset of validation data
        print("\nEvaluating on validation set...")
        accuracy = qa_system.evaluate(dataset["validation"], num_examples=100)
        print(f"Exact Match Accuracy: {accuracy:.2f}")
        
        # Interactive mode
        print("\nEntering interactive mode (type 'quit' to exit)...")
        while True:
            try:
                question = input("\nEnter your question: ").strip()
                if question.lower() == 'quit':
                    break
                
                if not question:
                    print("Please enter a question.")
                    continue
                
                answer = qa_system.question_answering(question)
                print(f"Answer: {answer}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error processing question: {e}")
                continue
                
    except Exception as e:
        print(f"Program error: {e}")