import os
import sys
from addict import Dict
import torch
import pandas as pd

from retab.datasets import Preprocessor
from retab.models import BaseTrainer
from retab.utils import get_summary_metrics

from retab.models.zeroshotllm.zeroshotllm import ZeroShotLLM

from .zeroshotllm.llm_utils import llm_call
from .zeroshotllm.prompt_generator import ADPromptGenerator



class Trainer(BaseTrainer):
    def __init__(self, data_params: Dict, model_params: Dict, preprocessor: Preprocessor, meta_info: Dict):
        super().__init__(data_params, model_params, preprocessor, meta_info)
        
        # Ensure we're working with serialized data
        if not self.is_serialized:
            raise ValueError("ZeroShotLLM requires serialized data. Set serialize=True in data_parameters.")
        
        self.model_params = model_params
        self.model = ZeroShotLLM(
            model_name=getattr(model_params, 'llm_model', 'gemini-2.5-pro'),
            batch_size=getattr(model_params, 'inference_batch_size', data_params.batch_size),
            max_retry=getattr(model_params, 'max_retry', 3),
            prompt_type=getattr(model_params, 'prompt_type', 'D')
        )
        
        # Set up LLM call function
        if llm_call is not None:
            self.model.set_llm_call_func(llm_call)
        else:
            print("Warning: LLM call function not available. Model will not work properly.")
        
        self.ckpt_path = os.path.join(
            meta_info.checkpoint_path, 
            meta_info.data_name, 
            meta_info.model_name, 
            meta_info.exp_id, 
            f'{meta_info.seed}.pth'
        )
        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
        
        # Initialize prompt generator if available
        self.prompt_generator = None
        if ADPromptGenerator is not None:
            try:
                # Calculate statistics from training data (already normal samples only)
                train_normal_df = pd.DataFrame(self.X_train, columns=self.column_names)
                
                # Calculate statistics for each numerical column
                stats = {}
                for col in self.column_names:
                    col_data = train_normal_df[col].dropna()
                    if len(col_data) > 0:
                        stats[col] = {
                            'count': len(col_data),
                            'mean': float(col_data.mean()),
                            'std': float(col_data.std()),
                            'min': float(col_data.min()),
                            'max': float(col_data.max()),
                            'q5': float(col_data.quantile(0.05)),
                            'q25': float(col_data.quantile(0.25)),
                            'q50': float(col_data.quantile(0.50)),
                            'q75': float(col_data.quantile(0.75)),
                            'q95': float(col_data.quantile(0.95))
                        }
                
                # Load text semantic
                metadata = self.preprocessor.textmeta

                # Create dataset info for prompt generation
                dataset_info = {
                    'metadata': metadata,
                    'train_data': train_normal_df,  # Normal training data for categorical analysis
                    'stats': stats  # Calculated statistics from normal samples
                }
                self.prompt_generator = ADPromptGenerator(dataset_info)
            except Exception as e:
                print(f"Warning: Could not initialize prompt generator: {e}")

    def train(self):
        """No training needed for zero-shot approach"""
        print("ZeroShotLLM: No training required for zero-shot inference")
        self.save()  # Save configuration only

    @torch.no_grad()
    def evaluate(self): 
        """Evaluate the model using zero-shot LLM inference"""
        self.load()
        
        if self.prompt_generator is None:
            raise ValueError("Prompt generator not available. Cannot evaluate ZeroShotLLM.")
        
        # Setup model for inference
        self.model.fit(
            data=None,  # No training data needed
            column_names=self.column_names,
            llm_call_func=llm_call,
            prompt_generator=self.prompt_generator
        )
        
        # Get anomaly scores using zero-shot inference
        ascs = self.model.decision_function(
            data=self.X_test, 
            column_names=self.column_names
        )
        
        # Calculate metrics
        metrics = get_summary_metrics(y_true=self.y_test, y_pred=ascs)
        
        # Save detailed results if available
        if hasattr(self.model, 'last_results') and self.model.last_results:
            self._save_detailed_results(self.model.last_results)
        
        return metrics

    def _save_detailed_results(self, results):
        """Save detailed LLM results for analysis (original detection_retry.py format)"""
        import json
        import pandas as pd
        
        if not results:
            print("Error: No results to save")
            return
            
        # Convert results to DataFrame and set record_id as index
        results_df = pd.DataFrame(results).set_index('record_id')
        
        # Get original test data for joining with results (not normalized)
        import numpy as np
        test_data = pd.DataFrame(self.X_test_original, columns=self.column_names)
        test_data['label'] = self.y_test
        test_data.index.name = 'record_id'
        
        # Rename label to true_label to match original format
        test_data_renamed = test_data.rename(columns={'label': 'true_label'})
        combined_df = results_df.join(test_data_renamed, how='left')
        
        # Order columns like original: fixed_cols + feature_cols + reasoning_cols
        fixed_cols = ['true_label', 'prediction', 'anomaly_score', 'confidence']
        feature_cols = [col for col in test_data.columns if col != 'label']
        reasoning_cols = ['reasoning', 'key_features']
        
        final_ordered_cols = fixed_cols + feature_cols + reasoning_cols
        final_columns = [col for col in final_ordered_cols if col in combined_df.columns]
        combined_df = combined_df[final_columns]
        
        # Save CSV with record_id as index (original format)
        csv_path = os.path.join(os.path.dirname(self.ckpt_path), "detection_results.csv")
        combined_df.to_csv(csv_path, index=True, index_label='record_id')
        
        # Prepare detailed results with first batch prompt
        detailed_results = {
            "detection_results": results,
            "metadata": {
                "model_name": self.model.model_name,
                "batch_size": self.model.batch_size,
                "prompt_type": self.model.prompt_type,
                "max_retry": self.model.max_retry,
                "total_test_samples": len(results)
            }
        }
        
        # Save JSON with additional information
        results_path = os.path.join(os.path.dirname(self.ckpt_path), "detailed_results.json")
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"Results saved to {csv_path} and {results_path}")
        if detailed_results["first_batch_prompt"]:
            print(f"First batch prompt included in detailed_results.json")

    def save(self):
        """Save model configuration"""
        path = os.path.join(self.ckpt_path, "pytorch_model.safetensors")
        self.model.save_state_dict(path)

    def load(self):
        """Load model configuration"""
        path = os.path.join(self.ckpt_path, "pytorch_model.safetensors")
        if os.path.exists(os.path.dirname(path)):
            self.model.load_from_state_dict(path)
        else:
            print("No saved model found, using default configuration")