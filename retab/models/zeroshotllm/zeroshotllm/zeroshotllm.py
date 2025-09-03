import os
import re
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, average_precision_score
)


class ParsingWarningError(Exception):
    """경고성 파싱 문제를 재시도 트리거로 승격하기 위한 커스텀 예외"""
    pass


class ZeroShotLLM:
    """
    Zero-shot anomaly detection using LLM
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        batch_size: int = 15,
        max_retry: int = 3,
        prompt_type: str = "D"
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retry = max_retry
        self.prompt_type = prompt_type
        self.llm_call_func = None  # Will be set during training
        
    def set_llm_call_func(self, llm_call_func):
        """Set the LLM call function"""
        self.llm_call_func = llm_call_func
        
    def _coerce_key_features(self, value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [v.strip() for v in value.split(',') if v.strip()]
        raise ParsingWarningError(f"Invalid type for key_features: {type(value)}")

    def _extract_json_single_attempt(self, text: str, indices: List[int]) -> List[Dict[str, Any]]:
        """단일 시도 파싱"""
        text = text.strip()

        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        json_text = json_match.group(0) if json_match else text

        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            parsed = [parsed]

        if not isinstance(parsed, list):
            raise ParsingWarningError("Top-level JSON is not a list/dict.")

        if len(parsed) != len(indices):
            raise ParsingWarningError(
                f"Record count mismatch: expected {len(indices)}, got {len(parsed)}"
            )

        results: List[Dict[str, Any]] = []

        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ParsingWarningError(f"Item {i} is not a JSON object.")

            try:
                prediction = int(item["prediction"])
                anomaly_score = float(item["anomaly_score"])
                confidence = float(item["confidence"])
            except KeyError as e:
                raise ParsingWarningError(f"Missing key in item {i}: {e}")
            except (TypeError, ValueError) as e:
                raise ParsingWarningError(f"Type coercion failed in item {i}: {e}")

            reasoning = str(item.get("reasoning", ""))
            key_features = self._coerce_key_features(item.get("key_features", []))

            if not (anomaly_score == anomaly_score):  # NaN check
                raise ParsingWarningError(f"anomaly_score is NaN in item {i}")
            if not (confidence == confidence):
                raise ParsingWarningError(f"confidence is NaN in item {i}")

            results.append({
                "record_id": indices[i],
                "prediction": prediction,
                "anomaly_score": anomaly_score,
                "confidence": confidence,
                "reasoning": reasoning,
                "key_features": key_features
            })

        return sorted(results, key=lambda x: x["record_id"])

    def extract_json_with_retry(
        self,
        response_text: str,
        record_indices: List[int],
        prompt: str = None
    ) -> List[Dict[str, Any]]:
        """JSON 파싱을 재시도하는 함수"""
        
        for attempt in range(self.max_retry):
            try:
                print(f"JSON 파싱 시도 {attempt + 1}/{self.max_retry}...")
                if attempt == 0:
                    current_response = response_text
                else:
                    if self.llm_call_func is None or prompt is None:
                        print("재시도 파라미터 부족. 재호출 불가. 중단합니다.")
                        break
                    print(f"LLM 재호출 중... (시도 {attempt + 1})")
                    current_response = self.llm_call_func(prompt=prompt, model=self.model_name)
                    time.sleep(1)

                results = self._extract_json_single_attempt(current_response, record_indices)
                print(f"JSON 파싱 성공! (시도 {attempt + 1})")
                return results

            except (json.JSONDecodeError, ParsingWarningError) as e:
                print(f"시도 {attempt + 1} 실패: {e}")
                if attempt == self.max_retry - 1:
                    print("모든 시도 실패.")
                    break
                else:
                    print(f"다시 시도합니다... ({attempt + 2}/{self.max_retry})")
                    continue
            except Exception as e:
                print(f"시도 {attempt + 1} 중 예상치 못한 오류: {e}")
                if attempt == self.max_retry - 1:
                    break
                continue

        # 모든 시도 실패: 기본값으로 반환
        print(f"모든 재시도 실패. 기본값으로 결과를 생성합니다.")
        default_results = [{
            "record_id": idx,
            "prediction": 0,
            "anomaly_score": 0.5,
            "confidence": 0.5,
            "reasoning": f"JSON parsing failed after {self.max_retry} attempts",
            "key_features": []
        } for idx in record_indices]

        return sorted(default_results, key=lambda x: x["record_id"])

    def fit(self, data, column_names, llm_call_func, prompt_generator):
        """
        Train the model (in zero-shot case, just set up parameters)
        
        Args:
            data: Training data (not used in zero-shot)
            column_names: Column names
            llm_call_func: Function to call LLM
            prompt_generator: Prompt generator object
        """
        self.column_names = column_names
        self.llm_call_func = llm_call_func
        self.prompt_generator = prompt_generator
        print("ZeroShotLLM model initialized for zero-shot inference")

    def decision_function(self, data, column_names):
        """
        Predict anomaly scores using zero-shot LLM inference
        
        Args:
            data: Test data (serialized format)
            column_names: Column names
            
        Returns:
            np.ndarray: Anomaly scores
        """
        if self.llm_call_func is None:
            raise ValueError("LLM call function not set. Call fit() first.")
            
        # Convert data to DataFrame for processing
        test_df = pd.DataFrame(data, columns=column_names)
        test_df.index.name = 'record_id'
        
        all_results = []
        first_batch_prompt = None  # Store first batch prompt
        
        print(f"Starting zero-shot anomaly detection with batch size {self.batch_size}...")
        
        for i in range(0, len(test_df), self.batch_size):
            batch_df = test_df.iloc[i:i + self.batch_size]
            record_indices = batch_df.index.tolist()

            # Generate prompt for this batch
            full_prompt = self.prompt_generator.build_full_prompt(
                data_df=batch_df, 
                prompt_type=self.prompt_type,
                format_style="record" 
            )

            batch_num = (i // self.batch_size) + 1
            
            # Print and store first batch prompt
            if batch_num == 1:
                print("=" * 80)
                print("FIRST BATCH PROMPT:")
                print("=" * 80)
                print(full_prompt)
                print("=" * 80)
                first_batch_prompt = full_prompt
            
            print(f"Analyzing Batch #{batch_num} (Records: {record_indices[0]}-{record_indices[-1]})...")
            
            try:
                # LLM 호출
                llm_response = self.llm_call_func(prompt=full_prompt, model=self.model_name)
                print(f"LLM Response preview: {llm_response[:200]}...")
                
                # JSON 파싱 with retry
                parsed_batch_results = self.extract_json_with_retry(
                    response_text=llm_response,
                    record_indices=record_indices,
                    prompt=full_prompt
                )
                
                all_results.extend(parsed_batch_results)
                print(f"✅ Batch #{batch_num} processed successfully.")

            except Exception as e:
                print(f"❌ Critical error processing Batch #{batch_num}: {e}")
                # 크리티컬 에러시에도 기본값으로 결과 생성
                default_batch_results = []
                for idx in record_indices:
                    default_batch_results.append({
                        'record_id': idx,
                        'prediction': 0,
                        'anomaly_score': 0.5,
                        'confidence': 0.5,
                        'reasoning': f"Critical error in batch processing: {str(e)}",
                        'key_features': []
                    })
                all_results.extend(default_batch_results)
        
        # Extract anomaly scores and sort by record_id
        results_df = pd.DataFrame(all_results).set_index('record_id').sort_index()
        anomaly_scores = results_df['anomaly_score'].values
        
        # Store results
        self.last_results = all_results   
             
        return anomaly_scores

    def save_state_dict(self, path: str):
        """Save model state (for compatibility)"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'max_retry': self.max_retry,
            'prompt_type': self.prompt_type
        }
        with open(path.replace('.safetensors', '_config.json'), 'w') as f:
            json.dump(state, f, indent=2)
        print(f"ZeroShotLLM configuration saved to {path}")

    def load_from_state_dict(self, path: str):
        """Load model state (for compatibility)"""
        config_path = path.replace('.safetensors', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                state = json.load(f)
            self.model_name = state.get('model_name', self.model_name)
            self.batch_size = state.get('batch_size', self.batch_size)
            self.max_retry = state.get('max_retry', self.max_retry)
            self.prompt_type = state.get('prompt_type', self.prompt_type)
            print(f"ZeroShotLLM configuration loaded from {config_path}")
        else:
            print(f"No configuration found at {config_path}, using default settings")