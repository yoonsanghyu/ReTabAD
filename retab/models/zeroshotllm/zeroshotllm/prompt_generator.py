import os
import json
import string
import pandas as pd
from typing import Dict, Any


class ADPromptGenerator:
    def __init__(self, dataset_info: Dict[str, Any]):
        self.metadata = dataset_info['metadata']
        self.stats = dataset_info['stats']
        self.data = dataset_info['train_data']
        self.dataset_name = self.metadata['dataset_name']
        self.anonymized_mapping = None 
        

    @staticmethod
    def get_excel_style_column_names(n: int):
        """Return ['AA', 'AB', 'AC', ..., ] of length n"""
        from itertools import product
        import string
        alphabet = string.ascii_uppercase
        names = [''.join(pair) for pair in product(alphabet, repeat=2)]
        return names[:n]

    def _create_anonymization_mapping(self, columns):
        """Create anonymization mapping for columns"""
        anon_names = self.get_excel_style_column_names(len(columns))
        return dict(zip(columns, anon_names))

    def _generate_dataset_description_section(self) -> str:
        """Generate dataset description section"""
        description = self.metadata['description']
        label_description = self.metadata['label_description']
        
        section = f"## Dataset: {self.dataset_name}\n"
        section += f"{description}\n\n"
        section += f"## Target Label: {label_description}\n\n"
        
        return section

    def _generate_column_description_section(self, use_anonymized: bool = False) -> str:
        """Generate column names and descriptions section"""
        descriptions = {col['name']: col['description'] for col in self.metadata['columns']}
        
        section = "## Features:\n"
        section += "### Numerical features:\n"
        numerical_columns = [
            col['name'] for col in self.metadata['columns']
            if col['logical_type'] == 'numerical'
        ]
        
        for col in numerical_columns:
            display_name = self.anonymized_mapping.get(col, col) if use_anonymized and self.anonymized_mapping else col
            if use_anonymized:
                desc = f"Numerical feature {display_name}"
            else:
                desc = descriptions.get(col, "No description available.")
            section += f"- **{display_name}**: {desc}\n"
        
        section += "### Categorical features:\n"
        categorical_columns = [
            col['name'] for col in self.metadata['columns']
            if col['logical_type'] in ['categorical', 'binary']
        ]
        
        for col in categorical_columns:
            display_name = self.anonymized_mapping.get(col, col) if use_anonymized and self.anonymized_mapping else col
            if use_anonymized:
                desc = f"Categorical feature {display_name}"
            else:
                desc = descriptions.get(col, "No description available.")
            section += f"- **{display_name}**: {desc}\n"
        
        return section

    def _generate_statistical_context_section(self, use_anonymized: bool = False) -> str:
        """Generate statistical context section with normal ranges and values"""
        if not self.stats or not self.metadata:
            return ""

        sample_count = next((s.get('count', 0) for s in self.stats.values()), 0)
        
        section = f"## Statistical Context (from {sample_count} normal cases)\n"
        section += "### Numerical features - Normal Ranges:\n"
        numerical_columns = [
            col['name'] for col in self.metadata['columns']
            if col['logical_type'] == 'numerical'
        ]
        
        for col in numerical_columns:
            if col in self.stats:
                stats = self.stats[col]
                display_name = self.anonymized_mapping.get(col, col) if use_anonymized and self.anonymized_mapping else col
                if stats['q5'] is not None and stats['q95'] is not None:
                    section += f"- **{display_name}**: Normal Range (5th-95th percentile): {stats['q5']:.3f} – {stats['q95']:.3f}\n"
        
        section += "### Categorical features - Normal Values:\n"
        if self.data is not None:
            categorical_columns = [
                col['name'] for col in self.metadata['columns']
                if col['logical_type'] in ['categorical', 'binary']
            ]
            for col in categorical_columns:
                if col in self.data.columns:
                    unique_vals = sorted(self.data[col].dropna().unique().tolist())
                    unique_vals_str = ', '.join(map(str, unique_vals)) if len(unique_vals) <= 20 else f"{len(unique_vals)} categories: {', '.join(map(str, unique_vals[:10]))}..."
                    display_name = self.anonymized_mapping.get(col, col) if use_anonymized and self.anonymized_mapping else col
                    section += f"- **{display_name}**: Normal Values: {unique_vals_str}\n"
        
        return section

    def prepare_data_string(self, data_df: pd.DataFrame, include_header: bool = True, 
                          max_samples: int = None, format_style: str = "csv", prompt_type: str = "D") -> str:
        """
        Convert DataFrame to string format for prompt.
        """
        if max_samples and len(data_df) > max_samples:
            data_df = data_df.head(max_samples)
            
        # Remove label column if present
        label_col = self.metadata['label_column']
        if label_col in data_df.columns:
            data_df = data_df.drop(columns=[label_col])

        # anonymized
        if prompt_type in ["0", "A"]:
            self.anonymized_mapping = self._create_anonymization_mapping(data_df.columns)
            data_df = data_df.rename(columns=self.anonymized_mapping)
        else:
            self.anonymized_mapping = None  # 초기화

        if format_style == "record":
            # Format as "Record {idx}:\ncol1=val1, col2=val2"
            batch_records = []
            for idx, row in data_df.iterrows():
                row_string = ", ".join([
                    f"{col}={val:.2f}" if isinstance(val, float) else f"{col}={val}" 
                    for col, val in row.items()
                ])
                batch_records.append(f"Record {idx}:\n{row_string}")
            return "\n".join(batch_records)
        else:
            # Default CSV format
            return data_df.to_csv(index=False, header=include_header)

    def build_system_prompt(self, prompt_type: str = "D") -> str:
        """Build system prompt based on prompt type."""
        use_anonymized = prompt_type in ["0", "A"]
        
        
        if prompt_type == "0":
            # Type 0: Batch-level anomaly detection
            system_prompt = "Analyze the following batch of data samples and identify which ones appear to be anomalies.\n\n"
            system_prompt += "## Analysis Guidelines:\n"
            system_prompt += "- Look for samples that differ significantly from the majority pattern.\n"
            system_prompt += "- Consider statistical outliers in numerical features.\n"
            system_prompt += "- **CRITICAL**: Don't just compare categorical values - analyze categorical-numerical patterns.\n"
            system_prompt += "- Within each categorical group in the batch, identify numerical patterns that deviate from the group norm.\n"
            system_prompt += "- A sample can be anomalous even if its categorical values appear frequently in the batch.\n"
            system_prompt += "- Use relative comparison within the batch to identify anomalies.\n\n"
            
        elif prompt_type == "A":
            # Type A: Statistical context only (anonymized)
            system_prompt = "Analyze the data for anomaly detection.\n\n"
            system_prompt += self._generate_statistical_context_section(use_anonymized=True)

        elif prompt_type == "B":
            # Type B: Domain descriptions only
            system_prompt = f"Analyze the {self.dataset_name} data for anomaly detection.\n\n"
            system_prompt += self._generate_column_description_section(use_anonymized=False)
            system_prompt += "\n" + self._generate_statistical_context_section(use_anonymized=False)
            system_prompt += "\n## Analysis Guidelines:\n"
            system_prompt += "- Use feature descriptions to understand data context.\n"
            system_prompt += "- Look for values that seem unusual given the feature meanings.\n"
            system_prompt += "- **CRITICAL**: Consider relationships between categorical and numerical features.\n"
            system_prompt += "- Check if categorical-numerical combinations make logical sense based on feature descriptions.\n"
            system_prompt += "- Even familiar categorical values can be anomalous if paired with inconsistent numerical values.\n\n"
            
        elif prompt_type == "C":
            # Type C: Dataset + Column descriptions
            system_prompt = f"Analyze the {self.dataset_name} data for anomaly detection.\n\n"
            system_prompt += self._generate_dataset_description_section()
            system_prompt += self._generate_column_description_section(use_anonymized=False)
            system_prompt += "\n## Analysis Guidelines:\n"
            system_prompt += "- Use dataset context and feature descriptions to guide analysis.\n"
            system_prompt += "- Consider domain-specific patterns and relationships.\n"
            system_prompt += "- **CRITICAL**: Evaluate categorical-numerical feature combinations for logical consistency.\n"
            system_prompt += "- Look for anomalies where categorical values are normal but numerical patterns don't align with domain expectations.\n"
            system_prompt += "- Use domain knowledge to identify implausible feature combinations.\n\n"
            
        else:  # Type D: Full context (default)
            # Type D: Dataset + Column + Statistical context
            system_prompt = f"You are a senior {self.dataset_name} expert analyzing data for anomaly detection.\n\n"
            system_prompt += self._generate_dataset_description_section()
            system_prompt += self._generate_column_description_section(use_anonymized=False)
            system_prompt += "\n" + self._generate_statistical_context_section(use_anonymized=False)
            system_prompt += "\n## Analysis Guidelines:\n"

            system_prompt += "### Numerical Analysis:\n"
            system_prompt += "- Compare each numerical feature to the normal distribution baselines above.\n"
            system_prompt += "- Identify values beyond 95th percentile OR below 5th percentile (likely abnormal).\n"
            system_prompt += "- Consider cumulative effect of multiple borderline values.\n"
            system_prompt += "- **Individual normal values do not guarantee normal overall patterns - evaluate holistic combinations.**\n\n"

            system_prompt += "### Categorical Pattern Analysis:\n"
            system_prompt += "- **Don't just check if categorical values are 'in-distribution**'.\n"
            system_prompt += "- Look beyond individual categorical validity - assess contextual appropriateness given other features.\n"
            system_prompt += "- Identify contradictory categorical combinations or misalignment with numerical severity indicators.\n\n"

            system_prompt += "### Domain Analysis:\n"
            system_prompt += "- Use dataset context and feature descriptions to guide analysis.\n"
            system_prompt += "- Look for meaningful patterns across multiple features.\n"
            system_prompt += "- Consider how features relate based on domain knowledge and semantic context.\n"
            system_prompt += "- Evaluate logical consistency between categorical and numerical features.\n"
            system_prompt += "- Assess whether feature combinations make domain-specific logical sense.\n"
            system_prompt += "- **Leverage target label definition**: Use the provided normal/abnormal label definition to understand what constitutes anomalous behavior in this specific domain.\n\n"

        # Common output requirements for all types
        system_prompt += "## CRITICAL OUTPUT REQUIREMENTS:\n"
        system_prompt += "- RETURN ONLY VALID JSON ARRAY - NO OTHER TEXT ALLOWED\n"
        system_prompt += "- NO markdown, NO explanations, NO headers, NO code blocks\n"
        system_prompt += "- START with '[' and END with ']'\n"
        system_prompt += "- Use the actual Record IDs (not row numbers) in your JSON response.\n"
        system_prompt += "- For multiple records: Return JSON array with one object per record\n"
        system_prompt += "- NEVER include batch numbers, analysis headers, or separators\n"

        return system_prompt

    def build_full_prompt(self, data_df: pd.DataFrame = None, data_string: str = None, 
                         prompt_type: str = "D", format_style: str = "record") -> str:
        """Build full prompt with specified type."""
        if data_string is None:
            if data_df is None:
                raise ValueError("Either data_df or data_string must be provided")
            # For individual detection, use smaller sample
            data_string = self.prepare_data_string(data_df, max_samples=len(data_df), 
                                                    format_style=format_style, prompt_type=prompt_type)
    
        system_prompt = self.build_system_prompt(prompt_type)
        
        full_prompt = f"{system_prompt}\n\n"
        full_prompt += "Analyze the following data based on the guidelines provided above:\n"
        full_prompt += f"{data_string}"
        
        full_prompt += "\n\nReturn valid JSON array:\n"
        full_prompt += '[{"record_id":"0","prediction":0,"anomaly_score":0.0,"confidence":0.0,"reasoning":"Analysis result","key_features":["feature1","feature2"]},{"record_id":"1","prediction":1,"anomaly_score":0.8,"confidence":0.9,"reasoning":"Anomaly detected","key_features":["feature3","feature4"]}]'
        
        return full_prompt

    def get_available_types(self) -> Dict[str, str]:
        """Return description of available prompt types"""
        return {
            "0": "Batch-level anomaly detection (no context, relative comparison)",
            "A": "Statistical context only (anonymized features)",
            "B": "Column descriptions only", 
            "C": "Dataset description + Column descriptions",
            "D": "Full context (Dataset + Column descriptions + Statistical context)"
        }
        
    def get_anonymization_mapping(self) -> Dict[str, str]:
        """Return current anonymization mapping"""
        return self.anonymized_mapping if self.anonymized_mapping else {}