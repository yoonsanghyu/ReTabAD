"""
Serialized tabular dataset implementation using HuggingFace datasets.
"""

import os
import typing as tp
import random
import pickle as pkl
from tqdm import tqdm
from datasets import Dataset as HuggingFaceDataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding


MAX_COL_LENGTH = 128
class SerializedTabularDataset(HuggingFaceDataset):
    """
    AnoLLM Dataset for serialized tabular data.
    
    Extends HuggingFace Dataset to handle tokenized tabular data
    with support for text serialization and tokenization.
    
    Attributes:
        tokenizer: Tokenizer from HuggingFace
    """

    def set_tokenizer(self, tokenizer):
        """
        Set the Tokenizer.

        Args:
            tokenizer: Tokenizer from HuggingFace
        """
        self.tokenizer = tokenizer

    def set_anomaly_label(self, labels):
        """
        Set anomaly labels for the dataset.
        
        Args:
            labels: List of anomaly labels
        """
        assert len(labels) == len(self._data)
        self.anomaly_labels = labels

    def set_textual_columns(self, columns: tp.Optional[tp.List[str]] = None):
        """
        Set which columns should be treated as textual.
        
        Args:
            columns: List of column names to treat as textual
        """
        if columns is None:
            columns = self.get_column_names()

        col_list = self.get_column_names()
        for col in columns:
            if col not in col_list:
                raise ValueError(f"Column {col} not in the dataset.")
        self.textual_columns = columns

    def get_n_columns(self):
        """Get the number of columns in the dataset."""
        row = self._data.fast_slice(0, 1)
        return row.num_columns

    def get_column_names(self):
        """Get the column names of the dataset."""
        row = self._data.fast_slice(0, 1)
        return row.column_names

    def shuffle_column_order(self):
        """
        Shuffle the column order for evaluation.
        
        Used in evaluation where the order of columns is shuffled 
        and then fixed for all data.
        """
        row = self._data.fast_slice(0, 1)
        self.shuffle_idx = list(range(row.num_columns))
        random.shuffle(self.shuffle_idx)

    def fix_column_order(self):
        """
        Set the column order to be default column order.
        
        Do not shuffle the columns.
        """
        row = self._data.fast_slice(0, 1)
        self.shuffle_idx = list(range(row.num_columns))

    def prepare(
        self,
        is_eval: bool = True,
        max_length_dict: tp.Optional[tp.Dict[str, int]] = {},
        data_path=None,
    ):
        """
        Preprocess the data by tokenizing each column and truncating to max_length.
        
        Args:
            is_eval: Whether this is for evaluation
            max_length_dict: Maximum length for each column
            data_path: Path to save/load processed data
        """
        self.is_eval = is_eval
        n_col = self.get_n_columns()
        column_names = self.get_column_names()
        self.processed_data = []
        self.tokenized_feature_names = []
        bos_token_id = self.tokenizer.bos_token_id

        for col_idx in range(n_col):
            feature_names = " " + column_names[col_idx] + " "
            # Use __call__ method instead of separate tokenize calls
            tokenized_feature_names = self.tokenizer(
                feature_names, add_special_tokens=False
            )
            tokenized_is = self.tokenizer("is ", add_special_tokens=False)

            self.tokenized_feature_names.append(
                tokenized_feature_names["input_ids"] + tokenized_is["input_ids"]
            )

        if data_path is not None and os.path.exists(data_path):
            self.processed_data = pkl.load(open(data_path, "rb"))
        else:
            for key in tqdm(range(len(self._data))):
                row = self._data.fast_slice(key, 1)
                tokenized_texts = []
                for col_idx in range(n_col):
                    feature_values = str(row.columns[col_idx].to_pylist()[0]).strip()
                    if len(feature_values) == 0:
                        feature_values = "None"
                    # Use __call__ method with add_special_tokens=False to avoid BOS token issues
                    data = self.tokenizer(feature_values, add_special_tokens=False)
                    tokenized_texts.append(data["input_ids"])

                    if len(data["input_ids"]) == 0:
                        print(
                            "Warning: tokenized text is empty.",
                            column_names[col_idx],
                            len(feature_values),
                            feature_values,
                        )
                self.processed_data.append(tokenized_texts)

            # truncate the columns that are too long
            for col_idx in range(n_col):
                name = column_names[col_idx]
                if name not in max_length_dict:
                    max_length = MAX_COL_LENGTH
                else:
                    max_length = max_length_dict[name]
                assert isinstance(max_length, int)

                for data_idx in range(len(self.processed_data)):
                    length = len(self.processed_data[data_idx][col_idx]) + len(
                        self.tokenized_feature_names[col_idx]
                    )
                    if length >= max_length:
                        self.processed_data[data_idx][col_idx] = self.processed_data[
                            data_idx
                        ][col_idx][
                            : max_length - len(self.tokenized_feature_names[col_idx])
                        ]
            if data_path is not None:
                pkl.dump(self.processed_data, open(data_path, "wb"))
        print("Preprocessing done.")

    def _getitem(
        self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs
    ) -> tp.Union[tp.Dict, tp.List]:
        """
        Get one instance of the tabular data, permuted, converted to text and tokenized.
        
        Args:
            key: Index or slice to retrieve
            decoded: Whether to return decoded data
            
        Returns:
            Tokenized text data
        """
        row = self._data.fast_slice(key, 1)

        # get shuffle_idx
        if "shuffle_idx" in self.__dict__:
            shuffle_idx = self.shuffle_idx
        else:
            shuffle_idx = list(range(row.num_columns))
            random.shuffle(shuffle_idx)

        # get tokenized text
        comma_id = self.tokenizer.convert_tokens_to_ids(",")
        eos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        bos_token_id = self.tokenizer.bos_token_id
        if self.is_eval:
            tokenized_text = {
                "input_ids": [],
                "attention_mask": [],
                "feature_value_start": [],
                "feature_value_end": [],
                "col_indices": shuffle_idx,
            }
        else:
            tokenized_text = {"input_ids": [], "attention_mask": []}
        if bos_token_id:
            tokenized_text["input_ids"] = [bos_token_id]

        if hasattr(self, "processed_data"):
            start_idx = 0
            for idx, col_idx in enumerate(shuffle_idx):
                tokenized_feature_names = self.tokenized_feature_names[col_idx]
                tokenized_feature_values = self.processed_data[key][col_idx]
                tokenized_col = tokenized_feature_names + tokenized_feature_values
                if idx == len(shuffle_idx) - 1:
                    tokenized_text["input_ids"] += tokenized_col + [eos_id]
                else:
                    tokenized_text["input_ids"] += tokenized_col + [comma_id]
                if self.is_eval:
                    tokenized_text["feature_value_start"].append(
                        start_idx + len(tokenized_feature_names) - 1
                    )
                    tokenized_text["feature_value_end"].append(
                        start_idx + len(tokenized_col)
                    )
                start_idx += len(tokenized_col) + 1
        else:
            raise ValueError(
                "processed_data is not found. Please run prepare function first."
            )
        tokenized_text["attention_mask"] += [1] * len(tokenized_text["input_ids"])
        return tokenized_text

    def get_item_test(self, key):
        """
        Get test item with shuffled text representation.
        
        Args:
            key: Index to retrieve
            
        Returns:
            tuple: (shuffled_text, tokenized_text)
        """
        row = self._data.fast_slice(key, 1)
        shuffle_idx = list(range(row.num_columns))
        random.shuffle(shuffle_idx)

        shuffled_text = ",".join(
            [
                " %s is %s "
                % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip())
                for i in shuffle_idx
            ]
        )
        # Use __call__ method directly instead of separate tokenize + pad
        tokenized_text = self.tokenizer(
            shuffled_text, padding=True, return_tensors="pt"
        )

        return shuffled_text, tokenized_text

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        """
        Get multiple items from the dataset.
        
        Args:
            keys: Keys to retrieve (int, slice, str, or list)
            
        Returns:
            List of items or single item
        """
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)


@dataclass
class DataCollator(DataCollatorWithPadding):
    """
    Data collator that extends DataCollatorWithPadding.
    
    Handles batching and padding of tokenized tabular data,
    and prepares labels for language modeling.
    """
    
    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        """
        Collate features into a batch.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            dict: Batched features with labels
        """
        # Use the parent class's __call__ method which handles fast tokenizers properly
        batch = super().__call__(features)
        batch["labels"] = batch["input_ids"].clone()
        return batch