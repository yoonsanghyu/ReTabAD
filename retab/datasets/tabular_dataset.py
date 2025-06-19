
"""
Tabular dataset implementation for PyTorch.
"""

from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data with categorical and continuous features.
    
    Handles both categorical and continuous features with their respective masks.
    """
    
    def __init__(self, X_cat_data, X_cat_mask, X_cont_data, X_cont_mask, y):
        """
        Initialize the tabular dataset.
        
        Args:
            X_cat_data: Categorical feature data
            X_cat_mask: Categorical feature mask
            X_cont_data: Continuous feature data  
            X_cont_mask: Continuous feature mask
            y: Target labels
        """
        self.X_cat_data = X_cat_data
        self.X_cat_mask = X_cat_mask
        self.X_cont_data = X_cont_data
        self.X_cont_mask = X_cont_mask
        self.y = y

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.y)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            dict: Dictionary containing features and labels
        """
        if self.X_cat_data:
            return {
                "cat_features": self.X_cat_data[idx],
                "cat_mask": self.X_cat_mask[idx],
                "cont_features": self.X_cont_data[idx],
                "cont_mask": self.X_cont_mask[idx],
                "label": self.y[idx],
            }
        else:
            return {
                "cont_features": self.X_cont_data[idx],
                "cont_mask": self.X_cont_mask[idx],
                "label": self.y[idx],
            }