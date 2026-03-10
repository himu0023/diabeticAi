"""
This module prepares CGM diabetes data for machine learning.

Responsibilities:

1. Handle missing values 
2. Smooth CGM sensor noise 
3. Remove physiologically impossible values
4. Generate normalized features
5. Prepare model-ready datafrome 
"""

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler