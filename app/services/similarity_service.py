from typing import List, Dict, Tuple
import math
from collections import defaultdict

from app.models import IRData, Document
from app.schemas import DocumentSimilarityScore, TermWeightingMethod, TermFrequencyMethod
from app.utils import tokenize_and_preprocess

