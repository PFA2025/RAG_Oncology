import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path setup
SCRIPT_DIR = Path(__file__).parent
HISTORY_DB = SCRIPT_DIR / 'patient_history.db'
HISTORY_CSV = SCRIPT_DIR / 'patient_history.csv'

load_dotenv()

class PatientHistoryStorage:
    def __init__(self):
        self.history_file = HISTORY_DB
        self.csv_file = HISTORY_CSV
        self._initialize_storage()

    def _initialize_storage(self):
        if not self.history_file.exists():
            with open(self.history_file, 'w') as f:
                json.dump([], f)
        
        if not self.csv_file.exists():
            pd.DataFrame(columns=[
                'timestamp', 'patient_id', 'query', 'answer', 
                'source', 'confidence', 'model', 'rag_context_used',
                'medical_context'
            ]).to_csv(self.csv_file, index=False)

    def _read_history(self) -> List[Dict[str, Any]]:
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_history(self, history: List[Dict[str, Any]]):
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def _append_to_csv(self, record: Dict[str, Any]):
        try:
            df = pd.read_csv(self.csv_file)
            new_row = pd.DataFrame([{
                'timestamp': record['timestamp'],
                'patient_id': record['patient_id'],
                'query': record['query'],
                'answer': record['answer'],
                'source': record['source'],
                'confidence': record['confidence'],
                'model': record.get('metadata', {}).get('model', ''),
                'rag_context_used': record.get('metadata', {}).get('rag_context_used', False),
                'medical_context': record.get('medical_context', '')
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.csv_file, index=False)
        except Exception as e:
            logger.error(f"Error writing to CSV: {e}")

    def store_interaction(
        self, 
        query: str, 
        answer: str, 
        source: str, 
        confidence: float,
        model: Optional[str] = None,
        rag_context_used: Optional[bool] = None,
        medical_context: Optional[str] = None,
        patient_id: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            timestamp = datetime.now().isoformat()
            metadata = {
                'model': model,
                'rag_context_used': rag_context_used
            }
            
            if additional_metadata:
                metadata.update(additional_metadata)
            
            record = {
                'timestamp': timestamp,
                'patient_id': patient_id or 'anonymous',
                'query': query,
                'answer': answer,
                'source': source,
                'confidence': float(confidence),
                'medical_context': medical_context or self._detect_medical_context(query),
                'metadata': metadata
            }
            
            history = self._read_history()
            history.append(record)
            self._write_history(history)
            self._append_to_csv(record)
            
            return record
            
        except Exception as e:
            logger.error(f"Error storing interaction: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'patient_id': patient_id or 'anonymous',
                'query': query,
                'answer': answer,
                'source': source,
                'confidence': float(confidence),
                'error': str(e)
            }

    def _detect_medical_context(self, query: str) -> str:
        query_lower = query.lower()
        if 'chemo' in query_lower:
            return 'chemotherapy'
        elif 'radiation' in query_lower:
            return 'radiation therapy'
        elif 'immunotherapy' in query_lower:
            return 'immunotherapy'
        elif 'surgery' in query_lower:
            return 'surgical treatment'
        return 'general oncology'

    def get_patient_history(self, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        history = self._read_history()
        if patient_id:
            return [rec for rec in history if rec['patient_id'] == patient_id]
        return history

    def export_to_dataframe(self) -> pd.DataFrame:
        history = self._read_history()
        return pd.DataFrame(history)