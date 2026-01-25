"""
Data Ingestion Module for Medical Diagnosis System
Handles loading, validating, preprocessing, and preparing datasets for training
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter
import argparse
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


class DataIngester:
    """Handles ingestion and preprocessing of medical diagnosis datasets"""
    
    def __init__(self, datasets_dir: str = "datasets", output_dir: str = "datasets"):
        """
        Initialize the data ingester
        
        Args:
            datasets_dir: Directory containing input datasets
            output_dir: Directory for processed output datasets
        """
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Valid urgency levels
        self.valid_urgency_levels = ["CRITICAL", "HIGH", "MODERATE", "LOW", "ROUTINE"]
        
    def load_dataset(self, file_path: str) -> List[Dict]:
        """
        Load a dataset from a JSON file
        
        Args:
            file_path: Path to the JSON dataset file
            
        Returns:
            List of dataset entries
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError(f"Dataset must be a list of entries, got {type(data)}")
            
            print(f"Loaded {len(data)} entries from {file_path}")
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            raise Exception(f"Error loading {file_path}: {e}")
    
    def validate_entry(self, entry: Dict, entry_index: int = None) -> Tuple[bool, List[str]]:
        """
        Validate a single dataset entry
        
        Args:
            entry: Dataset entry to validate
            entry_index: Index of entry (for error messages)
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        index_str = f" (entry {entry_index})" if entry_index is not None else ""
        
        # Check required fields
        required_fields = ["symptoms", "urgency", "causes", "risks"]
        for field in required_fields:
            if field not in entry:
                errors.append(f"Missing required field '{field}'{index_str}")
        
        # Validate symptoms
        if "symptoms" in entry:
            if not isinstance(entry["symptoms"], str):
                errors.append(f"'symptoms' must be a string{index_str}")
            elif len(entry["symptoms"].strip()) == 0:
                errors.append(f"'symptoms' cannot be empty{index_str}")
        
        # Validate urgency
        if "urgency" in entry:
            if not isinstance(entry["urgency"], str):
                errors.append(f"'urgency' must be a string{index_str}")
            elif entry["urgency"] not in self.valid_urgency_levels:
                errors.append(f"'urgency' must be one of {self.valid_urgency_levels}, got '{entry['urgency']}'{index_str}")
        
        # Validate causes
        if "causes" in entry:
            if not isinstance(entry["causes"], list):
                errors.append(f"'causes' must be a list{index_str}")
            elif len(entry["causes"]) == 0:
                errors.append(f"'causes' cannot be empty{index_str}")
            else:
                for i, cause in enumerate(entry["causes"]):
                    if not isinstance(cause, str):
                        errors.append(f"'causes[{i}]' must be a string{index_str}")
                    elif len(cause.strip()) == 0:
                        errors.append(f"'causes[{i}]' cannot be empty{index_str}")
        
        # Validate risks
        if "risks" in entry:
            if not isinstance(entry["risks"], list):
                errors.append(f"'risks' must be a list{index_str}")
            else:
                for i, risk in enumerate(entry["risks"]):
                    if not isinstance(risk, str):
                        errors.append(f"'risks[{i}]' must be a string{index_str}")
                    elif len(risk.strip()) == 0:
                        errors.append(f"'risks[{i}]' cannot be empty{index_str}")
        
        return len(errors) == 0, errors
    
    def validate_dataset(self, dataset: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Validate an entire dataset and return valid entries
        
        Args:
            dataset: List of dataset entries
            
        Returns:
            Tuple of (valid_entries, all_errors)
        """
        valid_entries = []
        all_errors = []
        
        for i, entry in enumerate(dataset):
            is_valid, errors = self.validate_entry(entry, i)
            if is_valid:
                valid_entries.append(entry)
            else:
                all_errors.extend(errors)
        
        return valid_entries, all_errors
    
    def clean_entry(self, entry: Dict) -> Dict:
        """
        Clean and normalize a dataset entry
        
        Args:
            entry: Dataset entry to clean
            
        Returns:
            Cleaned entry
        """
        cleaned = {}
        
        # Clean symptoms - strip whitespace, normalize
        if "symptoms" in entry:
            cleaned["symptoms"] = " ".join(entry["symptoms"].split())
        
        # Normalize urgency - uppercase
        if "urgency" in entry:
            cleaned["urgency"] = entry["urgency"].upper().strip()
        
        # Clean causes - strip whitespace, remove duplicates
        if "causes" in entry:
            causes = [cause.strip() for cause in entry["causes"] if cause.strip()]
            # Remove duplicates while preserving order
            seen = set()
            cleaned["causes"] = [cause for cause in causes if cause not in seen and not seen.add(cause)]
        
        # Clean risks - strip whitespace, remove duplicates
        if "risks" in entry:
            risks = [risk.strip() for risk in entry["risks"] if risk.strip()]
            seen = set()
            cleaned["risks"] = [risk for risk in risks if risk not in seen and not seen.add(risk)]
        
        return cleaned
    
    def preprocess_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """
        Preprocess an entire dataset
        
        Args:
            dataset: List of dataset entries
            
        Returns:
            Preprocessed dataset
        """
        # Validate first
        valid_entries, errors = self.validate_dataset(dataset)
        
        if errors:
            print(f"\nValidation warnings ({len(errors)} errors found):")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
            print(f"\nUsing {len(valid_entries)} valid entries out of {len(dataset)} total")
        
        # Clean valid entries
        cleaned_entries = [self.clean_entry(entry) for entry in valid_entries]
        
        return cleaned_entries
    
    def merge_datasets(self, datasets: List[List[Dict]]) -> List[Dict]:
        """
        Merge multiple datasets into one
        
        Args:
            datasets: List of datasets to merge
            
        Returns:
            Merged dataset
        """
        merged = []
        seen_symptoms = set()
        
        for dataset in datasets:
            for entry in dataset:
                # Use symptoms as a key to avoid exact duplicates
                symptoms_key = entry.get("symptoms", "").lower().strip()
                if symptoms_key and symptoms_key not in seen_symptoms:
                    seen_symptoms.add(symptoms_key)
                    merged.append(entry)
        
        return merged
    
    def get_dataset_statistics(self, dataset: List[Dict]) -> Dict:
        """
        Get statistics about a dataset
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary of statistics
        """
        if not dataset:
            return {"total_entries": 0}
        
        urgency_counts = Counter(entry.get("urgency", "UNKNOWN") for entry in dataset)
        total_causes = sum(len(entry.get("causes", [])) for entry in dataset)
        total_risks = sum(len(entry.get("risks", [])) for entry in dataset)
        
        # Count unique causes
        all_causes = []
        for entry in dataset:
            all_causes.extend(entry.get("causes", []))
        unique_causes = len(set(all_causes))
        
        # Average symptoms length
        avg_symptoms_length = sum(len(entry.get("symptoms", "").split()) for entry in dataset) / len(dataset)
        
        return {
            "total_entries": len(dataset),
            "urgency_distribution": dict(urgency_counts),
            "total_causes": total_causes,
            "unique_causes": unique_causes,
            "total_risks": total_risks,
            "average_symptoms_length": round(avg_symptoms_length, 2)
        }
    
    def save_dataset(self, dataset: List[Dict], output_path: str, pretty: bool = True):
        """
        Save a dataset to a JSON file
        
        Args:
            dataset: Dataset to save
            output_path: Path to save the dataset
            pretty: Whether to format JSON with indentation
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            else:
                json.dump(dataset, f, ensure_ascii=False)
        
        print(f"Saved {len(dataset)} entries to {output_path}")
    
    def ingest_all_datasets(self, output_file: str = "merged_dataset.json") -> List[Dict]:
        """
        Ingest all datasets from the datasets directory
        
        Args:
            output_file: Name of the output merged dataset file
            
        Returns:
            Merged and preprocessed dataset
        """
        print("=" * 70)
        print("Data Ingestion Process")
        print("=" * 70)
        
        # Find all JSON files in datasets directory
        if not self.datasets_dir.exists():
            print(f"Datasets directory not found: {self.datasets_dir}")
            return []
        
        json_files = list(self.datasets_dir.glob("*.json"))
        
        if not json_files:
            print(f"No JSON files found in {self.datasets_dir}")
            return []
        
        print(f"\nFound {len(json_files)} dataset file(s):")
        for file in json_files:
            print(f"  - {file.name}")
        
        # Load all datasets
        all_datasets = []
        for json_file in json_files:
            try:
                dataset = self.load_dataset(str(json_file))
                preprocessed = self.preprocess_dataset(dataset)
                all_datasets.append(preprocessed)
                
                # Show statistics
                stats = self.get_dataset_statistics(preprocessed)
                print(f"\n  Statistics for {json_file.name}:")
                print(f"    Total entries: {stats['total_entries']}")
                print(f"    Urgency distribution: {stats['urgency_distribution']}")
            except Exception as e:
                print(f"  Error processing {json_file.name}: {e}")
        
        # Merge all datasets
        print("\n" + "-" * 70)
        print("Merging datasets...")
        merged_dataset = self.merge_datasets(all_datasets)
        
        # Final statistics
        final_stats = self.get_dataset_statistics(merged_dataset)
        print("\n" + "=" * 70)
        print("Final Merged Dataset Statistics:")
        print("=" * 70)
        print(f"Total entries: {final_stats['total_entries']}")
        print(f"Urgency distribution:")
        for urgency, count in sorted(final_stats['urgency_distribution'].items()):
            print(f"  {urgency}: {count}")
        print(f"Unique causes: {final_stats['unique_causes']}")
        print(f"Average symptoms length: {final_stats['average_symptoms_length']} words")
        
        # Save merged dataset
        output_path = self.output_dir / output_file
        self.save_dataset(merged_dataset, str(output_path))
        
        return merged_dataset


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Ingest and preprocess medical diagnosis datasets")
    parser.add_argument("--input-dir", type=str, default="datasets",
                       help="Directory containing input datasets (default: datasets)")
    parser.add_argument("--output-dir", type=str, default="datasets",
                       help="Directory for output datasets (default: datasets)")
    parser.add_argument("--output-file", type=str, default="merged_dataset.json",
                       help="Output filename for merged dataset (default: merged_dataset.json)")
    parser.add_argument("--file", type=str, help="Process a single file instead of all files")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate, don't save processed data")
    
    args = parser.parse_args()
    
    ingester = DataIngester(
        datasets_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    if args.file:
        # Process single file
        print(f"Processing single file: {args.file}")
        dataset = ingester.load_dataset(args.file)
        preprocessed = ingester.preprocess_dataset(dataset)
        
        stats = ingester.get_dataset_statistics(preprocessed)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        if not args.validate_only:
            output_path = Path(args.file).stem + "_processed.json"
            ingester.save_dataset(preprocessed, output_path)
    else:
        # Process all datasets
        merged = ingester.ingest_all_datasets(args.output_file)
        
        if args.validate_only:
            print("\nValidation complete. No files saved (--validate-only flag set).")


if __name__ == "__main__":
    main()


def build_vector_store(self, dataset: List[Dict], vector_dir: str = "vector_store"):
    """
    Convert cleaned dataset into embeddings and store in FAISS
    """
    texts = []

    for entry in dataset:
        text = (
            f"Symptoms: {entry.get('symptoms')}\n"
            f"Urgency: {entry.get('urgency')}\n"
            f"Causes: {', '.join(entry.get('causes', []))}\n"
            f"Risks: {', '.join(entry.get('risks', []))}"
        )
        texts.append(text)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = FAISS.from_texts(texts, embeddings)
    db.save_local(vector_dir)

    print(f"Vector store saved to {vector_dir}")


self.build_vector_store(merged_dataset)
