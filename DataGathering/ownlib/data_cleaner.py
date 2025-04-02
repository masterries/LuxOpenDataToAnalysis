import os
import pandas as pd
import sqlite3
import hashlib
import json
import re
from typing import Dict, List, Optional, Tuple, Any


class DataCleaner:
    """
    Class for cleaning processed housing data and storing it in a SQLite database.
    Handles data from different sources, detects duplicates, and standardizes column names.
    """
    
    def __init__(self, db_path: str = "luxembourg_housing.db"):
        """
        Initialize the cleaner with a path to the SQLite database.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.stats = {
            "files_processed": 0,
            "records_inserted": 0,
            "duplicates_skipped": 0,
            "errors": 0
        }
        
        # Define column mapping for standardization
        self.column_mapping = {
            "commune": "municipality",
            "nombre_d'offres": "num_listings",
            "nombre_d'annonces": "num_listings",
            "prix_moyen_annoncé_________en_€_courant": "avg_price",
            "prix_moyen_annoncé_______en_€_courant": "avg_price",
            "prix_moyen_annoncé_en_€_courant": "avg_price",
            "prix_moyen_annoncé": "avg_price",
            "prix_moyen_annoncé_au_m²_en_€_courant": "avg_price_sqm", 
            "loyer_moyen_annoncé_en_€_courant": "avg_price",
            "loyer_mensuel_moyen_annoncé": "avg_price",
            "loyer_annoncé_au_m²_en_€_courant": "avg_price_sqm",
            "loyer_moyen_annoncé_au_m²": "avg_price_sqm"
        }
    
    def _connect_to_db(self) -> None:
        """
        Establish a connection to the SQLite database.
        Creates tables if they don't exist.
        """
        self.conn = sqlite3.connect(self.db_path)
        
        # Create tables if they don't exist
        cursor = self.conn.cursor()
        
        # Create file tracking table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT NOT NULL UNIQUE,
            file_hash TEXT NOT NULL,
            processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create main housing data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS housing_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            municipality TEXT NOT NULL,
            num_listings INTEGER,
            avg_price REAL,
            avg_price_sqm REAL,
            year INTEGER,
            property_type TEXT,
            transaction_type TEXT,
            source TEXT,
            file_id INTEGER,
            data_hash TEXT UNIQUE,
            FOREIGN KEY (file_id) REFERENCES processed_files (id)
        )
        ''')
        
        self.conn.commit()
    
    def _close_db(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def _get_file_hash(self, filepath: str) -> str:
        """
        Calculate MD5 hash of a file.
        
        Args:
            filepath (str): Path to the file
        
        Returns:
            str: MD5 hash of the file content
        """
        with open(filepath, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    
    def _is_file_processed(self, filepath: str, file_hash: str) -> bool:
        """
        Check if a file has already been processed.
        
        Args:
            filepath (str): Path to the file
            file_hash (str): MD5 hash of the file
        
        Returns:
            bool: True if the file has already been processed
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM processed_files WHERE filepath = ? OR file_hash = ?", 
            (filepath, file_hash)
        )
        result = cursor.fetchone()
        return result is not None
    
    def _register_file(self, filepath: str, file_hash: str) -> int:
        """
        Register a file as processed in the database.
        
        Args:
            filepath (str): Path to the file
            file_hash (str): MD5 hash of the file
        
        Returns:
            int: ID of the registered file
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO processed_files (filepath, file_hash) VALUES (?, ?)",
            (filepath, file_hash)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names using the mapping dictionary.
        
        Args:
            df (pd.DataFrame): DataFrame with original column names
        
        Returns:
            pd.DataFrame: DataFrame with standardized column names
        """
        renamed_columns = {}
        
        for original_col in df.columns:
            # Clean column name by removing whitespace and lowercasing
            col_clean = original_col.strip().lower()
            
            # Try to find the cleaned column in our mapping
            if col_clean in self.column_mapping:
                renamed_columns[original_col] = self.column_mapping[col_clean]
            else:
                # Try to find a fuzzy match by checking each key partially
                matched = False
                for key, value in self.column_mapping.items():
                    # Convert to proper comparison format
                    key_compare = key.lower().replace(' ', '').replace('_', '')
                    col_compare = col_clean.replace(' ', '').replace('_', '')
                    
                    if key_compare in col_compare or col_compare in key_compare:
                        renamed_columns[original_col] = value
                        matched = True
                        break
                
                # If no match found, keep original
                if not matched:
                    renamed_columns[original_col] = original_col
        
        # Rename the columns
        df = df.rename(columns=renamed_columns)
        return df
    
    def _extract_year_from_path(self, filepath: str) -> Optional[int]:
        """
        Extract year from filepath or filename.
        
        Args:
            filepath (str): Path to the file
        
        Returns:
            Optional[int]: Year if found, None otherwise
        """
        # First check for pattern like "year_2023" in the path
        year_folder_match = re.search(r'year_(\d{4})', filepath)
        if year_folder_match:
            return int(year_folder_match.group(1))
        
        # Then check for pattern like "2023-24" or "2023" in the filename
        filename = os.path.basename(filepath)
        year_match = re.search(r'(\d{4})(?:-\d{2})?', filename)
        if year_match:
            return int(year_match.group(1))
        
        return None
    
    def _extract_property_type(self, filepath: str) -> str:
        """
        Extract property type from filepath.
        
        Args:
            filepath (str): Path to the file
        
        Returns:
            str: Property type ('apartment', 'house', or 'unknown')
        """
        filename = os.path.basename(filepath).lower()
        
        if 'appartement' in filename or 'apartment' in filename or 'loyers' in filename:
            return 'apartment'
        elif 'maison' in filename or 'house' in filename:
            return 'house'
        else:
            return 'unknown'
    
    def _determine_transaction_type(self, source_folder: str) -> str:
        """
        Determine transaction type based on the source folder.
        
        Args:
            source_folder (str): Source folder name
        
        Returns:
            str: Transaction type ('rent', 'sale', or 'unknown')
        """
        folder_lower = source_folder.lower()
        
        if 'rent' in folder_lower or 'loyer' in folder_lower:
            return 'rent'
        elif 'price' in folder_lower or 'vente' in folder_lower or 'prix' in folder_lower:
            return 'sale'
        else:
            return 'unknown'
    
    def _generate_data_hash(self, row: pd.Series, metadata: Dict[str, Any]) -> str:
        """
        Generate a hash for a data row combined with metadata to detect duplicates.
        
        Args:
            row (pd.Series): Data row
            metadata (Dict[str, Any]): Metadata about the row
        
        Returns:
            str: Hash of the combined data
        """
        data_str = str(row.to_dict())
        metadata_str = str(metadata)
        combined = data_str + metadata_str
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _is_duplicate(self, data_hash: str) -> bool:
        """
        Check if a data record with the given hash already exists.
        
        Args:
            data_hash (str): Hash of the data record
        
        Returns:
            bool: True if the record already exists
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM housing_data WHERE data_hash = ?", (data_hash,))
        result = cursor.fetchone()
        return result is not None
    
    def process_csv_file(self, filepath: str, source_type: str) -> Dict[str, int]:
        """
        Process a single CSV file and import it to the database.
        
        Args:
            filepath (str): Path to the CSV file
            source_type (str): Source type identifier (e.g., 'luxembourg_logements_rent')
        
        Returns:
            Dict[str, int]: Statistics about the processing
        """
        result = {
            "records_inserted": 0,
            "duplicates_skipped": 0,
            "errors": 0
        }
        
        try:
            # Calculate file hash
            file_hash = self._get_file_hash(filepath)
            
            # Skip if already processed
            if self._is_file_processed(filepath, file_hash):
                print(f"File already processed: {filepath}")
                return result
            
            # Load CSV
            df = pd.read_csv(filepath, encoding='utf-8')
            
            # Skip empty files
            if df.empty:
                print(f"Empty file: {filepath}")
                result["errors"] += 1
                return result
            
            # Standardize column names
            df = self._standardize_column_names(df)
            
            # Extract metadata
            year = self._extract_year_from_path(filepath)
            property_type = self._extract_property_type(filepath)
            transaction_type = self._determine_transaction_type(source_type)
            
            # Register file
            file_id = self._register_file(filepath, file_hash)
            
            # Process each row
            for _, row in df.iterrows():
                # Skip rows with missing key data
                if pd.isna(row.get('municipality')):
                    continue
                
                metadata = {
                    'year': year,
                    'property_type': property_type,
                    'transaction_type': transaction_type,
                    'source': source_type
                }
                
                # Generate data hash
                data_hash = self._generate_data_hash(row, metadata)
                
                # Skip if duplicate
                if self._is_duplicate(data_hash):
                    result["duplicates_skipped"] += 1
                    continue
                
                # Convert any NaN values to None for SQLite
                row_dict = row.where(pd.notnull(row), None).to_dict()
                
                # Insert data
                cursor = self.conn.cursor()
                try:
                    cursor.execute(
                        """
                        INSERT INTO housing_data 
                        (municipality, num_listings, avg_price, avg_price_sqm, 
                         year, property_type, transaction_type, source, file_id, data_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row_dict.get('municipality'),
                            row_dict.get('num_listings'),
                            row_dict.get('avg_price'),
                            row_dict.get('avg_price_sqm'),
                            year,
                            property_type,
                            transaction_type,
                            source_type,
                            file_id,
                            data_hash
                        )
                    )
                    self.conn.commit()
                    result["records_inserted"] += 1
                except Exception as e:
                    self.conn.rollback()
                    print(f"Error inserting row: {e}")
                    result["errors"] += 1
            
            return result
            
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            result["errors"] += 1
            return result
    
    def process_folder(self, base_folder: str, source_type: str) -> Dict[str, int]:
        """
        Process all CSV files in a folder and its subfolders.
        
        Args:
            base_folder (str): Path to the base folder
            source_type (str): Source type identifier
        
        Returns:
            Dict[str, int]: Statistics about the processing
        """
        # Connect to database
        if not self.conn:
            self._connect_to_db()
        
        folder_stats = {
            "files_processed": 0,
            "records_inserted": 0,
            "duplicates_skipped": 0,
            "errors": 0
        }
        
        # Walk through directory and process all CSV files
        for root, _, files in os.walk(base_folder):
            for file in files:
                if file.endswith('.csv'):
                    filepath = os.path.join(root, file)
                    print(f"Processing: {filepath}")
                    
                    result = self.process_csv_file(filepath, source_type)
                    
                    folder_stats["files_processed"] += 1
                    folder_stats["records_inserted"] += result["records_inserted"]
                    folder_stats["duplicates_skipped"] += result["duplicates_skipped"]
                    folder_stats["errors"] += result["errors"]
        
        # Update global stats
        self.stats["files_processed"] += folder_stats["files_processed"]
        self.stats["records_inserted"] += folder_stats["records_inserted"]
        self.stats["duplicates_skipped"] += folder_stats["duplicates_skipped"]
        self.stats["errors"] += folder_stats["errors"]
        
        return folder_stats
    
    def process_from_config(self, config_path: str = "./config/scraper_config.json", 
                          processed_base_dir: str = "processed") -> Dict[str, Any]:
        """
        Process all CSV files from the processed directories based on config file.
        
        Args:
            config_path (str): Path to the config file
            processed_base_dir (str): Base directory for processed files
        
        Returns:
            Dict[str, Any]: Statistics about the processing
        """
        # Connect to database
        if not self.conn:
            self._connect_to_db()
        
        # Reset statistics
        self.stats = {
            "files_processed": 0,
            "records_inserted": 0,
            "duplicates_skipped": 0,
            "errors": 0,
            "sites_processed": 0
        }
        
        try:
            # Load configuration
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Process each site
            for site in config.get("sites", []):
                site_name = site.get("name", "unnamed_site")
                subfolder = site.get("subfolder", site_name)
                
                # Define processed folder path
                processed_folder = os.path.join(processed_base_dir, subfolder)
                
                # Check if folder exists
                if not os.path.exists(processed_folder):
                    print(f"Processed folder not found: {processed_folder}")
                    self.stats["errors"] += 1
                    continue
                
                print(f"\nProcessing data from site: {site_name}")
                print(f"  Folder: {processed_folder}")
                
                # Process this folder
                folder_stats = self.process_folder(processed_folder, subfolder)
                
                print(f"  Files processed: {folder_stats['files_processed']}")
                print(f"  Records inserted: {folder_stats['records_inserted']}")
                print(f"  Duplicates skipped: {folder_stats['duplicates_skipped']}")
                print(f"  Errors: {folder_stats['errors']}")
                
                self.stats["sites_processed"] += 1
            
            # Print summary statistics
            print("\nProcessing complete!")
            print(f"Sites processed: {self.stats['sites_processed']}")
            print(f"Files processed: {self.stats['files_processed']}")
            print(f"Records inserted: {self.stats['records_inserted']}")
            print(f"Duplicates skipped: {self.stats['duplicates_skipped']}")
            print(f"Errors encountered: {self.stats['errors']}")
            
        except Exception as e:
            print(f"Error processing from config: {e}")
            self.stats["errors"] += 1
        
        # Close database connection
        self._close_db()
        
        return self.stats
    
    def query_data(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a query against the database and return results.
        
        Args:
            query (str): SQL query
            params (Tuple): Query parameters
        
        Returns:
            List[Dict[str, Any]]: Query results as a list of dictionaries
        """
        if not self.conn:
            self._connect_to_db()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            
            # Get column names
            columns = [col[0] for col in cursor.description]
            
            # Get results
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
        except Exception as e:
            print(f"Error executing query: {e}")
            return []
        finally:
            self._close_db()
    
    def get_available_years(self) -> List[int]:
        """
        Get a list of all available years in the database.
        
        Returns:
            List[int]: List of available years
        """
        results = self.query_data("SELECT DISTINCT year FROM housing_data WHERE year IS NOT NULL ORDER BY year")
        return [result['year'] for result in results]
    
    def get_available_municipalities(self) -> List[str]:
        """
        Get a list of all available municipalities in the database.
        
        Returns:
            List[str]: List of available municipalities
        """
        results = self.query_data("SELECT DISTINCT municipality FROM housing_data ORDER BY municipality")
        return [result['municipality'] for result in results]
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get the current processing statistics.
        
        Returns:
            Dict[str, int]: Processing statistics
        """
        return self.stats
    
    def get_db_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the database content.
        
        Returns:
            Dict[str, Any]: Database summary statistics
        """
        # Connect to database
        if not self.conn:
            self._connect_to_db()
        
        try:
            summary = {}
            
            # Total records
            result = self.query_data("SELECT COUNT(*) as count FROM housing_data")
            summary["total_records"] = result[0]["count"] if result else 0
            
            # Records by transaction type
            result = self.query_data(
                "SELECT transaction_type, COUNT(*) as count FROM housing_data GROUP BY transaction_type"
            )
            summary["by_transaction_type"] = {r["transaction_type"]: r["count"] for r in result}
            
            # Records by property type
            result = self.query_data(
                "SELECT property_type, COUNT(*) as count FROM housing_data GROUP BY property_type"
            )
            summary["by_property_type"] = {r["property_type"]: r["count"] for r in result}
            
            # Records by year
            result = self.query_data(
                "SELECT year, COUNT(*) as count FROM housing_data WHERE year IS NOT NULL GROUP BY year ORDER BY year"
            )
            summary["by_year"] = {r["year"]: r["count"] for r in result}
            
            # Total files processed
            result = self.query_data("SELECT COUNT(*) as count FROM processed_files")
            summary["total_files_processed"] = result[0]["count"] if result else 0
            
            return summary
            
        except Exception as e:
            print(f"Error getting database summary: {e}")
            return {"error": str(e)}
        finally:
            self._close_db()


def clean_data_from_config(config_path: str = "./config/scraper_config.json", 
                         processed_base_dir: str = "processed",
                         db_path: str = "luxembourg_housing.db") -> Dict[str, Any]:
    """
    Helper function to clean and import data from processed directories into SQLite.
    
    Args:
        config_path (str): Path to the configuration file
        processed_base_dir (str): Base directory for processed files
        db_path (str): Path to the SQLite database file
        
    Returns:
        Dict[str, Any]: Statistics about the processing
    """
    cleaner = DataCleaner(db_path=db_path)
    return cleaner.process_from_config(config_path=config_path, processed_base_dir=processed_base_dir)