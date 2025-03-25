import os
import pandas as pd
import json

class ExcelDataProcessor:
    """
    Class for finding and extracting tables from Excel files.
    """
    
    def __init__(self, min_table_rows=3, min_table_cols=2):
        """
        Initialize the processor.
        
        Args:
            min_table_rows (int): Minimum number of rows to consider a table
            min_table_cols (int): Minimum number of columns to consider a table
        """
        self.min_table_rows = min_table_rows
        self.min_table_cols = min_table_cols
    
    def load_excel(self, file_path):
        """
        Load all sheets from an Excel file.
        
        Args:
            file_path (str): Path to the Excel file
            
        Returns:
            dict: Dictionary of sheet names mapped to DataFrames
        """
        print(f"Loading all sheets from {os.path.basename(file_path)}")
        try:
            # Load all sheets into a dictionary
            excel_data = pd.read_excel(file_path, sheet_name=None)
            return excel_data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def find_tables_in_sheet(self, sheet_df):
        """
        Find potential tables within a DataFrame representing a sheet.
        
        Args:
            sheet_df (pandas.DataFrame): DataFrame containing sheet data
            
        Returns:
            list: List of dictionaries with table coordinates
        """
        # Basic implementation to find tables by looking for non-empty regions
        tables = []
        non_empty_rows = []
        
        # Find rows with content
        for i, row in sheet_df.iterrows():
            if not row.isnull().all():
                non_empty_rows.append(i)
        
        if not non_empty_rows:
            return tables
        
        # Group consecutive rows as potential tables
        table_start = non_empty_rows[0]
        prev_row = non_empty_rows[0]
        
        for row_idx in non_empty_rows[1:] + [None]:  # Add None to handle the last group
            if row_idx is None or row_idx > prev_row + 1:
                # Found a gap, or end of data - check if the previous group forms a table
                table_end = prev_row
                table_rows = table_end - table_start + 1
                
                if table_rows >= self.min_table_rows:
                    # Find the columns with data in this range
                    table_data = sheet_df.iloc[table_start:table_end+1]
                    non_empty_cols = [col for col in table_data.columns 
                                     if not table_data[col].isnull().all()]
                    
                    if len(non_empty_cols) >= self.min_table_cols:
                        # Found a table
                        tables.append({
                            'start_row': table_start,
                            'end_row': table_end,
                            'columns': non_empty_cols
                        })
                
                if row_idx is not None:
                    # Start a new potential table
                    table_start = row_idx
            
            if row_idx is not None:
                prev_row = row_idx
        
        return tables
    
    def extract_table(self, sheet_df, table_coords):
        """
        Extract a table from a sheet DataFrame using the coordinates.
        
        Args:
            sheet_df (pandas.DataFrame): DataFrame containing sheet data
            table_coords (dict): Dictionary with table coordinates
            
        Returns:
            pandas.DataFrame: Extracted table
        """
        start_row = table_coords['start_row']
        end_row = table_coords['end_row']
        columns = table_coords['columns']
        
        # Extract the table data
        table_df = sheet_df.iloc[start_row:end_row+1][columns].copy()
        
        # If the first row contains headers, use it as column names
        if not table_df.iloc[0].isnull().all():
            new_headers = table_df.iloc[0]
            table_df = table_df.iloc[1:].copy()
            table_df.columns = new_headers
        
        # Reset index
        table_df = table_df.reset_index(drop=True)
        
        return table_df


class ConfigProcessor:
    """
    Class for processing Excel files based on configuration settings.
    Can be imported and used in notebooks or other Python code.
    """
    
    def __init__(self, min_table_rows=3, min_table_cols=2):
        """
        Initialize the processor.
        
        Args:
            min_table_rows (int): Minimum number of rows to consider a table
            min_table_cols (int): Minimum number of columns to consider a table
        """
        self.processor = ExcelDataProcessor(min_table_rows=min_table_rows, 
                                       min_table_cols=min_table_cols)
        self.stats = {
            "sites_processed": 0,
            "files_processed": 0,
            "tables_extracted": 0,
            "errors": 0
        }
    
    def process_from_config(self, config_path="./config/scraper_config.json", site_name=None, 
                           verbose=True, base_output_dir="processed"):
        """
        Process Excel files based on configuration settings.
        
        Args:
            config_path (str): Path to the configuration JSON file
            site_name (str, optional): Specific site name to process. If None, process all sites.
            verbose (bool): Whether to print progress information
            base_output_dir (str): Base directory for processed output
            
        Returns:
            dict: Statistics about processing (files processed, tables extracted, etc.)
        """
        # Reset statistics
        self.stats = {
            "sites_processed": 0,
            "files_processed": 0,
            "tables_extracted": 0,
            "errors": 0,
            "processed_files": []  # Track processed files for easy access in notebooks
        }
        
        # Load configuration
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            if verbose:
                print(f"Error loading configuration file {config_path}: {e}")
            self.stats["error"] = f"Failed to load config: {str(e)}"
            return self.stats
        
        # Get base folder from config
        base_folder = config.get("base_download_folder", "download")
        
        # Get sites to process
        sites_to_process = []
        for site in config.get("sites", []):
            if site_name is None or site.get("name") == site_name:
                sites_to_process.append(site)
        
        if not sites_to_process:
            message = f"No matching sites found in config" + (f" for site name '{site_name}'" if site_name else "")
            if verbose:
                print(message)
            self.stats["error"] = "No matching sites found"
            return self.stats
        
        # Process each site
        for site in sites_to_process:
            site_name = site.get("name", "unnamed_site")
            subfolder = site.get("subfolder", site_name)
            
            # Define input and output paths
            input_folder = os.path.join(base_folder, subfolder)
            output_folder = os.path.join(base_output_dir, subfolder)
            
            # Check if input folder exists
            if not os.path.exists(input_folder):
                if verbose:
                    print(f"Input folder not found: {input_folder}")
                self.stats["errors"] += 1
                continue
            
            # Create output folder
            os.makedirs(output_folder, exist_ok=True)
            
            if verbose:
                print(f"\nProcessing site: {site_name}")
                print(f"  Input folder: {input_folder}")
                print(f"  Output folder: {output_folder}")
            
            # Get list of Excel files
            excel_files = [f for f in os.listdir(input_folder) 
                        if f.endswith(('.xls', '.xlsx')) and not f.startswith('~$')]
            
            if verbose:
                print(f"  Found {len(excel_files)} Excel files to process")
            
            # Process each file
            for excel_file in excel_files:
                file_path = os.path.join(input_folder, excel_file)
                file_name = os.path.splitext(excel_file)[0]
                
                if verbose:
                    print(f"  Processing: {excel_file}")
                
                try:
                    # Process this file
                    file_results = self._process_excel_file(
                        file_path=file_path, 
                        file_name=file_name,
                        output_folder=output_folder,
                        verbose=verbose
                    )
                    
                    # Track processed files for notebook access
                    self.stats["processed_files"].extend(file_results["output_files"])
                    self.stats["tables_extracted"] += file_results["tables_extracted"]
                    self.stats["files_processed"] += 1
                    
                except Exception as e:
                    if verbose:
                        print(f"    Error processing {excel_file}: {e}")
                    self.stats["errors"] += 1
            
            self.stats["sites_processed"] += 1
        
        # Print summary statistics
        if verbose:
            print("\nProcessing complete!")
            print(f"Sites processed: {self.stats['sites_processed']}")
            print(f"Files processed: {self.stats['files_processed']}")
            print(f"Tables extracted: {self.stats['tables_extracted']}")
            print(f"Errors encountered: {self.stats['errors']}")
        
        return self.stats
    
    def _process_excel_file(self, file_path, file_name, output_folder, verbose=True):
        """
        Process a single Excel file and extract tables from each sheet.
        
        Args:
            file_path (str): Path to the Excel file
            file_name (str): Base name of the file (without extension)
            output_folder (str): Folder to save processed CSV files
            verbose (bool): Whether to print progress information
            
        Returns:
            dict: Results of processing this file
        """
        results = {
            "tables_extracted": 0,
            "output_files": []
        }
        
        # Load all sheets from the Excel file
        all_sheets = self.processor.load_excel(file_path)
        
        # Process each sheet in the file
        for sheet_name, sheet_df in all_sheets.items():
            if verbose:
                print(f"    Analyzing sheet: {sheet_name}")
            
            # Extract year information from sheet name if available
            year = self._extract_year(sheet_name, file_name)
            
            # Find tables in the current sheet
            tables = self.processor.find_tables_in_sheet(sheet_df)
            
            if tables:
                # Find the largest table in this sheet
                largest_table, table_info = self._get_largest_table(sheet_df, tables)
                
                if largest_table is not None:
                    if verbose:
                        print(f"      Found table with {table_info['rows']} rows, {table_info['cols']} columns")
                    
                    # Clean the table
                    largest_table.columns = [str(col).lower().strip().replace(' ', '_') for col in largest_table.columns]
                    largest_table = largest_table.dropna(how='all')
                    
                    # Create year subfolder if available
                    output_subdir = output_folder
                    if year:
                        output_subdir = os.path.join(output_folder, f"year_{year}")
                        os.makedirs(output_subdir, exist_ok=True)
                    
                    # Create appropriate filename
                    if len(all_sheets) == 1:
                        output_file = os.path.join(output_subdir, f"{file_name}.csv")
                    else:
                        # For multi-sheet files, include sheet name in filename
                        sheet_suffix = self._clean_string_for_filename(sheet_name) if isinstance(sheet_name, str) else f"sheet_{sheet_name}"
                        output_file = os.path.join(output_subdir, f"{file_name}_{sheet_suffix}.csv")
                    
                    # Save to CSV
                    largest_table.to_csv(output_file, index=False)
                    results["output_files"].append(output_file)
                    
                    if verbose:
                        print(f"      Saved to {output_file}")
                        print(f"      Table shape: {largest_table.shape}")
                        column_preview = ", ".join(str(col) for col in largest_table.columns[:5])
                        if len(largest_table.columns) > 5:
                            column_preview += "..."
                        print(f"      Columns: {column_preview}")
                    
                    results["tables_extracted"] += 1
            elif verbose:
                print(f"      No tables found in sheet '{sheet_name}'")
        
        return results
    
    def _extract_year(self, sheet_name, file_name):
        """Extract year from sheet name or file name"""
        year = None
        
        # Try to get year from sheet_name first
        if isinstance(sheet_name, str):
            # Look for 4-digit numbers that might be years
            potential_years = [part.strip() for part in sheet_name.split() if part.strip().isdigit() and len(part.strip()) == 4]
            if potential_years:
                year = potential_years[0]
            
            # If not found, check for year ranges like "2023-24"
            if not year and "-" in sheet_name:
                parts = [p.strip() for p in sheet_name.split("-")]
                for part in parts:
                    if part.isdigit() and len(part) == 4:
                        year = part
                        break
        
        # If no year in sheet_name, try filename
        if not year and "-" in file_name:
            parts = file_name.split("-")
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    year = part
                    break
        
        return year
    
    def _get_largest_table(self, sheet_df, tables):
        """Find the largest table in a sheet"""
        largest_table = None
        max_cells = 0
        table_info = {"rows": 0, "cols": 0}
        
        for table_coords in tables:
            table_df = self.processor.extract_table(sheet_df, table_coords)
            num_cells = table_df.shape[0] * table_df.shape[1]
            
            # Check if this table has more cells than the current largest
            if num_cells > max_cells and num_cells > 0:
                max_cells = num_cells
                largest_table = table_df
                table_info = {"rows": table_df.shape[0], "cols": table_df.shape[1]}
        
        return largest_table, table_info
    
    def _clean_string_for_filename(self, s):
        """Convert a string to be usable as part of a filename"""
        if not isinstance(s, str):
            return f"value_{s}"
        
        # Replace problematic characters
        s = s.replace(" ", "_")
        s = s.replace("/", "-")
        s = s.replace("\\", "-")
        s = s.replace(":", "-")
        s = s.replace("*", "")
        s = s.replace("?", "")
        s = s.replace("\"", "")
        s = s.replace("<", "")
        s = s.replace(">", "")
        s = s.replace("|", "-")
        
        return s
    
    def get_processed_files(self):
        """Get a list of all processed files (useful for notebooks)"""
        return self.stats.get("processed_files", [])
    
    def get_stats(self):
        """Get processing statistics"""
        return self.stats


