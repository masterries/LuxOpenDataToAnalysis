import requests
from bs4 import BeautifulSoup
import os
import re
import json
import pandas as pd
from typing import List, Dict, Union, Optional
from urllib.parse import urlparse


class DataScraper:
    """
    A flexible data scraper for downloading datasets from various sources.
    """
    
    def __init__(self, base_download_folder: str = "download"):
        """
        Initialize the scraper with a base download folder.
        
        Args:
            base_download_folder: The base folder where downloads will be stored
        """
        self.base_download_folder = base_download_folder
        
    def create_folder(self, subfolder: str = "") -> str:
        """
        Create a download folder if it doesn't exist.
        
        Args:
            subfolder: Optional subfolder within the base download folder
            
        Returns:
            The path to the created folder
        """
        # Combine base folder with subfolder if provided
        if subfolder:
            folder_path = os.path.join(self.base_download_folder, subfolder)
        else:
            folder_path = self.base_download_folder
            
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder at {folder_path}")
        else:
            print(f"Folder exists at {folder_path}")
            
        return folder_path
    
    def get_html_content(self, url: str) -> str:
        """
        Fetch HTML content from a URL.
        
        Args:
            url: The URL to fetch
            
        Returns:
            The HTML content as a string
            
        Raises:
            requests.exceptions.RequestException: If request fails
        """
        print(f"Fetching HTML from {url}")
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    
    def extract_download_links(self, html_content: str, link_pattern: str) -> List[Dict[str, str]]:
        """
        Extract download links from HTML content based on a pattern.
        
        Args:
            html_content: The HTML content to parse
            link_pattern: A string pattern to match in href attributes
            
        Returns:
            A list of dictionaries containing URL and text for each link
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        download_links = []
        
        for a_tag in soup.find_all('a', href=True):
            if link_pattern in a_tag['href']:
                download_links.append({
                    'url': a_tag['href'],
                    'text': a_tag.get_text().strip()
                })
                
        return download_links
    
    def download_file(self, link_info: Dict[str, str], download_folder: str) -> str:
        """
        Download a file from a link, handling redirects and determining filename.
        
        Args:
            link_info: Dictionary containing 'url' and 'text' keys
            download_folder: Folder to save the downloaded file
            
        Returns:
            Path to the downloaded file
            
        Raises:
            requests.exceptions.RequestException: If download fails
        """
        link = link_info['url']
        name = link_info['text']
        
        print(f"Processing file: {name}")
        
        # Make a HEAD request first to check redirects and get filename
        head_response = requests.head(link, allow_redirects=True)
        final_url = head_response.url
        
        print(f"Initial URL: {link}")
        print(f"Final URL after redirect: {final_url}")
        
        # Try to get filename from Content-Disposition header
        filename = None
        if 'Content-Disposition' in head_response.headers:
            content_disp = head_response.headers['Content-Disposition']
            filename_match = re.search(r'filename="(.+?)"', content_disp)
            if filename_match:
                filename = filename_match.group(1)
        
        # If no filename from header, extract from URL
        if not filename:
            filename = os.path.basename(final_url)
            # Remove query parameters if any
            if '?' in filename:
                filename = filename.split('?')[0]
        
        # If still no valid filename, use the domain and a timestamp
        if not filename or len(filename) < 3:
            # Try to determine file extension
            file_ext = self._guess_extension(final_url)
            # Use domain name as part of filename
            domain = urlparse(final_url).netloc.split('.')[-2]
            # Use a default name with the link index
            import time
            timestamp = int(time.time())
            filename = f"{domain}_dataset_{timestamp}{file_ext}"
        
        # Sanitize filename to remove invalid characters
        filename = re.sub(r'[\\/*?:"<>|]', '', filename)
        
        file_path = os.path.join(download_folder, filename)
        
        print(f"Downloading as: {filename}")
        
        # Now make a GET request to download the file
        file_response = requests.get(final_url, stream=True)
        file_response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Saved to {file_path}")
        return file_path
    
    def _guess_extension(self, url: str) -> str:
        """
        Guess file extension from URL.
        
        Args:
            url: URL to analyze
            
        Returns:
            File extension with dot
        """
        # Common file extensions to check
        extensions = {
            '.xls': ['xls'],
            '.xlsx': ['xlsx'],
            '.csv': ['csv'],
            '.json': ['json'],
            '.xml': ['xml'],
            '.pdf': ['pdf'],
            '.zip': ['zip'],
            '.txt': ['txt']
        }
        
        # Try to find extension in URL
        for ext, patterns in extensions.items():
            for pattern in patterns:
                if f'.{pattern}' in url.lower() or f'/{pattern}' in url.lower():
                    return ext
        
        # Default to .dat if no match
        return '.dat'
    
    def list_downloaded_files(self, folder: str) -> List[Dict[str, Union[str, float]]]:
        """
        List all files in the download folder with sizes.
        
        Args:
            folder: Folder to list files from
            
        Returns:
            List of dictionaries with file information
        """
        files_info = []
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path) / 1024  # Size in KB
                files_info.append({
                    'name': file,
                    'path': file_path,
                    'size_kb': round(file_size, 2)
                })
        return files_info
    
    def preview_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Try to preview a data file (CSV, Excel, etc.).
        
        Args:
            file_path: Path to the file
            
        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            if file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
                print(f"Successfully read Excel file: {os.path.basename(file_path)}")
                return df
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                print(f"Successfully read CSV file: {os.path.basename(file_path)}")
                return df
            else:
                print(f"File type not supported for preview: {os.path.basename(file_path)}")
                return None
        except Exception as e:
            print(f"Error reading file {os.path.basename(file_path)}: {e}")
            return None
    
    def scrape_and_download(self, url: str, link_pattern: str, subfolder: str = "") -> List[str]:
        """
        Main method to scrape a URL, find download links, and download files.
        
        Args:
            url: The URL to scrape
            link_pattern: Pattern to match in href attributes
            subfolder: Optional subfolder within base download folder
            
        Returns:
            List of paths to downloaded files
        """
        # Create download folder
        download_folder = self.create_folder(subfolder)
        
        # Get HTML content
        html_content = self.get_html_content(url)
        
        # Extract download links
        download_links = self.extract_download_links(html_content, link_pattern)
        
        print(f"\nFound {len(download_links)} download links:")
        for i, link_info in enumerate(download_links):
            print(f"{i+1}. {link_info['text']} - {link_info['url']}")
        
        # Download each file
        downloaded_files = []
        for i, link_info in enumerate(download_links):
            try:
                file_path = self.download_file(link_info, download_folder)
                downloaded_files.append(file_path)
            except Exception as e:
                print(f"Error downloading {link_info['url']}: {e}")
        
        # List downloaded files
        files_info = self.list_downloaded_files(download_folder)
        print("\nDownloaded files:")
        for file_info in files_info:
            print(f"- {file_info['name']} ({file_info['size_kb']} KB)")
        
        return downloaded_files


def scrape_from_config(config_file: str = "./config/scraper_config.json") -> Dict[str, List[str]]:
    """
    Scrape data using configuration from a JSON file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Dictionary mapping site names to lists of downloaded files
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config file {config_file} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error parsing config file {config_file}.")
        return {}
    
    # Create scraper
    base_folder = config.get('base_download_folder', 'download')
    scraper = DataScraper(base_folder)
    
    results = {}
    # Process each site in config
    sites = config.get('sites', [])
    for site in sites:
        name = site.get('name', 'unnamed_site')
        url = site.get('url')
        link_pattern = site.get('link_pattern')
        subfolder = site.get('subfolder', name)
        
        if not url or not link_pattern:
            print(f"Skipping site {name}: missing URL or link pattern.")
            continue
        
        print(f"\n--- Processing site: {name} ---")
        downloaded_files = scraper.scrape_and_download(url, link_pattern, subfolder)
        results[name] = downloaded_files
    
    return results


# Example config file structure:
# {
#   "base_download_folder": "download",
#   "sites": [
#     {
#       "name": "luxembourg_housing",
#       "url": "https://data.public.lu/fr/datasets/loyers-annonces-des-logements-par-commune/",
#       "link_pattern": "data.public.lu/fr/datasets/r/",
#       "subfolder": "luxembourg_housing"
#     }
#   ]
# }