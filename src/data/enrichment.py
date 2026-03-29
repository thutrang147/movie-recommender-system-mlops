import os
import re
import time
import logging
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from dotenv import load_dotenv

# 1. SETUP LOGGING 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 2. LOAD API KEY FROM .env
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    logger.error("TMDB_API_KEY not found! Please check .env file")
    exit(1)

# 3. SETUP RETRY MECHANISM
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

BASE_URL = "https://api.themoviedb.org/3/search/movie"

def extract_title_and_year(title_string):
    """Split title and year for API search"""
    match = re.search(r'(.*)\s\((\d{4})\)', str(title_string))
    if match:
        return match.group(1).strip(), match.group(2)
    return str(title_string), ""

def fetch_tmdb_data():
    logger.info("Starting Data Enrichment Pipeline from TMDB API...")
    
    input_path = 'data/processed/movies.parquet'
    output_path = 'data/processed/movies_enriched.parquet'
    
    if not os.path.exists(input_path):
        logger.error(f"Not found {input_path}. Please run validation.py first!")
        return

    movies_df = pd.read_parquet(input_path)
    overviews = []
    poster_paths = []

    logger.info(f"Fetching metadata for {len(movies_df)} movies...")

    # Use tqdm to show progress bar and ETA
    for _, row in tqdm(movies_df.iterrows(), total=len(movies_df), desc="Fetching Data"):
        clean_title, year = extract_title_and_year(row['title'])
        
        params = {
            'api_key': TMDB_API_KEY,
            'query': clean_title,
            'year': year,
            'language': 'en-US'
        }
        
        try:
            # Use session configured with retry to call API
            response = session.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status() # Raise error if HTTP code is not 200 OK
            data = response.json()
            
            if data.get('results') and len(data['results']) > 0:
                best_match = data['results'][0]
                overviews.append(best_match.get('overview', ''))
                poster_paths.append(best_match.get('poster_path', ''))
            else:
                overviews.append('')
                poster_paths.append('')
                
        except Exception as e:
            logger.warning(f"Skip movie '{clean_title}' due to connection error: {e}")
            overviews.append('')
            poster_paths.append('')
        
        # Sleep 0.05 seconds to respect server Rate Limit
        time.sleep(0.05) 

    # Update new data to DataFrame
    movies_df['overview'] = overviews
    movies_df['poster_path'] = poster_paths

    movies_df.to_parquet(output_path, index=False)
    logger.info(f"Completed! Saved clean and enriched data to {output_path}")

if __name__ == "__main__":
    fetch_tmdb_data()