import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
import re

class MinutesScraper:
    def __init__(self, start_year=2018, end_year=2024):
        self.start_year = start_year
        self.end_year = end_year
        self.base_url = "https://www.federalreserve.gov"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.data = []

    def get_soup(self, url):
        try:
            print(f"Fetching: {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            time.sleep(random.uniform(1, 3))
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def scrape(self):
        print("\n--- Starting FOMC Minutes Scraping ---")
        for year in range(self.start_year, self.end_year + 1):
            if year >= 2020:
                url = f"{self.base_url}/monetarypolicy/fomccalendars.htm"
            else:
                url = f"{self.base_url}/monetarypolicy/fomchistorical{year}.htm"
            
            soup = self.get_soup(url)
            if not soup:
                continue

            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                if f'fomcminutes{year}' in href and href.endswith('.htm'):
                    minutes_url = self.base_url + href if href.startswith('/') else href
                    
                    date_match = re.search(r'(\d{8})', href)
                    if date_match:
                        date_str = date_match.group(1)
                        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                    else:
                        formatted_date = f"{year}-01-01"

                    self._parse_detail(minutes_url, formatted_date)

    def _parse_detail(self, url, date):
        soup = self.get_soup(url)
        if not soup:
            return

        content_div = soup.find('div', id='article')
        if not content_div:
            content_div = soup.find('div', class_='col-xs-12 col-sm-8 col-md-8')

        if content_div:
            for script in content_div(["script", "style"]):
                script.decompose()
            
            paragraphs = content_div.find_all('p')
            for i, p in enumerate(paragraphs):
                p_text = p.get_text(strip=True)
                if len(p_text) > 50:
                    self.data.append({
                        'date': date,
                        'type': 'Minutes',
                        'title': 'FOMC Minutes',
                        'text': p_text,
                        'segment_id': i,
                        'url': url
                    })
            print(f"Parsed Minutes from {url}")

    def save(self):
        df = pd.DataFrame(self.data)
        
        # Get the directory where the script is located (scraping/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the project root (Fed_Project/)
        project_root = os.path.dirname(script_dir)
        # Define the path to the data folder
        data_dir = os.path.join(project_root, "data/raw")
        
        # Ensure the data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        filename = os.path.join(data_dir, "fed_minutes.csv")
        
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\nMinutes saved to {filename}. Total records: {len(df)}")

if __name__ == "__main__":
    scraper = MinutesScraper(start_year=2018, end_year=2024)
    scraper.scrape()
    scraper.save()
