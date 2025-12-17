import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
import re

class SpeechScraper:
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
        print("\n--- Starting Speeches Scraping ---")
        for year in range(self.start_year, self.end_year + 1):
            url = f"{self.base_url}/newsevents/speech/{year}-speeches.htm"
            soup = self.get_soup(url)
            if not soup:
                continue

            events = soup.find_all('div', class_='eventlist__event')
            if not events:
                events = soup.find_all('div', class_='row')

            for event in events:
                text_content = event.get_text()
                if "Jerome H. Powell" not in text_content and "Chair Powell" not in text_content:
                    continue

                link_tag = event.find('a', href=True)
                if not link_tag:
                    continue
                
                speech_url = self.base_url + link_tag['href'] if link_tag['href'].startswith('/') else link_tag['href']
                title = link_tag.get_text(strip=True)
                
                date_tag = event.find('time')
                date = date_tag.get_text(strip=True) if date_tag else "Unknown"

                self._parse_detail(speech_url, date, title)

    def _parse_detail(self, url, date, title):
        soup = self.get_soup(url)
        if not soup:
            return

        content_div = soup.find('div', class_='col-xs-12 col-sm-8 col-md-8')
        if not content_div:
            content_div = soup.find('div', id='article')

        if content_div:
            for script in content_div(["script", "style"]):
                script.decompose()
            
            paragraphs = content_div.find_all('p')
            for i, p in enumerate(paragraphs):
                p_text = p.get_text(strip=True)
                if len(p_text) > 50:
                    self.data.append({
                        'date': date,
                        'type': 'Speech',
                        'title': title,
                        'text': p_text,
                        'segment_id': i,
                        'url': url
                    })
            print(f"Parsed Speech: {title[:30]}...")

    def save(self):
        df = pd.DataFrame(self.data)
        
        # Get the directory where the script is located (scraping/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the project root (Fed_Project/)
        project_root = os.path.dirname(script_dir)
        # Define the path to the data folder
        data_dir = os.path.join(project_root, "data")
        
        # Ensure the data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        filename = os.path.join(data_dir, "fed_speeches.csv")
        
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\nSpeeches saved to {filename}. Total records: {len(df)}")

if __name__ == "__main__":
    scraper = SpeechScraper(start_year=2018, end_year=2024)
    scraper.scrape()
    scraper.save()
