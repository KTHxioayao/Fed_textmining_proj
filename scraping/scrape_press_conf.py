import requests
from bs4 import BeautifulSoup
import os
import time

class FedPressConfScraper:
    def __init__(self, data_folder):
        self.pdf_folder = os.path.join(data_folder, "raw", "press_conf_pdfs")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        if not os.path.exists(self.pdf_folder):
            os.makedirs(self.pdf_folder)

    def download_pdfs(self, start_year=2018, end_year=2024):
        print(f"\n[Downloading PDFs] Starting download to {self.pdf_folder}...")
        downloaded = set(os.listdir(self.pdf_folder))
        base_url = "https://www.federalreserve.gov"

        for year in range(start_year, end_year + 1):
            target_urls = [
                f"{base_url}/monetarypolicy/fomchistorical{year}.htm",
                f"{base_url}/monetarypolicy/fomccalendars.htm"
            ]
            
            print(f"  Checking year {year}...")
            found_year = False
            
            for index_url in target_urls:
                if found_year and "fomccalendars" in index_url: break 
                try:
                    resp = requests.get(index_url, headers=self.headers, timeout=10)
                    if resp.status_code != 200: continue
                    
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    links = soup.find_all('a', href=True)
                    
                    for link in links:
                        href = link['href']
                        text = link.get_text().strip()
                        
                        if str(year) not in href and str(year) not in text: continue
                        
                        is_target_pdf = href.lower().endswith('.pdf') and ("presconf" in href.lower() or "confcall" in href.lower())
                        
                        if is_target_pdf:
                            self._download_file(base_url, href, downloaded)
                            found_year = True
                        elif "fomcpresconf" in href.lower() and href.lower().endswith('.htm'):
                            self._handle_intermediate_page(base_url, href, downloaded)
                            found_year = True
                            
                except Exception as e:
                    print(f"    Error accessing {index_url}: {e}")
        
        print(f"\n[Download Complete] Total PDFs: {len(os.listdir(self.pdf_folder))}")

    def _handle_intermediate_page(self, base_url, href, downloaded):
        full_url = base_url + href if href.startswith('/') else href
        try:
            resp = requests.get(full_url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, 'html.parser')
                for link in soup.find_all('a', href=True):
                    if link['href'].lower().endswith('.pdf') and "presconf" in link['href'].lower():
                        self._download_file(base_url, link['href'], downloaded)
        except Exception:
            pass

    def _download_file(self, base_url, href, downloaded):
        filename = href.split('/')[-1]
        if filename in downloaded: return
        
        url = base_url + href if href.startswith('/') else href
        save_path = os.path.join(self.pdf_folder, filename)
        
        try:
            print(f"    Downloading: {filename}")
            r = requests.get(url, headers=self.headers, timeout=15)
            with open(save_path, 'wb') as f:
                f.write(r.content)
            downloaded.add(filename)
            time.sleep(1)
        except Exception as e:
            print(f"    Failed: {e}")

if __name__ == "__main__":
    DATA_ROOT = r"e:\Textming\data"
    scraper = FedPressConfScraper(DATA_ROOT)
    scraper.download_pdfs(start_year=2018, end_year=2024)
