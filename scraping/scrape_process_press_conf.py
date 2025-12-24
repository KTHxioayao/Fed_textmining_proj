import pdfplumber
import pandas as pd
import os
import re
import requests
from bs4 import BeautifulSoup
import time
import random
import csv

class FedPressConfProcessor:
    def __init__(self, data_folder):
        self.pdf_folder = os.path.join(data_folder, "raw", "press_conf_pdfs")
        self.output_csv = os.path.join(data_folder, "processed", "fed_press_conf_structured.csv")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        if not os.path.exists(self.pdf_folder):
            os.makedirs(self.pdf_folder)

    def download_pdfs(self, start_year=2018, end_year=2024):
        print(f"\n[Step 1] Starting Download to {self.pdf_folder}...")
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

    def process_pdfs(self):
        print(f"\n[Step 2] Processing PDFs in {self.pdf_folder}...")
        all_rows = []
        files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        
        for filename in files:
            print(f"  Processing: {filename}")
            file_path = os.path.join(self.pdf_folder, filename)
            
            date_str = self._extract_date_from_filename(filename)
            is_conf_call = "confcall" in filename.lower()
            
            raw_text = self._extract_text_with_crop(file_path)
            if not raw_text: continue
            
            clean_text = self._clean_noise(raw_text)
            
            if is_conf_call:
                segments = self._process_conf_call(clean_text, date_str)
            else:
                segments = self._structure_press_conf(clean_text, date_str)
            
            all_rows.extend(segments)
            
        df = pd.DataFrame(all_rows)
        if not df.empty:
            cols = ['date', 'section', 'text']
            for c in df.columns:
                if c not in cols:
                    cols.append(c)
            df = df[cols]
        
        # Ensure quoting is set to QUOTE_ALL to handle commas/newlines in text correctly
        df.to_csv(self.output_csv, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        print(f"\n[Done] Saved {len(df)} segments to {self.output_csv}")

    def _extract_date_from_filename(self, filename):
        match = re.search(r'(20\d{2})(\d{2})(\d{2})', filename)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        match_year = re.search(r'20\d{2}', filename)
        return f"{match_year.group(0)}-01-01" if match_year else "Unknown"

    def _extract_text_with_crop(self, pdf_path):
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    crop_box = (0, page.height * 0.15, page.width, page.height * 0.9)
                    try:
                        text += page.crop(crop_box).extract_text() + "\n"
                    except:
                        text += page.extract_text() + "\n"
        except Exception as e:
            print(f"    Error reading PDF: {e}")
            return None
        return text

    def _clean_noise(self, text):
        lines = text.split('\n')
        cleaned = []
        patterns = [
            r"Page \d+ of \d+", 
            r"^FINAL$", 
            r"Chair(man)? Powell['’]?s? Press Conference", 
            r"Transcript of .* Press Conference", 
            r"^[A-Z][a-z]+ \d{1,2}, 20\d{2}$",
            r"Chair Powell['’]?s? Press Conference Call" 
        ]
        
        for line in lines:
            line = line.strip()
            if not line or line.isdigit(): continue
            is_noise = False
            for pat in patterns:
                if re.search(pat, line, re.IGNORECASE):
                    is_noise = True
                    break
            if not is_noise:
                cleaned.append(line)
        return " ".join(cleaned)

    def _structure_press_conf(self, text, date_str):
        split_markers = [
            "I will now take your questions", 
            "I am happy to take your questions",
            "I'm happy to take your questions",
            "happy to take your questions", 
            "happy to respond to your questions",
            "We will now take questions",
            "Q & A",
            "MICHELLE SMITH. Thank you",
            "MICHELLE SMITH. We will now go to",
            "MICHELLE SMITH. We'll go to",
            "MICHELLE SMITH. Let's go to",
            "MICHELLE SMITH. Our first question",
            "I'll be happy to take your questions",
            "I will be happy to take your questions",
            "we'll be happy to take your questions",
            "MICHELLE SMITH.",
            # Added based on 2023-07-26 and others
            "look forward to your questions",
            "I look forward to your questions",
            "I look forward to taking your questions",
            "I would be happy to take your questions"
        ]
        
        split_idx = -1
        text_lower = text.lower()
        
        # Find the earliest marker
        min_idx = float('inf')
        found_marker = None
        
        for marker in split_markers:
            idx = text_lower.find(marker.lower())
            if idx != -1 and idx < min_idx:
                min_idx = idx
                found_marker = marker
        
        if min_idx != float('inf'):
            split_idx = min_idx
            print(f"DEBUG: Found earliest marker '{found_marker}' at {split_idx} for date {date_str}")
        else:
            print(f"DEBUG: No marker found for date {date_str}")
        
        segments = []
        if split_idx != -1:
            raw_opening = text[:split_idx]
            raw_qa = text[split_idx:]
            
            op_text = self._extract_speaker_segments(raw_opening, is_qa=False)
            if op_text:
                segments.append({'date': date_str, 'section': 'Opening Statement', 'text': op_text})
                
            qa_texts = self._extract_speaker_segments(raw_qa, is_qa=True)
            for t in qa_texts:
                segments.append({'date': date_str, 'section': 'Q&A', 'text': t})
        else:
            full_texts = self._extract_speaker_segments(text, is_qa=False)
            if isinstance(full_texts, list):
                for t in full_texts:
                    segments.append({'date': date_str, 'section': 'Full Text', 'text': t})
            else:
                segments.append({'date': date_str, 'section': 'Full Text', 'text': full_texts})
        return segments

    def _process_conf_call(self, text, date_str):
        segments = []
        speaker_pattern = re.compile(r'(?:^|\s)([A-Z\s\.\'’-]{3,})\s*[\.:]')
        parts = speaker_pattern.split(text)
        if len(parts) >= 2:
            for i in range(1, len(parts), 2):
                speaker = parts[i].strip().upper()
                content = parts[i+1].strip()
                if self._is_noise_content(content): continue
                if "TRANSCRIPT OF" in speaker or "PAGE" in speaker: continue
                
                segments.append({
                    'date': date_str, 
                    'section': 'Conference Call', 
                    'speaker': speaker,
                    'text': content
                })
        return segments

    def _is_noise_content(self, text):
        noise_phrases = [
            "Thank you", "Thanks", "You're on mute", "Can you hear me", 
            "[No response]", "(No response)", "hearing no objection",
            "so moved", "second", "all in favor", "aye"
        ]
        text_lower = text.lower().strip("., ")
        if len(text) < 5: return True
        for phrase in noise_phrases:
            if text_lower == phrase.lower(): return True
            if len(text) < 30 and phrase.lower() in text_lower: return True
        return False

    def _extract_speaker_segments(self, text, is_qa):
        pattern = re.compile(r'(?:^|\s)([A-Z\s\.\'’-]{3,}|Transcript of the Federal Open Market Committee Conference Call)\s*[\.:]')
        
        parts = pattern.split(text)
        valid_segments = []
        
        if len(parts) < 2:
            return [] if is_qa else text

        for i in range(1, len(parts), 2):
            speaker = parts[i].strip().upper()
            content = parts[i+1].strip()
            
            if "POWELL" in speaker or "TRANSCRIPT OF THE FEDERAL" in speaker:
                if "Transcript of" in content[:100]:
                     content = content.split("Transcript of")[0]
                if self._is_noise_content(content): continue
                if len(content) > 10: 
                    valid_segments.append(content)
        
        if is_qa:
            return valid_segments 
        else:
            return " ".join(valid_segments)

if __name__ == "__main__":
    DATA_ROOT = r"e:\Textming\data"
    processor = FedPressConfProcessor(DATA_ROOT)
    processor.download_pdfs(start_year=2018, end_year=2024)
    processor.process_pdfs()