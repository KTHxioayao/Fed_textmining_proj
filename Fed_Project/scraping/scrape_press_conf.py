import pdfplumber
import pandas as pd
import os
import re
import requests
from bs4 import BeautifulSoup
import time
import random

def download_press_conf_pdfs(pdf_folder, start_year=2018, end_year=2024):
    """
    从美联储官网下载发布会实录 PDF
    """
    base_url = "https://www.federalreserve.gov"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        print(f"Created directory: {pdf_folder}")

    print(f"\n--- Starting PDF Download (Target: {pdf_folder}) ---")
    print("Note: Before 2019, Press Conferences were only held quarterly (Mar, Jun, Sep, Dec).")
    
    downloaded_files = set(os.listdir(pdf_folder))

    for year in range(start_year, end_year + 1):
        potential_urls = [
            f"{base_url}/monetarypolicy/fomchistorical{year}.htm",
            f"{base_url}/monetarypolicy/fomccalendars.htm"
        ]
        
        found_year_pdfs = False
        count_for_year = 0
        
        for index_url in potential_urls:
            try:
                if "fomccalendars" in index_url and found_year_pdfs and year < 2023:
                    continue
                    
                print(f"Checking Listing URL: {index_url}")
                response = requests.get(index_url, headers=headers, timeout=10)
                
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                links = soup.find_all('a', href=True)
                
                for link in links:
                    href = link['href']
                    text = link.get_text().strip()
                    full_href = base_url + href if href.startswith('/') else href
                    
                    if str(year) not in href and str(year) not in text:
                        continue
                        
                    is_pdf = href.lower().endswith('.pdf')
                    is_target = "presconf" in href.lower() or "confcall" in href.lower()
                    
                    if is_pdf and is_target:
                        if _download_pdf(full_href, pdf_folder, headers, downloaded_files):
                            count_for_year += 1
                        found_year_pdfs = True

                    elif "fomcpresconf" in href.lower() and href.lower().endswith('.htm'):
                        try:
                            sub_response = requests.get(full_href, headers=headers, timeout=10)
                            if sub_response.status_code == 200:
                                sub_soup = BeautifulSoup(sub_response.content, 'html.parser')
                                sub_links = sub_soup.find_all('a', href=True)
                                for sub_link in sub_links:
                                    sub_href = sub_link['href']
                                    if sub_href.lower().endswith('.pdf') and "presconf" in sub_href.lower():
                                        pdf_url = base_url + sub_href if sub_href.startswith('/') else sub_href
                                        if _download_pdf(pdf_url, pdf_folder, headers, downloaded_files):
                                            count_for_year += 1
                                        found_year_pdfs = True
                            time.sleep(0.5)
                        except Exception as e:
                            print(f"  Error accessing intermediate page {full_href}: {e}")

            except Exception as e:
                print(f"Error accessing {index_url}: {e}")
                
        if count_for_year > 0:
             print(f"  > Found {count_for_year} Press Conferences for {year}")
        elif not found_year_pdfs:
             print(f"  > Warning: No Press Conference PDFs found for {year} yet.")

    print("--- Download Complete ---\n")

def _download_pdf(url, folder, headers, downloaded_set):
    filename = url.split('/')[-1]
    if "presconf" not in filename.lower() and "confcall" not in filename.lower():
        return False
    if filename in downloaded_set:
        return True
    save_path = os.path.join(folder, filename)
    try:
        print(f"  Downloading: {filename}")
        r = requests.get(url, headers=headers, timeout=15)
        with open(save_path, 'wb') as f:
            f.write(r.content)
        downloaded_set.add(filename)
        time.sleep(random.uniform(1, 2))
        return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """
    使用 pdfplumber 精确提取美联储 PDF 文本
    """
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                width = page.width
                height = page.height
                
                # 调整：稍微放宽顶部裁剪，因为有些年份的页眉很高
                # 只裁剪最上面的 15% 和最下面的 10%
                crop_box = (0, height * 0.15, width, height * 0.9)
                
                try:
                    cropped_page = page.crop(crop_box)
                    text = cropped_page.extract_text()
                except ValueError:
                    # 如果裁剪区域出错(比如页面太小)，就提取全页
                    text = page.extract_text()
                
                if text:
                    full_text += text + "\n"
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None
    return full_text

def clean_and_structure_transcript(text, date_year):
    """
    清洗文本并分割为 Opening Statement 和 Q&A
    """
    if not text:
        return []
    
    # 1. 强力清洗：去除 PDF 页眉页脚残留
    lines = text.split('\n')
    cleaned_lines = []
    
    # 定义需要剔除的噪音行 (Regex)
    # 比如 "March 21, 2018", "Chairman Powell's Press Conference", "FINAL", "Page 1 of 22"
    noise_patterns = [
        r"Page \d+ of \d+",            # 页码
        r"Chairman Powell'?s? Press Conference", # 标题
        r"Transcript of .* Press Conference",
        r"^FINAL$",                    # 状态标记
        r"^[A-B][a-z]+ \d{1,2}, 20\d{2}$", # 日期行 (e.g. March 21, 2018)
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.isdigit(): # 纯数字行
            continue
            
        # 检查是否是噪音行
        is_noise = False
        for pattern in noise_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                is_noise = True
                break
        
        if not is_noise:
            cleaned_lines.append(line)
        
    full_clean_text = " ".join(cleaned_lines)
    
    # 2. 分割 Opening Statement 和 Q&A
    # 扩展了分割关键词，覆盖 2018 年的措辞
    qa_start_markers = [
        "I will now take your questions", 
        "I'll be happy to take your questions", # 2018 specific
        "happy to take your questions",
        "happy to respond to your questions",
        "We will now take questions",
        "Q & A",
    ]
    
    split_index = -1
    for marker in qa_start_markers:
        # 使用 lower() 查找以忽略大小写差异
        idx = full_clean_text.lower().find(marker.lower())
        if idx != -1:
            split_index = idx
            break
    
    structured_data = []
    
    # 构造准确的日期 (从文件名或 PDF 内容解析会更准，这里简化用 Year-01-01)
    # 你可以在后续用 pandas 修正日期
    base_date = f"{date_year}-01-01"
    
    if split_index != -1:
        # 找到了分割点
        opening_statement = full_clean_text[:split_index].strip()
        qa_session = full_clean_text[split_index:].strip()
        
        structured_data.append({
            'date': base_date,
            'section': 'Press Conf - Opening Statement',
            'sentence_text': opening_statement
        })
        
        structured_data.append({
            'date': base_date,
            'section': 'Press Conf - Q&A',
            'sentence_text': qa_session
        })
    else:
        # 没找到分割点，保存全文
        structured_data.append({
            'date': base_date,
            'section': 'Press Conf - Full Text',
            'sentence_text': full_clean_text
        })
        
    return structured_data

def process_all_pdfs(pdf_folder, output_csv):
    """
    批量处理文件夹下的 PDF
    """
    all_rows = []
    
    if not os.path.exists(pdf_folder):
        print(f"Folder not found: {pdf_folder}")
        return

    files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    print(f"Found {len(files)} PDFs. Starting extraction...")
    
    for filename in files:
        print(f"Processing: {filename}")
        file_path = os.path.join(pdf_folder, filename)
        
        # 尝试从文件名提取具体日期 (YYYYMMDD)
        # 例如: FOMCpresconf20180321.pdf
        date_match = re.search(r'(20\d{2})(\d{2})(\d{2})', filename)
        if date_match:
            year, month, day = date_match.groups()
            formatted_date = f"{year}-{month}-{day}"
            date_year = formatted_date # 直接用完整日期
        else:
            # Fallback 到只取年份
            year_match = re.search(r'20\d{2}', filename)
            date_year = year_match.group(0) if year_match else "Unknown"
        
        # 1. 提取原始文本
        raw_text = extract_text_from_pdf(file_path)
        
        # 2. 结构化处理
        structured_segments = clean_and_structure_transcript(raw_text, date_year)
        
        all_rows.extend(structured_segments)
        
    # 保存结果
    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nProcessing Complete. Saved to {output_csv}")

if __name__ == "__main__":
    PDF_FOLDER = r"e:\Textming\Fed_Project\data\press_conf_pdfs"
    OUTPUT_CSV = r"e:\Textming\Fed_Project\data\fed_press_conf_structured.csv"
    
    # 1. 先下载 PDF
    download_press_conf_pdfs(PDF_FOLDER, start_year=2018, end_year=2024)
    
    # 2. 再处理 PDF
    process_all_pdfs(PDF_FOLDER, OUTPUT_CSV)