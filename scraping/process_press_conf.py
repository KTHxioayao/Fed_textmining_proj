import pdfplumber
import pandas as pd
import os
import re
import csv
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.utilities import (
    split_into_sentences_robust, 
    clean_common_noise, 
    is_noise_content, 
    preprocess_content
)

class FedPressConfProcessor:
    def __init__(self, data_folder):
        """
        初始化处理器。
        设定输入文件夹 (raw PDFs) 和输出路径 (processed CSV)。
        """
        # Auto-detect path if data_folder is generic, or trust input
        # Here we assume data_folder is correct root (e.g. e:\Textming\data)
        self.pdf_folder = os.path.join(data_folder, "raw", "press_conf_pdfs")
        self.output_csv = os.path.join(data_folder, "processed", "fed_press_conf_structured.csv")

    def process_pdfs(self):
        """
        【主控制流】
        遍历所有 PDF 文件，执行清洗、结构化分割，最后保存为 CSV。
        """
        print(f"\n[Processing PDFs] Processing files in {self.pdf_folder}...")
        all_rows = []
        if not os.path.exists(self.pdf_folder):
             print(f"Error: Folder {self.pdf_folder} does not exist.")
             return

        # 获取目录下所有 PDF 文件
        files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        
        for filename in files:
            print(f"  Processing: {filename}")
            file_path = os.path.join(self.pdf_folder, filename)
            
            # 1. 元数据提取：从文件名中解析出准确的日期 (YYYY-MM-DD)
            date_str = self._extract_date_from_filename(filename)
            
            # 2. 类型判断：文件名包含 "confcall" 说明是电话会议，处理逻辑不同
            is_conf_call = "confcall" in filename.lower()
            
            # 3. 文本提取：使用 pdfplumber 进行“物理裁剪”，去除页眉页脚的干扰
            raw_text = self._extract_text_with_crop(file_path)
            if not raw_text: continue
            
            # 4. 噪音清洗：使用正则去除 PDF 中残留的页码、标题、日期行
            clean_text = self._clean_noise(raw_text)
            
            # 5. 结构化分割与说话人提取 (核心逻辑)
            if is_conf_call:
                # 如果是电话会议：提取多方对话（因为没有 Q&A 环节）
                segments = self._process_conf_call(clean_text, date_str)
            else:
                # 如果是常规发布会：分割 Opening 和 Q&A，并只提取鲍威尔的发言
                segments = self._structure_press_conf(clean_text, date_str)
            
            all_rows.extend(segments)
            
        # 6. 数据保存
        df = pd.DataFrame(all_rows)
        if not df.empty:
            # 重新排列列顺序，确保 date 在最前面，方便阅读
            cols = ['date', 'section', 'text']
            for c in df.columns:
                if c not in cols:
                    cols.append(c)
            df = df[cols]
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        
            # 关键参数 quoting=csv.QUOTE_ALL：
            # 强制给所有文本加引号。防止因为文本里包含逗号(,)，导致 CSV 列错位。
            df.to_csv(self.output_csv, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            print(f"\n[Processing Complete] Saved {len(df)} segments to {self.output_csv}")
        else:
            print("\n[Processing Complete] No data found.")

    def _extract_date_from_filename(self, filename):
        """
        从文件名 (如 FOMCpresconf20230322.pdf) 提取日期。
        如果提取不到具体日期，则回退到提取年份。
        """
        match = re.search(r'(20\d{2})(\d{2})(\d{2})', filename)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        match_year = re.search(r'20\d{2}', filename)
        return f"{match_year.group(0)}-01-01" if match_year else "Unknown"

    def _extract_text_with_crop(self, pdf_path):
        """
        【物理去噪】
        PDF 每一页都有页眉（Page X of Y）和页脚（FINAL）。
        与其用正则删，不如直接按坐标裁剪掉页面顶部 15% 和底部 10%。
        """
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # 定义保留区域：Top 15% 到 Bottom 10% 之间的部分
                    crop_box = (0, page.height * 0.15, page.width, page.height * 0.9)
                    try:
                        text += page.crop(crop_box).extract_text() + "\n"
                    except:
                        # 如果裁剪失败（如页面太小），则提取整页作为兜底
                        text += page.extract_text() + "\n"
        except Exception as e:
            print(f"    Error reading PDF: {e}")
            return None
        return text

    def _clean_noise(self, text):
        """
        【正则去噪】
        即使裁剪了，有些位于正文区域的标题、日期、或 'FINAL' 标记仍可能残留。
        使用正则表达式将这些特定模式的行剔除。
        """
        # First use shared common noise cleaner
        text = clean_common_noise(text)
        
        # Then clean specific Press Conference artifacts
        lines = text.split('\n')
        cleaned = []
        patterns = [
            r"Chair(man)? Powell['']?s? Press Conference", # 标题变体
            r"Chair Powell['']?s? Press Conference Call" 
        ]
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            is_noise = False
            for pat in patterns:
                if re.search(pat, line, re.IGNORECASE):
                    is_noise = True
                    break
            if not is_noise:
                cleaned.append(line)
        return " ".join(cleaned)

    def _structure_press_conf(self, text, date_str):
        """
        【结构化分割】
        将发布会切分为 'Opening Statement' (开场白) 和 'Q&A' (问答)。
        """
        # 定义所有标志着 Q&A 环节开始的短语 (覆盖 2018-2024 所有变体)
        split_markers = [
            "I will now take your questions", 
            "I am happy to take your questions",
            "I'm happy to take your questions",
            "happy to take your questions", 
            "happy to respond to your questions",
            "We will now take questions",
            "Q & A",
            "MICHELLE SMITH. Thank you", # 主持人转场
            "MICHELLE SMITH. We will now go to",
            "MICHELLE SMITH. We'll go to",
            "MICHELLE SMITH. Let's go to",
            "MICHELLE SMITH. Our first question",
            "I'll be happy to take your questions",
            "I will be happy to take your questions",
            "we'll be happy to take your questions",
            "MICHELLE SMITH.", # 兜底策略：如果出现主持人名字，通常意味着 Q&A 开始
            "look forward to your questions",
            "I look forward to your questions",
            "I look forward to taking your questions",
            "I would be happy to take your questions"
        ]
        
        split_idx = -1
        text_lower = text.lower()
        
        # 寻找在文本中出现得 *最早* 的分割标记
        min_idx = float('inf')
        found_marker = None
        
        for marker in split_markers:
            idx = text_lower.find(marker.lower())
            if idx != -1 and idx < min_idx:
                min_idx = idx
                found_marker = marker
        
        if min_idx != float('inf'):
            split_idx = min_idx
            # print(f"DEBUG: Found earliest marker '{found_marker}' at {split_idx} for date {date_str}")
        else:
            print(f"DEBUG: No marker found for date {date_str}")
        
        segments = []
        if split_idx != -1:
            # 切分文本
            raw_opening = text[:split_idx]
            raw_qa = text[split_idx:]
            
            # --- 处理 Opening Statement ---
            # 调用 speaker_extraction 只提取 Powell 的发言 (去掉了前面的主持人介绍)
            op_sentences = self._extract_speaker_segments(raw_opening, is_qa=False)
            if isinstance(op_sentences, list):
                for sent in op_sentences:
                    segments.append({'date': date_str, 'section': 'Opening Statement', 'text': sent})
            elif op_sentences:
                segments.append({'date': date_str, 'section': 'Opening Statement', 'text': op_sentences})
                
            # --- 处理 Q&A ---
            # 关键：过滤掉记者的提问，只提取 Powell 的回答
            qa_sentences = self._extract_speaker_segments(raw_qa, is_qa=True)
            for sent in qa_sentences:
                segments.append({'date': date_str, 'section': 'Q&A', 'text': sent})
        else:
            # 如果没找到分割点，作为全文处理 (Full Text)
            full_sentences = self._extract_speaker_segments(text, is_qa=False)
            if isinstance(full_sentences, list):
                for sent in full_sentences:
                    segments.append({'date': date_str, 'section': 'Full Text', 'text': sent})
            elif full_sentences:
                segments.append({'date': date_str, 'section': 'Full Text', 'text': full_sentences})
        return segments

    def _process_conf_call(self, text, date_str):
        """
        处理电话会议 (Conference Call)。
        特点：没有 Q&A 环节，是 Powell 和其他委员的对话。
        策略：保留所有有效的发言内容。
        """
        segments = []
        # 正则：匹配 "全大写名字 + 标点" (如 MS. BRAINARD.)
        speaker_pattern = re.compile(r"(?:^|\s)([A-Z\s\.\''-]{3,})\s*[\.\:]")
        parts = speaker_pattern.split(text)
        
        if len(parts) >= 2:
            for i in range(1, len(parts), 2):
                speaker = parts[i].strip().upper()
                content = parts[i+1].strip()
                # 过滤 "Thank you", "You're on mute" 等无效对话
                if is_noise_content(content): continue
                # 过滤误判的标题 (如 "TRANSCRIPT OF")
                if "TRANSCRIPT OF" in speaker or "PAGE" in speaker: continue

                # 对发言内容进行句子分割，与其他部分保持一致
                if len(content.strip()) > 10:
                    # 预处理内容以获得更细粒度的分割
                    content_segments = preprocess_content(content)
                    for segment in content_segments:
                        if len(segment.strip()) > 10:
                            sentences = split_into_sentences_robust(segment)
                            for sentence in sentences:
                                segments.append({
                                    'date': date_str,
                                    'section': 'Conference Call',
                                    'speaker': speaker, # 保留说话人名字，因为可能有其他人
                                    'text': sentence
                                })
        return segments

    def _extract_speaker_segments(self, text, is_qa):
        """
        【说话人过滤器】(Speaker Filter)
        这是本脚本最核心的功能：只提取 Powell 的发言，丢弃记者的提问。
        """
        # 正则：寻找 "CHAIR POWELL." 或 "MR. POWELL:" 这样的标记
        pattern = re.compile(r"(?:^|\s)([A-Z\s\.\''-]{3,}|Transcript of the Federal Open Market Committee Conference Call)\s*[\.\:]")
        
        # split 后会得到 [垃圾, 名字, 内容, 名字, 内容...]
        parts = pattern.split(text)
        valid_segments = []
        
        if len(parts) < 2:
            # 如果没找到任何说话人标记（可能是格式异常），非 Q&A 时保留全文
            if not is_qa and text:
                return split_into_sentences_robust(text)
            return []

        for i in range(1, len(parts), 2):
            speaker = parts[i].strip().upper()
            content = parts[i+1].strip()
            
            # 【过滤器】：只有当说话人名字里包含 "POWELL" 时才保留
            # 这就自动过滤掉了所有 "MICHELLE SMITH", "NANCY MARSHALL-GENZER" 等其他人
            if "POWELL" in speaker or "TRANSCRIPT OF THE FEDERAL" in speaker:
                # 清洗可能残留的标题
                if "Transcript of" in content[:100]:
                     content = content.split("Transcript of")[0]
                
                if is_noise_content(content): continue
                
                # 将这一段发言切分成句子
                if len(content) > 10:
                    # 预处理：将内容按段落分割，以获得更细粒度的文本块
                    content_segments = preprocess_content(content)
                    for segment in content_segments:
                        if len(segment.strip()) > 10:
                            sentences = split_into_sentences_robust(segment)
                            valid_segments.extend(sentences)
        
        return valid_segments

if __name__ == "__main__":
    # Use relative path if running from scraping directory, or adapt
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming standard structure e:\Textming\scraping\process_press_conf.py
    # and data in e:\Textming\data
    DATA_ROOT = os.path.join(os.path.dirname(current_dir), "data")
    
    print(f"Data root: {DATA_ROOT}")
    processor = FedPressConfProcessor(DATA_ROOT)
    processor.process_pdfs()