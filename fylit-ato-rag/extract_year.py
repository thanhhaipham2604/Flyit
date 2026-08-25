import json
import re

def extract_financial_year(doc):
    """
    Applies the precedence chain to extract the Financial Year:
    1. Rule 1: URL year
    2. Rule 2: Explicit applicability statement
    3. Rule 3: Heading / Title
    4. Rule 4: Null (leave null rather than guess)
    """
    url = doc.get("source_url") or ""
    title = doc.get("title") or ""
    content = doc.get("cleaned_content") or ""

    # Pattern 1: Year range (e.g., 2024-25, 2024-2025, 2024_25)
    range_pattern = r'\b(20\d{2})[-_](?:20)?(\d{2})\b'
    # Pattern 2: Single 4-digit year (e.g., 2025)
    single_pattern = r'\b(20\d{2})\b'

    # ------------------------------------------------------------------------
    # RULE 1: URL year
    # ------------------------------------------------------------------------
    match_range = re.search(range_pattern, url)
    if match_range:
        fy = f"{match_range.group(1)}-{match_range.group(2)}"
        return fy, "url_year"
    
    match_single = re.search(single_pattern, url)
    if match_single:
        return match_single.group(1), "url_year"

    # ------------------------------------------------------------------------
    # RULE 2: Explicit applicability statement (in content)
    # E.g., "applies to the 2024-25 financial year", "effective for 2025"
    # ------------------------------------------------------------------------
    app_pattern = r'(?:applies|apply|applicable|effective)\s+(?:from|to|for)?\s*(?:the)?\s*(?:income|tax|financial)?\s*(?:year)?\s*\b(20\d{2}(?:[-_](?:20)?\d{2})?)\b'
    match_app = re.search(app_pattern, content, re.IGNORECASE)
    if match_app:
        raw_fy = match_app.group(1)
        fy = re.sub(r'[_]', '-', raw_fy)
        return fy, "explicit_applicability_statement"

    # ------------------------------------------------------------------------
    # RULE 3: Heading / Title
    # ------------------------------------------------------------------------
    match_title_range = re.search(range_pattern, title)
    if match_title_range:
        fy = f"{match_title_range.group(1)}-{match_title_range.group(2)}"
        return fy, "heading"

    match_title_single = re.search(single_pattern, title)
    if match_title_single:
        return match_title_single.group(1), "heading"

    # ------------------------------------------------------------------------
    # RULE 4: Null (no guessing)
    # ------------------------------------------------------------------------
    return None, None


def process_corpus(input_path, output_path):
    """
    Reads cleaned_corpus.jsonl, extracts the requested Step 3 columns,
    and appends financial_year along with fy_source.
    """
    processed_count = 0
    
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
                
            doc = json.loads(line)
            
            # Run the precedence chain
            fy, fy_source = extract_financial_year(doc)
            
            # Map exact Step 3 schema requirements
            row = {
                "doc_id": doc.get("id"),
                "source_url": doc.get("source_url"),
                "title": doc.get("title"),
                "menu_path": doc.get("menu_path_text"),
                "audience": doc.get("category"),
                "qc_code": doc.get("qc_code", None),
                "last_updated": doc.get("last_updated_display"),
                "scraped_at": doc.get("scraped_at"),
                "content_hash": doc.get("content_hash"),
                "financial_year": fy,
                "fy_source": fy_source
            }
            
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            processed_count += 1

    print(f"Processing complete: {processed_count} documents processed.")
    print(f"Saved output to: {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "data/processed/cleaned_corpus.jsonl"
    OUTPUT_FILE = "data/processed/documents_tagged.jsonl"
    
    process_corpus(INPUT_FILE, OUTPUT_FILE)