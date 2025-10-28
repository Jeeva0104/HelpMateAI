import pdfplumber
import pandas as pd
from pathlib import Path
from operator import itemgetter
import json
from typing import List, Tuple, Dict, Any
from app.utils.logger import get_logger

logger = get_logger(__name__)

def check_bboxes(word: Dict[str, Any], table_bbox: Tuple[float, float, float, float]) -> bool:
    """Check whether word is inside a table bbox."""
    l = word["x0"], word["top"], word["x1"], word["bottom"]
    r = table_bbox
    return l[0] > r[0] and l[1] > r[1] and l[2] < r[2] and l[3] < r[3]

def extract_text_from_pdf(pdf_path: str) -> List[List[str]]:
    """
    Extract text from a PDF file, handling tables and regular text separately.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of [page_number, extracted_text] pairs
    """
    logger.info(f"Processing PDF: {pdf_path}")
    
    p = 0
    full_text = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_no = f"Page {p+1}"
                
                # Extract tables
                tables = page.find_tables()
                table_bboxes = [i.bbox for i in tables]
                tables = [{"table": i.extract(), "top": i.bbox[1]} for i in tables]
                
                # Extract non-table words
                non_table_words = [
                    word
                    for word in page.extract_words()
                    if not any(
                        [check_bboxes(word, table_bbox) for table_bbox in table_bboxes]
                    )
                ]
                
                lines = []
                
                # Cluster objects by vertical position
                for cluster in pdfplumber.utils.cluster_objects(
                    non_table_words + tables, itemgetter("top"), tolerance=5
                ):
                    if "text" in cluster[0]:
                        try:
                            lines.append(" ".join([i["text"] for i in cluster]))
                        except KeyError:
                            pass
                    elif "table" in cluster[0]:
                        lines.append(json.dumps(cluster[0]["table"]))

                full_text.append([page_no, " ".join(lines)])
                p += 1

        logger.info(f"Successfully extracted {len(full_text)} pages")
        return full_text
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        raise

def process_documents(pdf_path: str) -> pd.DataFrame:
    """
    Process documents and return a filtered DataFrame with metadata.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        DataFrame with processed document data
    """
    # Extract text from PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Convert to DataFrame
    df = pd.DataFrame(extracted_text, columns=["Page No.", "Page_Text"])
    df["Document Name"] = Path(pdf_path).name
    
    logger.info(f"Extracted {len(df)} pages from document")
    
    # Filter out empty pages (less than 10 words)
    df["Text_Length"] = df["Page_Text"].apply(lambda x: len(x.split(" ")))
    df = df.loc[df["Text_Length"] >= 10]
    
    logger.info(f"After filtering: {len(df)} pages with meaningful content")
    
    # Add metadata
    df["Metadata"] = df.apply(
        lambda x: {
            "Policy_Name": "Principal-Sample-Life-Insurance-Policy",
            "Page_No.": x["Page No."],
        },
        axis=1,
    )
    
    return df
