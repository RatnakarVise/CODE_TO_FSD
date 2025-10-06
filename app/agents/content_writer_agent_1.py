import os
import json
import logging
from typing import Any, Dict, List, Tuple, Optional
from openai import OpenAI
import re
import time
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger("content_writer_agent")
logging.basicConfig(level=logging.INFO)

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = "gpt-5"
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.txt")

# If some bundles are too large we can split them automatically (tweak as needed)
MAX_SECTIONS_PER_BATCH = int(os.getenv("MAX_SECTIONS_PER_BATCH", 4))

# ----------------- Utilities -----------------
def load_sections_from_template(template_file: str) -> List[Dict[str, str]]:
    """Load #Section blocks from a text template file into [{'title','content'}, ...]."""
    sections: List[Dict[str, str]] = []
    if not os.path.exists(template_file):
        logger.warning("Template file not found: %s", template_file)
        return sections
    with open(template_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    current_title: Optional[str] = None
    current_content: List[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if line.strip().startswith("#"):
            # new section
            if current_title is not None:
                sections.append({"title": current_title.strip(), "content": "\n".join(current_content).strip()})
            current_title = line.lstrip("#").strip()
            current_content = []
        else:
            if current_title is not None:
                current_content.append(line)
    if current_title is not None:
        sections.append({"title": current_title.strip(), "content": "\n".join(current_content).strip()})
    return sections

def filter_payload_by_keys(payload: Dict[str, Any], required_keys: List[str]) -> Dict[str, Any]:
    """Return a subset of payload that contains only keys listed in required_keys (if present)."""
    if not required_keys:
        return payload.copy()
    return {k: payload[k] for k in required_keys if k in payload}

def make_placeholder_for_table(fields: List[str]) -> str:
    """Return a simple markdown table with one To Be Filled row for given fields."""
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join(["---"] * len(fields)) + " |"
    row = "| " + " | ".join(["[To Be Filled]"] * len(fields)) + " |"
    return "\n".join([header, sep, row])

def safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

# ----------------- Section Bundles (same logic as your original version) -----------------
SECTION_BUNDLES: List[Tuple[List[str], List[str]]] = [
    (["Document Information", "Introduction", "Business Requirement Overview", "Business Process Flow"], ['pgm_name','type', 'inc_name', 'explanation']),
    (["UI Requirement"], ["selectionscreen"]),
    (["Functional Scope"], ['pgm_name', 'type', 'explanation']),
    (["Functional Solution Approach"], ['pgm_name', 'type', 'explanation']),
    (["Output"], ['pgm_name', 'type', 'explanation']),
    (["Functional Requirements"], ['pgm_name', 'type', 'explanation']),
    (["Interfaces & Integration","Error Handling & Notifications", "Assumptions & Dependencies", "Authorization & Security"], ['selectionscreen', 'declarations', 'explanation']),
    (["Test Scenario"], ['selectionscreen', 'declarations', 'explanation']),
    (["Sign-Off"], []),
]

# ----------------- Agent -----------------
class ContentWriterAgent:
    def __init__(self, model: str = OPENAI_MODEL, template_path: str = TEMPLATE_PATH):
        self.model = model
        self.template_sections = load_sections_from_template(template_path)
        self.results: List[Dict[str, str]] = []
        logger.info("Loaded %d template sections from %s", len(self.template_sections), template_path)

    def run(self, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        """Main entry: generate content for all template sections in order and return list of dicts."""
        if not payload:
            logger.error("No payload provided to ContentWriterAgent.run()")
            return [{"section_name": "ERROR", "content": "No payload provided"}]

        # Build a map for quick bible lookup
        bible_map = {s["title"].strip().lower(): s["content"] for s in self.template_sections}

        # Prepare results list
        self.results = []
        handled_sections = set()

        # Iterate bundles; generate in batches (split large bundles)
        for section_names, payload_keys in SECTION_BUNDLES:
            # Avoid duplicates if same section appears again
            section_names_to_process = [s for s in section_names if s not in handled_sections]

            # split into smaller batches to avoid truncation
            batches = [section_names_to_process[i:i+MAX_SECTIONS_PER_BATCH] for i in range(0, len(section_names_to_process), MAX_SECTIONS_PER_BATCH)] or [[]]
            for batch in batches:
                if not batch:
                    continue
                sub_payload = filter_payload_by_keys(payload, payload_keys)

                logger.info("Generating batch for sections: %s (payload keys: %s)", batch, payload_keys)
                section_texts = self._generate_sections_batch(batch, {s: bible_map.get(s.strip().lower(), "") for s in batch}, sub_payload)

                for s in batch:
                    content = section_texts.get(s)
                    if not content:
                        content = f"[Error: Missing output for section {s}]"
                    # safe fallback for table-like sections named "Functional Requirements" etc
                    if s.strip().lower() == "functional requirements" and (not content or content.strip() == ""):
                        # make sure we have a table placeholder if model returned nothing
                        content = make_placeholder_for_table(["Requirement ID", "Requirement Description", "Business Rule", "Priority", "Comments"])
                    self.results.append({"section_name": s, "content": content})
                    handled_sections.add(s)

        # Ensure final ordering follows the template file order
        template_order = [s["title"] for s in self.template_sections]
        ordered_results: List[Dict[str, str]] = []
        for section_name in template_order:
            matched = next((x for x in self.results if x["section_name"].strip().lower() == section_name.strip().lower()), None)
            if matched:
                ordered_results.append(matched)
            else:
                ordered_results.append({"section_name": section_name, "content": "[ERROR: Section content missing]"})
        return ordered_results

    def _generate_sections_batch(self, section_names: List[str], section_bibles: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate content for a batch of sections in a single LLM call.
        The LLM is required to wrap each section between exact tokens:
        <<START:Section Name>>
        ...
        <<END:Section Name>>
        """
        # Build prompt
        payload_json = safe_json_dumps(payload)
        prompt_parts: List[str] = []
        prompt_parts.append("You are a professional SAP ABAP specification writer.")
        prompt_parts.append("You will produce content for multiple SECTIONS. For each section you MUST output the content between exact delimiters.")
        prompt_parts.append("Use the exact delimiters (case-sensitive) for each section as shown below:")
        for s in section_names:
            prompt_parts.append(f"<<START:{s}>>")
            prompt_parts.append(f"(section content for {s})")
            prompt_parts.append(f"<<END:{s}>>\n")
        prompt_parts.append("Important rules:")
        prompt_parts.append("1) You MUST output every requested section. If you have no information, still output the delimiters and inside write '[To Be Filled]'.")
        prompt_parts.append("2) Do not output additional headings, explanation text, or extraneous metadata outside the marked sections.")
        prompt_parts.append("3) For table sections, output a Markdown table. If no rows exist, put a single row with '[To Be Filled]'.")
        prompt_parts.append("\nSECTION BIBLES (authoritative guidance) follow:\n")
        for s in section_names:
            bible = section_bibles.get(s, "")
            if bible:
                prompt_parts.append(f"--- BIBLE for {s} ---\n{bible}\n")
            else:
                prompt_parts.append(f"--- BIBLE for {s} ---\n[No bible text provided]\n")

        prompt_parts.append("\nHere is the payload JSON (use ONLY this data to infer content):")
        prompt_parts.append("```json\n" + payload_json + "\n```\n")

        # Add explicit per-section instruction (helps LLM not to skip)
        for s in section_names:
            if s.strip().lower() == "functional requirements":
                prompt_parts.append(
                    "EXTRA: For 'Functional Requirements' create a Markdown table with columns: "
                    "[Requirement ID, Requirement Description, Business Rule, Priority, Comments]. "
                    "Number Requirement ID as REQ-001, REQ-002,... Use payload fields selectionscreen/declarations/explanation to derive rows."
                    "If nothing found, still produce a table and one row with [To Be Filled]."
                )
            if s.strip().lower() == "business process flow":
                prompt_parts.append(
                    "EXTRA: For Business Process Flow output a single line: Step1 -> Step2 -> ... -> End. If branching, indicate branches as: A -> [Yes] B -> [No] C."
                )

        batched_prompt = "\n\n".join(prompt_parts)

        # Debug log the prompt length (do not log extremely long payloads in production)
        logger.debug("Generated batched prompt (length %d chars) for sections: %s", len(batched_prompt), section_names)

        # Call OpenAI (simple wrapper with retry)
        raw_output = self._call_openai_with_retries(batched_prompt)

        # Log raw output for debugging (very helpful)
        logger.info("\n\n==== RAW LLM OUTPUT START ====\n%s\n==== RAW LLM OUTPUT END ====\n\n", raw_output)

        # Parse output: case-insensitive search for tags, but capture original-case content slice
        result: Dict[str, str] = {}
        lower_output = raw_output.lower()

        for s in section_names:
            start_tag_lower = f"<<start:{s.lower()}>>"
            end_tag_lower = f"<<end:{s.lower()}>>"
            if start_tag_lower in lower_output and end_tag_lower in lower_output:
                # find indices in lower_output, then slice original using those indices
                start_idx = lower_output.index(start_tag_lower) + len(start_tag_lower)
                end_idx = lower_output.index(end_tag_lower)
                # slice from raw_output to preserve original formatting
                section_content = raw_output[start_idx:end_idx].strip()
                # clean up surrounding whitespace and common wrapping code fences
                section_content = section_content.strip("\n ").strip()
                # If nothing inside, put placeholder
                if not section_content:
                    section_content = "[To Be Filled]"
                result[s] = section_content
            else:
                # Not found: try a tolerant regex search for tags with optional spaces
                pattern = re.compile(rf"<<\s*start\s*:\s*{re.escape(s)}\s*>>(.*?)<<\s*end\s*:\s*{re.escape(s)}\s*>>", re.IGNORECASE | re.DOTALL)
                m = pattern.search(raw_output)
                if m:
                    content = m.group(1).strip()
                    result[s] = content if content else "[To Be Filled]"
                else:
                    # as last resort, mark error. Upstream will handle placeholders for certain sections.
                    logger.warning("Section '%s' not found in LLM output.", s)
                    result[s] = ""  # empty — higher-level logic will add table placeholder for Functional Requirements
        return result

    def _call_openai_with_retries(self, prompt: str, retries: int = 2, backoff: float = 1.0) -> str:
        """Call OpenAI (new >=1.0 client) and return text, with retry logic."""
        for attempt in range(retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a professional SAP ABAP documentation expert."},
                        {"role": "user", "content": prompt}
                    ],
                    # temperature=0.05,
                    max_completion_tokens=3500,  # new param name
                )

                text = response.choices[0].message.content.strip()
                return text

            except Exception as e:
                logger.exception("OpenAI call failed (attempt %d/%d): %s", attempt + 1, retries + 1, e)
                if attempt < retries:
                    time.sleep(backoff * (2 ** attempt))
                    continue
                raise


# ----------------- If run as script: simple demo -----------------
if __name__ == "__main__":
    agent = ContentWriterAgent()
    demo_payload = {
        "pgm_name": "ZMM_PO_CLOSE",
        "type": "Report",
        "explanation": "Utility to close open POs. Selection screen includes S_MATNR,S_WERKS,S_LIFNR,S_MTART. Output is ALV grid with close button.",
        "selectionscreen": "S_MATNR,S_WERKS,S_LIFNR,S_MTART",
        "declarations": "Tables: EKKO, EKPO, EKET",
        "inc_name": "Caparo",
    }
    res = agent.run(demo_payload)
    for r in res:
        print("----", r["section_name"], "----")
        print(r["content"][:1000])
        print()
