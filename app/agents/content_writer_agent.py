import os
import json
import logging
from typing import Any, Dict, List
import openai
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

OPENAI_MODEL = "gpt-4.1"
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.txt")

def load_sections_from_template(template_file: str) -> list:
    sections = []
    with open(template_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    current_title, current_content = None, []
    for line in lines:
        line = line.rstrip()
        if line.startswith("#"):
            if current_title and current_content:
                sections.append({"title": current_title, "content": "\n".join(current_content).strip()})
            current_title = line.lstrip("#").strip()
            current_content = []
        elif current_title is not None:
            current_content.append(line)
    if current_title and current_content:
        sections.append({"title": current_title, "content": "\n".join(current_content).strip()})
    return sections

def filter_payload_by_keys(payload: Dict[str, Any], required_keys: List[str]) -> Dict[str, Any]:
    """Utility to subset a dict for only listed keys (if provided)."""
    if not required_keys:
        return payload
    return {k: payload[k] for k in required_keys if k in payload}

# =============================================================================
# 💡 HERE you define how to club which sections and what payload keys to use
# Each bundle is ( [section_name1, section_name2, ...], [payload_key1, payload_key2, ...])
# NOT clubbed sections can be left as ["Section"], ["payload_key1"] so they're handled individually.
SECTION_BUNDLES = [
    (["Document Information", "Introduction", "Business Requirement Overview", "Business Process Flow"], ['pgm_name','type', 'inc_name', 'explanation']),
    (["UI Requirement"], ["selectionscreen"]),
    (["Functional Scope"], ['pgm_name', 'type', 'explanation']),
    (["Functional Solution Approach"], ['pgm_name', 'type', 'explanation']),
    (["Output"], ['pgm_name', 'type', 'explanation']),
    (["Functional Requirements"], ['pgm_name', 'type', 'explanation']),
    (["Interfaces & Integration","Error Handling & Notifications", "Assumptions & Dependencies", "Authorization & Security"], [ 'selectionscreen', 'declarations', 'explanation']),
    (["Test Scenario"], [ 'selectionscreen', 'declarations', 'explanation']),
    # (["Transport Management"], ['transport']),
    (["Sign-Off"], []),
]

# =============================================================================

def fetch_bible_knowledge(section_lookup, section_name):
    """Find the BIBLE (template) for a given section name."""
    for section in section_lookup:
        if section["title"].strip().lower() == section_name.strip().lower():
            return section["content"]
    return ""

class ContentWriterAgent:
    """
    Generates each section or sections bundle based on template.txt and input payload.
    Each template section is the 'bible'. AI must adhere to it.
    Stores output as a table-style list of dicts: [{section_name, content}, ...]
    """
    def __init__(self, model=OPENAI_MODEL, template_path=TEMPLATE_PATH):
        self.model = model
        self.template_path = template_path
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.template_sections = load_sections_from_template(self.template_path)
        self.results = []  # Will hold each row: {"section_name": "<>", "content": "<>"}

    def run(self, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        if not payload:
            logger.error("No payload provided.")
            return [{"section_name": "ERROR", "content": "No payload provided"}]

        self.results = []

        handled_sections = set()
        for section_names, payload_keys in SECTION_BUNDLES:
            for s in section_names:
                if s in handled_sections:
                    continue
            section_bibles = {s: fetch_bible_knowledge(self.template_sections, s) for s in section_names}
            sub_payload = filter_payload_by_keys(payload, payload_keys)
            logger.info(f"Generating content for sections: {section_names} with keys {payload_keys}")

            section_texts = self.generate_sections(section_names, section_bibles, sub_payload)
            for s in section_names:
                self.results.append({
                    "section_name": s,
                    "content": section_texts.get(s, f"[Error: Missing output for section {s}]")
                })
                handled_sections.add(s)

        # Guarantee RAG/template order
        template_order = [s["title"] for s in self.template_sections]
        ordered_results = []
        for section_name in template_order:
            matched = next((x for x in self.results if x["section_name"] == section_name), None)
            if matched:
                ordered_results.append(matched)
            else:
                ordered_results.append({"section_name": section_name, "content": "[ERROR: Section content missing]"})
        return ordered_results

    def generate_sections(self, section_names, section_bibles, payload) -> Dict[str, str]:
        """
        Given a list of section names, their bible (prompt), and the relevant payload,
        ask LLM to generate them all in one go and split output into separate parts.
        Returns dict of {section_name: content}
        """
        # Compose a single prompt
        context_json = json.dumps(payload, indent=2)
        batched_prompt = "You are an expert SAP ABAP technical specification writer.\n"
        batched_prompt += "You will generate content for multiple SECTIONS of an SAP ABAP document in one response. For each section:\n"
        batched_prompt += "- Strictly follow its 'BIBLE' (authoritative knowledge) shown for that section\n"
        batched_prompt += "- Use ONLY the information in the provided payload JSON\n"
        batched_prompt += "- For each section, output as:\n<<START:{Section Name}>>\n<content>\n<<END:{Section Name}>>\n\n"
        batched_prompt += "Important: You must output every section, even if the content is empty or you have no information. Do NOT skip any sections.\n"
        batched_prompt += f"\nThe relevant context (payload) for all these sections is:\n```json\n{context_json}\n```\n"

        for idx, s in enumerate(section_names):
            batched_prompt += f"\n====== SECTION: {s} ======\n"
            batched_prompt += f"---START BIBLE---\n{section_bibles.get(s, '')}\n---END BIBLE---\n"
            batched_prompt += f"Generate the content for section titled '{s}'. Do NOT output any headings or numbers. Only the body/content as per BIBLE.\n\n"
            # Special handling (Flow Diagram, etc)
            if s.strip().lower() == "flow diagram":
                batched_prompt += (
                    "\nEXTRA INSTRUCTIONS FOR THIS SECTION:"
                    "\n1. Output ONLY a single line, listing all the process steps joined by '->', e.g.:"
                    "\n   Start -> Get Input -> Validate -> Save -> End"
                    "\n2. DO NOT return code, code block, bullets, text, prose, explanations, legend, markdown, or headings."
                    "\n3. Do NOT return 'Flow Diagram', 'Diagram:', 'mermaid', or anything else. Just the process flow line."
                    "\n4. Do NOT use markdown, do NOT use mermaid, ONLY reply one line: Step1 -> Step2 -> ... -> StepN"
                    "\n5. If there are NO relevant ABAP logic/process steps, output: 'Start -> No Relevant Logic -> End'."  
                    "\n6. - Format [Linear:  A -> B -> C ]  - [ Branching: A -> [Yes] B -> [No] C ] [ Multiple branches, separated by ';']"
                )


        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional SAP ABAP documentation expert."},
                    {"role": "user", "content": batched_prompt}
                ],
                temperature=0.1
                # max_tokens omitted!
            )
            output = response.choices[0].message.content.strip()
            # Now split output into sections
            # Split output into sections using delimiters
            result = {}
            for s in section_names:
                start_tag = f"<<START:{s}>>"
                end_tag = f"<<END:{s}>>"
                section_content = ""
                if start_tag in output and end_tag in output:
                    section_content = output.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
                else:
                    section_content = f"[Error: Section {s} not found in LLM output.]"
                result[s] = section_content
            return result
        except Exception as e:
            logger.error(f"AI generation failed for sections {section_names}: {e}")
            # Return all as error entries
            return {s: f"[Error: AI section generation failed for '{s}': {e}]" for s in section_names}