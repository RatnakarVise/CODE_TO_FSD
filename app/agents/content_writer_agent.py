import os
import json
import logging
from typing import Any, Dict, List
import openai
from dotenv import load_dotenv
import asyncio
# from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()
logger = logging.getLogger("content_writer_agent")
logging.basicConfig(level=logging.INFO)
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables.")

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
def consolidate_field_logic(field_logic_list):
    merged = {}
    for item in field_logic_list:
        fname = item.get("FIELD_NAME")
        flogic = item.get("FIELD_LOGIC", "").strip()

        if not fname:
            continue

        if fname not in merged:
            merged[fname] = flogic
        else:
            # Append only if new logic adds info
            if flogic not in merged[fname]:
                merged[fname] += " " + flogic

    return [{"FIELD_NAME": k, "FIELD_LOGIC": v} for k, v in merged.items()]

# =============================================================================
# 💡 HERE you define how to club which sections and what payload keys to use
# Each bundle is ( [section_name1, section_name2, ...], [payload_key1, payload_key2, ...])
# NOT clubbed sections can be left as ["Section"], ["payload_key1"] so they're handled individually.
SECTION_BUNDLES = [
    (["Output"], ['pgm_name', 'type', 'explanation']),
    (["Functional Requirements"], ['pgm_name', 'type', 'explanation']),
    (["Field Logic"], ['FIELD_LOGIC']),
    (["Document Information", "Introduction", "Business Requirement Overview", "Business Process Flow"], ['pgm_name','type', 'inc_name', 'explanation']),
    (["UI Requirement"], ["selectionscreen"]),
    (["Functional Scope"], ['pgm_name', 'type', 'explanation']),
    # (["Processing logic"], ['pgm_name', 'type', 'explanation']),
    (["Functional Solution Approach"], ['pgm_name', 'type', 'explanation']),
    (["Interfaces & Integration","Error Handling & Notifications", "Assumptions & Dependencies", "Authorization & Security"], ['selectionscreen', 'declarations', 'explanation']),
    (["Test Scenario"], ['selectionscreen', 'declarations', 'explanation']),
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
        # self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.llm = ChatOpenAI(
        model_name=self.model,
        temperature=0,
        verbose=True,
        api_key=openai_api_key
        )
        self.template_sections = load_sections_from_template(self.template_path)
        self.results = []

    # --- Synchronous run ---
    def run(self, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        if not payload:
            logger.error("No payload provided.")
            return [{"section_name": "ERROR", "content": "No payload provided"}]

                # 🔥 FIELD_LOGIC consolidation added here
        if "FIELD_LOGIC" in payload and isinstance(payload["FIELD_LOGIC"], list):
            payload["FIELD_LOGIC"] = consolidate_field_logic(payload["FIELD_LOGIC"])
            logger.info(f"FIELD_LOGIC consolidated to {len(payload['FIELD_LOGIC'])} unique fields")

        self.results = []
        handled_sections = set()

        for section_names, payload_keys in SECTION_BUNDLES:
            section_bibles = {s: fetch_bible_knowledge(self.template_sections, s) for s in section_names}
            sub_payload = filter_payload_by_keys(payload, payload_keys)
            logger.info(f"Generating content for sections: {section_names} with keys {payload_keys}")

            # Use synchronous version (you can adapt later if needed)
            section_texts = asyncio.run(self._generate_sections(section_names, section_bibles, sub_payload))

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

    # --- Public async run method ---
    async def run_async(self, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Async version of run() for use in FastAPI background tasks.
        """
        if not payload:
            logger.error("No payload provided.")
            return [{"section_name": "ERROR", "content": "No payload provided"}]

        self.results = []
        handled_sections = set()

                # 🔥 FIELD_LOGIC consolidation added here
        if "FIELD_LOGIC" in payload and isinstance(payload["FIELD_LOGIC"], list):
            payload["FIELD_LOGIC"] = consolidate_field_logic(payload["FIELD_LOGIC"])
            logger.info(f"FIELD_LOGIC consolidated to {len(payload['FIELD_LOGIC'])} unique fields")

        for section_names, payload_keys in SECTION_BUNDLES:
            section_bibles = {s: fetch_bible_knowledge(self.template_sections, s) for s in section_names}
            sub_payload = filter_payload_by_keys(payload, payload_keys)
            logger.info(f"[Async] Generating content for sections: {section_names} with keys {payload_keys}")

            section_texts = await self._generate_sections(section_names, section_bibles, sub_payload)

            for s in section_names:
                self.results.append({
                    "section_name": s,
                    "content": section_texts.get(s, f"[Error: Missing output for section {s}]")
                })
                handled_sections.add(s)

        # Guarantee template order
        template_order = [s["title"] for s in self.template_sections]
        ordered_results = []
        for section_name in template_order:
            matched = next((x for x in self.results if x["section_name"] == section_name), None)
            if matched:
                ordered_results.append(matched)
            else:
                ordered_results.append({"section_name": section_name, "content": "[ERROR: Section content missing]"})
        return ordered_results

    # --- Private async section generator ---
    # --- Private async section generator with LangChain / LangSmith ---
    async def _generate_sections(self, section_names, section_bibles, payload, max_retries=3) -> Dict[str, str]:
        """
        Generate each section’s content using LangChain ChatOpenAI (LangSmith tracing enabled),
        with automatic retries for missing sections. Returns dict {section_name: content}.
        """

        results = {}
        remaining_sections = section_names.copy()

        # Initialize LangChain LLM (tracing enabled)
        llm = ChatOpenAI(
            model_name=self.model,
            temperature=0,
            verbose=True,
            openai_api_key=openai_api_key
        )

        async def call_llm_for_sections(sections_subset):
            context_json = json.dumps(payload, indent=2)
            batched_prompt = (
                "You are an expert SAP ABAP functional specification writer.\n"
                "Srictly follow the section names as mentioned. Do not infere any other similar name\n"
                "Generate content for multiple SECTIONS of an SAP ABAP document.\n"
                "For each section:\n"
                "- Follow its 'BIBLE' strictly.\n"
                "- Use ONLY info from the JSON payload.\n"
                "- Output each section in format:\n"
                "<<START:{Section Name}>>\n<content>\n<<END:{Section Name}>>\n\n"
                f"Payload:\n```json\n{context_json}\n```\n"
            )

            for s in sections_subset:
                batched_prompt += f"\n====== SECTION: {s} ======\n"
                batched_prompt += f"---START BIBLE---\n{section_bibles.get(s, '')}\n---END BIBLE---\n"
                batched_prompt += f"Generate content for '{s}'. No titles or numbering.\n"
                batched_prompt += (
                        "\nIMPORTANT:\n"
                        "Output for each section *must* exactly follow this format:\n"
                        "<<START:{Section Name as given below}>>\n"
                        "<your content>\n"
                        "<<END:{Section Name as given below}>>\n"
                        "Do not modify section names (no plural/singular change, no rewording).\n"
                    )
                if s.strip().lower() == "flow diagram":
                    batched_prompt += (
                        "\nEXTRA INSTRUCTIONS:\n"
                        "1. Output one line: Step1 -> Step2 -> Step3.\n"
                        "2. No markdown, no code, no prose.\n"
                        "3. If none, output: Start -> No Relevant Logic -> End.\n"
                    )

            try:
                timeout_seconds = 600
                # Wrap the LLM call in asyncio.wait_for to enforce timeout
                response = await asyncio.wait_for(
                    llm.agenerate([[HumanMessage(content=batched_prompt)]]),
                    timeout=timeout_seconds
                )
                return response.generations[0][0].text.strip()
            except asyncio.TimeoutError:
                logger.warning(f"[Timeout] LLM call exceeded {timeout_seconds}s for sections: {sections_subset}")
                return ""  # empty string indicates timeout
            except Exception as e:
                logger.error(f"[Async Batch Error] {e}")
                return ""

        # --- Retry loop ---
        for attempt in range(1, max_retries + 1):
            if not remaining_sections:
                break

            logger.info(f"🌀 Attempt {attempt} for sections: {remaining_sections}")
            output = await call_llm_for_sections(remaining_sections)
            logger.debug(f"🧾 Raw LLM Output:\n{output[:1500]}")
            newly_completed = []
            for s in remaining_sections:
                start_tag, end_tag = f"<<START:{s}>>", f"<<END:{s}>>"
                if start_tag in output and end_tag in output:
                    content = output.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
                    results[s] = content
                    newly_completed.append(s)
                else:
                    logger.warning(f"[Attempt {attempt}] Missing section: {s}")

            remaining_sections = [s for s in remaining_sections if s not in newly_completed]
            if remaining_sections:
                await asyncio.sleep(1.5)

        # --- Fill missing after all retries ---
        for s in remaining_sections:
            results[s] = f"[Error: Section {s} not found after {max_retries} retries.]"

        return results