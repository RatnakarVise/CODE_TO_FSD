import os
import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app.agents.content_writer_agent import ContentWriterAgent
from app.doc.doc_constructor_agent import build_document
from app.doc.flow_diagram_agent import FlowDiagramAgent

app = FastAPI(title="ABAP Functional Spec Generator")

# In-memory store for job tracking (not for production use)
JOBS = {}

def generate_doc_background(payload, job_id):
    """
    Synchronous function to run in background: 
    Generates the document and updates the job status.
    """
    try:
        writer_agent = ContentWriterAgent()
        results = writer_agent.run(payload)

        sections = []
        for sec in writer_agent.template_sections:
            sec_type = "text"
            sec_title = sec["title"]
            if "diagram" in sec_title.lower():
                sec_type = "diagram"
            elif "| " in sec["content"]:
                sec_type = "table"
            sections.append({
                "title": sec_title,
                "type": sec_type
            })

        diagram_agent = FlowDiagramAgent()
        doc = build_document(results, sections, flow_diagram_agent=diagram_agent, diagram_dir="diagrams")

        output_filename = f"Functional_Spec_{job_id}.docx"
        output_path = os.path.abspath(output_filename)
        doc.save(output_path)

        JOBS[job_id]['status'] = "done"
        JOBS[job_id]['file_path'] = output_path
    except Exception as e:
        JOBS[job_id]['status'] = "failed"
        JOBS[job_id]['error'] = str(e)

@app.post("/generate_doc")
async def generate_doc(payload: dict, background_tasks: BackgroundTasks):
    """
    POST endpoint to start document generation.
    Returns job_id immediately; job runs in the background.
    """
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "pending", "file_path": None, "error": None}
    background_tasks.add_task(generate_doc_background, payload, job_id)
    return {"job_id": job_id, "status": "started"}

@app.get("/generate_doc/{job_id}")
async def get_doc(job_id: str):
    """
    GET endpoint to poll/check job status or download the file when ready.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Invalid job_id")
    if job["status"] == "pending":
        return {"status": "pending"}
    if job["status"] == "failed":
        return JSONResponse(status_code=500, content={"status": "failed", "error": job["error"]})
    if job["status"] == "done":
        return FileResponse(
            job["file_path"],
            filename=os.path.basename(job["file_path"]),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )