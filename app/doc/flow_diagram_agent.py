import io
import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def extract_arrow_flow(text):
    if not text:
        return ""
    for line in text.splitlines():
        stripped = line.strip("` ").strip()
        if "->" in stripped and not stripped.lower().startswith(('diagram', 'flow', 'legend', '#')):
            return stripped
    if "->" in text:
        return text.strip()
    return ""

def parse_flow_string(flow_str):
    if not flow_str:
        return []
    branch_flows = [b.strip() for b in re.split(r";|\n", flow_str) if b.strip()]
    flows = []
    for branch in branch_flows:
        steps = []
        for part in branch.split("->"):
            s = part.strip()
            if s:
                steps.append(s)
        if steps:
            flows.append(steps)
    return flows

def unique_nodes(flows):
    seen = []
    for flow in flows:
        for node in flow:
            if node not in seen:
                seen.append(node)
    return seen

def build_edges(flows):
    edges = []
    for flow in flows:
        for i in range(len(flow)-1):
            edges.append( (flow[i], flow[i+1]) )
    return edges

class FlowDiagramAgent:
    def run(self, content):
        if isinstance(content, dict):
            flow_desc = extract_arrow_flow(content.get("content", ""))
        elif isinstance(content, str):
            flow_desc = extract_arrow_flow(content)
        else:
            flow_desc = ""
        print("Flow Input for Diagram:", flow_desc)
        if not flow_desc.strip():
            flow_desc = "Start -> Step1 -> Step2 -> End"
        flows = parse_flow_string(flow_desc)
        if not flows:
            flows = [["Start", "Diagram Not Provided", "End"]]
        nodes = unique_nodes(flows)
        edges = build_edges(flows)
        node_positions = {}
        y_base = 0
        x_col = 0
        node_idx_map = { n:i for i, n in enumerate(nodes) }
        for i, node in enumerate(nodes):
            node_positions[node] = (0, -i*1.5)
        if len(flows) > 1:
            min_len = min(len(flow) for flow in flows)
            common = []
            for j in range(min_len):
                s = set(flow[j] for flow in flows)
                if len(s) == 1:
                    common.append(flows[0][j])
                else:
                    break
            split_at = len(common)
            for b, flow in enumerate(flows):
                for j, node in enumerate(flow):
                    if j < split_at:
                        node_positions[node] = (0, -j*1.5)
                    else:
                        spread = max(1, len(flows))
                        offset = b - (len(flows)-1)/2
                        node_positions[node] = (offset * 3, -(split_at+b+(j-split_at))*1.5)
        fig, ax = plt.subplots(figsize=(6.5, max(3.0, 1.5 * len(nodes))) )
        ax.axis("off")
        box_width = 2.5
        box_height = 0.9
        for node, (x, y) in node_positions.items():
            rect = mpatches.FancyBboxPatch(
                (x - box_width/2, y - box_height/2), box_width, box_height,
                boxstyle="round,pad=0.16",
                linewidth=1.8, facecolor="#e2f0fa", edgecolor="#4882ab", zorder=10,
            )
            ax.add_patch(rect)
            ax.text(x, y, node, ha='center', va='center', fontsize=13, zorder=15, wrap=True)
        for src, tgt in edges:
            x0, y0 = node_positions[src]
            x1, y1 = node_positions[tgt]
            ax.annotate('',
                        xy=(x1, y1+box_height/2 if y1 < y0 else y1-box_height/2),
                        xytext=(x0, y0-box_height/2 if y1 < y0 else y0+box_height/2),
                        arrowprops=dict(arrowstyle="->", lw=2, color="#53676b", shrinkA=7, shrinkB=7),
            )
        margin = 1.8
        xs = [xy[0] for xy in node_positions.values()]
        ys = [xy[1] for xy in node_positions.values()]
        ax.set_xlim(min(xs)-box_width-margin, max(xs)+box_width+margin)
        ax.set_ylim(min(ys)-box_height-margin, max(ys)+box_height+margin)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf