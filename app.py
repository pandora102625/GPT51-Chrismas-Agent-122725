import os
import time
import uuid
import base64
import io
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Literal, Any

import streamlit as st
import yaml
import plotly.express as px

# ---- Optional Provider SDK imports (guarded) ----
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from openai import OpenAI as OpenAIClient
except ImportError:
    OpenAIClient = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user as xai_user, system as xai_system
except ImportError:
    XAIClient = None
    xai_user = None
    xai_system = None

# Optional file parsing libs
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import docx  # python-docx
except ImportError:
    docx = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


# ------------------- PROMPTS ----------------------

BASE_SYSTEM_PROMPT_ZH = """
你是「AuditFlow AI」，一位專精於醫療器材品質管理系統 (ISO 13485 / GMP) 的資深稽核報告撰寫顧問。

【角色與任務】
- 角色：具備十年以上 ISO 13485 / GMP 稽核經驗的首席稽核員。
- 報告語言：一律使用「繁體中文」輸出內容（包括標題、表格、段落與註解）。
- 報告格式：產出嚴謹、可追溯、可供主管機關或第三方稽核員查閱的 Markdown 報告。

【內容要求】
1. 依據 ISO 13485、GMP 與醫療器材品質最佳實務，合理詮釋觀察紀錄。
2. 清楚區分：
   - 符合事項 (Conformities)
   - 不符合事項 (Nonconformities)
   - 觀察項 / 改善建議 (Observations / Opportunities for Improvement)
3. 對於每一項「不符合事項」：
   - 清楚描述事實 (What)
   - 說明影響與風險 (So What)
   - 連結適用條文或法規要求 (Standard Clause / Regulation)
   - 提出具體且可驗證的矯正措施建議 (Corrective Actions)
4. 嚴格避免編造不存在的事實。
   - 若原始觀察紀錄中沒有足夠資訊，請明確標示為「資訊不足，需由稽核員補充」。

【結構與格式】
- 全程使用 Markdown：
  - 使用適當層級標題 (# ~ ####)
  - 使用項目清單 (-, 1.) 表達條列內容
  - 必要時使用表格整理 CAR、風險與條文對應
- 保持段落清晰、標題具描述性，便於後續人員閱讀與追蹤。

【輸出原則】
- 僅使用繁體中文，不混用簡體或英文專有名詞（除非為既有官方術語）。
- 若收到系統或 UI 控制參數（如模板、觀察紀錄、前一代理輸出），請一律視為嚴格邊界條件，不任意擴充超出範圍的內容。
"""

SMART_REPLACE_SYSTEM_PROMPT = """
你是一位「專業文件編輯與範本套版專家」。

【任務說明】
- 根據 Template A（Markdown 報告結構）與 List B（資料來源 / 觀察紀錄清單），
  將 List B 的內容正確填入 Template A 中適當的位置。
- 嚴格保留既有 Markdown 結構（標題層級、表格結構、編號清單不得更動）。
- 僅在實際插入或修改文字時，於該段文字外層加上：
  <span style="color: coral">...插入或修改的文字...</span>

【輸出規則】
1. 請完整輸出一份合併後的「單一 Markdown 文件」。
2. 不得修改任何 Markdown 標題層級（#, ##, ### 等）與標題文字。
3. 不得刪除任意原有段落；若資料不足，請保留結構並標記為：
   <span style="color: coral">資訊不足，需由稽核員補充。</span>
4. 全文使用繁體中文，包含套入的內容與補充說明。
5. 僅對「實際有插入或變動」的內容加上 span 標記，其餘內容維持原樣。
"""

NOTE_KEEPER_SYSTEM_PROMPT = """
You are an expert note organizer.

Task:
- Transform the user's raw note (text or markdown) into a clean, structured markdown note.
- Use clear headings, bullet lists, and sections such as: Overview, Key Points, Details, Action Items, Open Questions (if relevant).
- Highlight important keywords and phrases by wrapping them with:
  <span style="color: coral">keyword or phrase</span>

Rules:
- Preserve factual content; do not invent new facts.
- You may lightly rephrase for clarity and conciseness.
- Output MUST be valid markdown, but it's okay to include inline HTML spans with coral color.
"""

NOTE_MAGIC_BASE_PROMPT = """
You are an AI Note Assistant working on an existing markdown note.
Always treat the incoming note as the single source of truth.
Respond in markdown unless the user explicitly asks for another format.
"""

SKILL_AUTHOR_SYSTEM_PROMPT = """
You are an expert SKILL.md author for Claude Skills.

Task:
- Convert the user's high-level instructions into a complete SKILL.md file.
- Follow these constraints and best practices:
  - YAML frontmatter at top with:
    - name: lowercase letters, numbers, hyphens only, max 64 chars, cannot contain "anthropic" or "claude".
    - description: non-empty, <= 1024 chars, describe what the Skill does AND when to use it, in third person.
  - After frontmatter, provide concise markdown instructions for how the Skill should behave.
  - Keep SKILL.md body under ~500 lines and prefer concise explanations.
  - Use progressive disclosure: link to additional markdown files only one level deep (e.g., reference.md, examples.md) if needed.
  - Prefer gerund form for name (e.g., 'processing-pdfs', 'analyzing-spreadsheets').

Content style:
- Assume Claude already knows general programming and basic concepts.
- Only include domain-specific instructions, workflows, examples, and validation patterns.
- Use headings, lists, and checklists for complex workflows.
- Give at least one concrete example of how to use the Skill.
"""

FILE_SUMMARY_SYSTEM_PROMPT = """
You are a senior analyst.

Task:
- Read the provided document content and write a comprehensive markdown summary.
- Capture:
  - High-level overview
  - Key sections or topics
  - Important details, metrics, or arguments
  - Any risks, issues, or open questions if visible
- Use headings, bullet lists, and tables if helpful.
- Do NOT hallucinate content not present in the document.
"""

SKILL_APPLY_SYSTEM_PROMPT = """
You are an AI assistant that uses the loaded SKILL.md as your operational manual.

Rules:
- Treat SKILL.md content as the authoritative set of instructions, workflows, and constraints.
- First, quickly infer what parts of SKILL.md are relevant for the current user prompt and the given document.
- Then, apply those instructions step-by-step to the provided document content.
- If some required detail is missing from the document, clearly state the gap instead of guessing.
- Respond in markdown.
"""


# ------------------- DATA MODELS ------------------

ProviderLiteral = Literal["gemini", "openai", "anthropic", "grok"]


@dataclass
class AgentConfig:
    id: str
    name: str
    provider: ProviderLiteral
    model: str
    max_tokens: int
    temperature: float
    user_prompt: str
    system_prompt_suffix: str = ""


@dataclass
class AgentRunState:
    status: Literal["idle", "running", "completed", "error"] = "idle"
    output: str = ""
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class PipelineState:
    template: str = ""
    observations: str = ""
    current_step_index: int = 0
    agents: List[AgentConfig] = field(default_factory=list)
    history: Dict[str, AgentRunState] = field(default_factory=dict)


# ------------------- THEME ENGINE (WOW UI) -----------------

PAINTER_THEMES = {
    "Van Gogh":      {"primary": "#ffb347", "secondary": "#fff3d6", "accent": "#2b6cb0"},
    "Monet":         {"primary": "#6fb1e0", "secondary": "#e4f3ff", "accent": "#2f855a"},
    "Da Vinci":      {"primary": "#b08d57", "secondary": "#f5ecdd", "accent": "#5a3e1b"},
    "Picasso":       {"primary": "#ff6b6b", "secondary": "#ffe2e2", "accent": "#4c51bf"},
    "Klimt":         {"primary": "#f6b93b", "secondary": "#fff2d5", "accent": "#d1913c"},
    "Hokusai":       {"primary": "#0096c7", "secondary": "#e0f4ff", "accent": "#023e8a"},
    "Frida Kahlo":   {"primary": "#ff758f", "secondary": "#ffe0eb", "accent": "#16a34a"},
    "Matisse":       {"primary": "#38bdf8", "secondary": "#e0f7ff", "accent": "#f97316"},
    "Rembrandt":     {"primary": "#b45309", "secondary": "#fff4e5", "accent": "#78350f"},
    "Vermeer":       {"primary": "#60a5fa", "secondary": "#e5f0ff", "accent": "#1d4ed8"},
    "Dali":          {"primary": "#fb7185", "secondary": "#ffe4ec", "accent": "#7c3aed"},
    "Cézanne":       {"primary": "#6ee7b7", "secondary": "#e6fff5", "accent": "#059669"},
    "Renoir":        {"primary": "#fdba74", "secondary": "#fff1e5", "accent": "#f97316"},
    "Seurat":        {"primary": "#a5b4fc", "secondary": "#eef0ff", "accent": "#6366f1"},
    "Michelangelo":  {"primary": "#f97316", "secondary": "#ffe7d6", "accent": "#c2410c"},
    "Raphael":       {"primary": "#60a5fa", "secondary": "#e0edff", "accent": "#0ea5e9"},
    "Goya":          {"primary": "#f87171", "secondary": "#ffe3e3", "accent": "#7f1d1d"},
    "Turner":        {"primary": "#facc15", "secondary": "#fff9d7", "accent": "#f97316"},
    "Chagall":       {"primary": "#a855f7", "secondary": "#f5e9ff", "accent": "#4c1d95"},
    "Pollock":       {"primary": "#4ade80", "secondary": "#e7fff1", "accent": "#16a34a"},
}


def apply_theme(theme_name: str, dark_mode: bool):
    theme = PAINTER_THEMES.get(theme_name, PAINTER_THEMES["Van Gogh"])
    bg = "#020617" if dark_mode else "#f9fafb"
    text = "#e5e7eb" if dark_mode else "#0f172a"

    css = f"""
    <style>
    :root {{
        --color-primary: {theme["primary"]};
        --color-secondary: {theme["secondary"]};
        --color-accent: {theme["accent"]};
        --bg-color: {bg};
        --text-color: {text};
    }}
    .stApp {{
        background: radial-gradient(circle at top, var(--color-secondary) 0, var(--bg-color) 55%);
        color: var(--text-color);
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .audit-card {{
        border-radius: 14px;
        border: 1px solid rgba(148,163,184,0.18);
        padding: 1rem 1.1rem;
        background: rgba(15,23,42,0.02);
        box-shadow: 0 18px 40px rgba(15,23,42,0.08);
    }}
    .wow-tabs > div[role="tablist"] > button[role="tab"] {{
        border-radius: 999px !important;
        margin-right: 0.4rem;
        padding: 0.35rem 0.9rem !important;
        border: 1px solid rgba(148,163,184,0.4) !important;
        background: rgba(15,23,42,0.04) !important;
    }}
    .status-badge {{
        display:inline-flex;
        align-items:center;
        gap:0.3rem;
        padding:0.15rem 0.55rem;
        border-radius:999px;
        font-size:0.72rem;
        font-weight:600;
        text-transform:uppercase;
        letter-spacing:0.07em;
    }}
    .status-idle {{
        background:rgba(148,163,184,0.18);
        color:#6b7280;
    }}
    .status-running {{
        background:radial-gradient(circle at top, var(--color-accent), #f97316);
        color:white;
        box-shadow:0 0 16px rgba(248,113,113,0.6);
        animation:pulse-running 1.4s ease-in-out infinite;
    }}
    .status-completed {{
        background:rgba(34,197,94,0.12);
        color:#16a34a;
    }}
    .status-error {{
        background:rgba(239,68,68,0.14);
        color:#ef4444;
    }}
    @keyframes pulse-running {{
        0% {{ box-shadow:0 0 4px rgba(248,113,113,0.4); }}
        50% {{ box-shadow:0 0 18px rgba(248,113,113,0.9); }}
        100% {{ box-shadow:0 0 4px rgba(248,113,113,0.4); }}
    }}
    .wow-chip {{
        display:inline-flex;
        align-items:center;
        gap:0.25rem;
        padding:0.1rem 0.5rem;
        border-radius:999px;
        font-size:0.72rem;
        border:1px solid rgba(148,163,184,0.35);
        background:rgba(15,23,42,0.02);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ------------------- PROVIDER HELPERS ----------------

def get_api_key(name: str, ui_value: Optional[str]) -> Optional[str]:
    """Prefer environment variable; fallback to UI input."""
    env_val = os.getenv(name)
    if env_val:
        return env_val
    return ui_value or None


def _extract_gemini_text(resp) -> str:
    chunks: List[str] = []
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        for p in parts:
            text = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
            if text:
                chunks.append(text)
    if chunks:
        return "\n".join(chunks).strip()
    try:
        txt = getattr(resp, "text", "") or ""
        return txt.strip()
    except Exception:
        return ""


def call_gemini(
    model: str,
    api_key: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    if genai is None:
        raise RuntimeError("google-generativeai SDK not installed.")

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
    )
    start = time.time()
    resp = gemini_model.generate_content(
        user_content,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    elapsed = (time.time() - start) * 1000
    text = _extract_gemini_text(resp)
    finish_reason = None
    candidates = getattr(resp, "candidates", None) or []
    if candidates:
        fr = getattr(candidates[0], "finish_reason", None)
        finish_reason = str(fr) if fr is not None else None
    usage = getattr(resp, "usage_metadata", None)

    if not text:
        msg = "Gemini returned no content. "
        if finish_reason:
            msg += f"(finish_reason={finish_reason}) "
        msg += (
            "This often happens when safety filters block the output or the model "
            "halts before producing text. Try simplifying/redacting the input, or "
            "switching to another provider/model."
        )
        raise RuntimeError(msg)

    return {
        "text": text,
        "latency_ms": elapsed,
        "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
        "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
    }


def call_openai(
    model: str,
    api_key: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    if OpenAIClient is None:
        raise RuntimeError("openai SDK not installed.")
    client = OpenAIClient(api_key=api_key)
    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed = (time.time() - start) * 1000
    choice = resp.choices[0]
    text = choice.message.content
    usage = getattr(resp, "usage", None)
    return {
        "text": text,
        "latency_ms": elapsed,
        "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
    }


def call_anthropic(
    model: str,
    api_key: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    if anthropic is None:
        raise RuntimeError("anthropic SDK not installed.")
    client = anthropic.Anthropic(api_key=api_key)
    start = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    elapsed = (time.time() - start) * 1000
    text_chunks: List[str] = []
    for block in getattr(resp, "content", []):
        if getattr(block, "type", None) == "text":
            text_chunks.append(block.text)
    text = "\n".join(text_chunks)
    usage = getattr(resp, "usage", None)
    return {
        "text": text,
        "latency_ms": elapsed,
        "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
    }


def call_grok(
    model: str,
    api_key: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    if XAIClient is None or xai_user is None or xai_system is None:
        raise RuntimeError("xai_sdk not installed.")
    client = XAIClient(api_key=api_key, timeout=3600)
    start = time.time()
    chat = client.chat.create(model=model)
    chat.append(xai_system(system_prompt))
    chat.append(xai_user(user_content))
    response = chat.sample(
        options={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
    )
    elapsed = (time.time() - start) * 1000
    text = getattr(response, "content", "") or ""
    usage = getattr(response, "usage", None)
    return {
        "text": text,
        "latency_ms": elapsed,
        "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
    }


def call_llm(
    agent: AgentConfig,
    system_prompt: str,
    user_content: str,
    api_keys: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    provider = agent.provider
    if provider == "gemini":
        key = api_keys["gemini"]
        if not key:
            raise RuntimeError("Gemini API key missing.")
        return call_gemini(
            model=agent.model,
            api_key=key,
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=agent.max_tokens,
            temperature=agent.temperature,
        )
    if provider == "openai":
        key = api_keys["openai"]
        if not key:
            raise RuntimeError("OpenAI API key missing.")
        return call_openai(
            model=agent.model,
            api_key=key,
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=agent.max_tokens,
            temperature=agent.temperature,
        )
    if provider == "anthropic":
        key = api_keys["anthropic"]
        if not key:
            raise RuntimeError("Anthropic API key missing.")
        return call_anthropic(
            model=agent.model,
            api_key=key,
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=agent.max_tokens,
            temperature=agent.temperature,
        )
    if provider == "grok":
        key = api_keys["grok"]
        if not key:
            raise RuntimeError("XAI (Grok) API key missing.")
        return call_grok(
            model=agent.model,
            api_key=key,
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=agent.max_tokens,
            temperature=agent.temperature,
        )
    raise RuntimeError(f"Unsupported provider: {provider}")


# ------------------- MODEL OPTIONS -----------------

def get_models_for_provider(provider: ProviderLiteral) -> List[str]:
    if provider == "gemini":
        return ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    if provider == "openai":
        return ["gpt-4o-mini", "gpt-4.1-mini"]
    if provider == "anthropic":
        return ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]
    if provider == "grok":
        return ["grok-4-fast-reasoning", "grok-3-mini"]
    return []


# ------------------- PIPELINE LOGIC ----------------

def build_agent_input(
    pipeline: PipelineState,
    agent_index: int,
    base_system_prompt: str,
) -> Dict[str, str]:
    agent = pipeline.agents[agent_index]
    prev_output = ""
    if agent_index > 0:
        prev_agent = pipeline.agents[agent_index - 1]
        prev_state = pipeline.history.get(prev_agent.id)
        prev_output = prev_state.output if prev_state else ""

    combined_system = base_system_prompt + "\n\n" + (agent.system_prompt_suffix or "")

    user_context = f"""[Template]
{pipeline.template}

[Observations]
{pipeline.observations}

[Previous Agent Output]
{prev_output}

[Agent Task Instructions]
{agent.user_prompt}
"""
    return {"system": combined_system, "user": user_context}


def run_agent(
    pipeline: PipelineState,
    agent_index: int,
    api_keys: Dict[str, Optional[str]],
) -> AgentRunState:
    agent = pipeline.agents[agent_index]
    state = pipeline.history.get(agent.id, AgentRunState())
    state.status = "running"
    pipeline.history[agent.id] = state

    try:
        prompts = build_agent_input(pipeline, agent_index, BASE_SYSTEM_PROMPT_ZH)
        result = call_llm(agent, prompts["system"], prompts["user"], api_keys)
        state.output = result["text"]
        state.status = "completed"
        state.latency_ms = result["latency_ms"]
        state.input_tokens = result["input_tokens"]
        state.output_tokens = result["output_tokens"]
        state.error = None
    except Exception as e:
        state.status = "error"
        state.error = str(e)
        try:
            log_event("error", f"Agent '{agent.name}' failed: {e}")
        except Exception:
            pass
    return state


def run_full_pipeline(
    pipeline: PipelineState,
    api_keys: Dict[str, Optional[str]],
):
    for i in range(len(pipeline.agents)):
        run_agent(pipeline, i, api_keys)


# ------------------- SMART REPLACEMENT --------------

def run_smart_replace(
    template_a: str,
    list_b: str,
    provider: ProviderLiteral,
    model: str,
    max_tokens: int,
    temperature: float,
    api_keys: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    user_content = f"""[Template A]
{template_a}

[List B]
{list_b}
"""
    dummy_agent = AgentConfig(
        id="smart-replace",
        name="Smart Replace",
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        user_prompt="依照任務說明進行套版與內容合併。",
        system_prompt_suffix="",
    )
    return call_llm(dummy_agent, SMART_REPLACE_SYSTEM_PROMPT, user_content, api_keys)


# ------------------- AGENT YAML UTILITIES -----------

def load_agents_from_yaml(path: Optional[str] = "agents.yaml") -> PipelineState:
    if not path:
        return PipelineState()

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                return PipelineState()
            data = yaml.safe_load(raw)
    except FileNotFoundError:
        return PipelineState()
    except yaml.YAMLError as e:
        print(f"[agents.yaml] YAML parse error, ignoring file: {e}")
        return PipelineState()

    if not isinstance(data, dict):
        return PipelineState()

    pipe = data.get("pipeline", {}) or {}
    agents_data = pipe.get("agents", []) or []

    agents: List[AgentConfig] = []
    for a in agents_data:
        if not isinstance(a, dict):
            continue
        agents.append(
            AgentConfig(
                id=a.get("id") or str(uuid.uuid4()),
                name=a.get("name", "Unnamed"),
                provider=a.get("provider", "gemini"),
                model=a.get("model", "gemini-2.5-flash"),
                max_tokens=int(a.get("max_tokens", 12000)),
                temperature=float(a.get("temperature", 0.2)),
                user_prompt=a.get("user_prompt", ""),
                system_prompt_suffix=a.get("system_prompt_suffix", ""),
            )
        )

    return PipelineState(
        template=pipe.get("template", "") or "",
        observations=pipe.get("observations", "") or "",
        current_step_index=0,
        agents=agents,
        history={},
    )


def export_agents_to_yaml(pipeline: PipelineState) -> str:
    data = {
        "pipeline": {
            "template": pipeline.template,
            "observations": pipeline.observations,
            "agents": [asdict(a) for a in pipeline.agents],
        }
    }
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


# ------------------- LANGUAGE LABELS ----------------

UI_LABELS = {
    "en": {
        "title": "AuditFlow WOW Studio – Medical Quality & Skill Workbench",
        "pipeline_tab": "Audit Pipeline",
        "smart_tab": "Smart Replace",
        "note_tab": "AI Note Keeper",
        "skill_tab": "SKILL Studio",
        "dashboard_tab": "Dashboard & Logs",
        "template": "Audit Template (Markdown)",
        "observations": "Raw Observations",
        "run_pipeline": "Run Full Pipeline",
        "run_agent": "Run This Agent",
        "agent_output": "Agent Output (Editable)",
        "api_keys": "API Keys",
        "gemini_key": "Gemini API Key",
        "openai_key": "OpenAI API Key",
        "anthropic_key": "Anthropic API Key",
        "grok_key": "XAI (Grok) API Key",
        "theme": "Painter Style",
        "dark_mode": "Dark Mode",
        "language": "UI Language",
        "style_wheel": "Style Jackpot",
        "smart_template": "Template A (Markdown Structure)",
        "smart_list": "List B (Data / Observations)",
        "smart_run": "Run Smart Replacement",
        "smart_preview": "Preview (HTML)",
        "smart_raw": "Raw Markdown",
        "export_md": "Download as Markdown",
        "api_env_used": "Using environment variable",
        "api_missing": "No environment variable; please enter here.",
        # Note Keeper
        "note_raw": "Raw Note (Text / Markdown)",
        "note_transform": "Transform Note to Structured Markdown",
        "note_transformed": "Transformed Note (Markdown, editable)",
        "note_view_mode": "Note View Mode",
        "note_view_md": "Markdown View",
        "note_view_text": "Plain Text View",
        "note_magic_title": "AI Magics",
        "note_magic_select": "Select Magic",
        "note_magic_custom_prompt": "Optional additional instructions for this magic",
        "note_magic_run": "Run Magic",
        "note_magic_result": "Magic Result",
        # SKILL Studio
        "skill_instr_title": "Instructions for SKILL.md",
        "skill_generate": "Generate SKILL.md from Instructions",
        "skill_md_editor": "SKILL.md (Markdown, editable)",
        "skill_view_mode": "SKILL.md View Mode",
        "skill_default": "Load Default Example SKILL.md",
        "skill_upload": "Upload SKILL.md File",
        "skill_download": "Download SKILL.md",
        "skill_file_upload": "Upload Document (docx, txt, csv, pdf, json, md)",
        "skill_file_preview": "Document Preview",
        "skill_file_summary_prompt": "Summary Prompt",
        "skill_file_summary_btn": "Summarize Document",
        "skill_file_summary_result": "Document Summary",
        "skill_apply_prompt": "Prompt to run with SKILL.md on this document",
        "skill_apply_btn": "Run SKILL on Document",
        "skill_apply_result": "Result (SKILL-driven)",
        "model": "Model",
        "provider": "Provider",
        "max_tokens": "Max tokens",
        "temp": "Temperature",
        "agent_output_view": "Output View",
        "agent_output_md": "Markdown",
        "agent_output_text": "Text",
    },
    "zh": {
        "title": "AuditFlow WOW Studio – 醫療稽核與 SKILL 工作台",
        "pipeline_tab": "稽核流程管線",
        "smart_tab": "智慧套版 (Smart Replace)",
        "note_tab": "AI 筆記管家",
        "skill_tab": "SKILL.md 工具箱",
        "dashboard_tab": "儀表板與日誌",
        "template": "稽核報告模板 (Markdown)",
        "observations": "原始觀察紀錄",
        "run_pipeline": "執行整體管線",
        "run_agent": "僅執行此代理",
        "agent_output": "代理輸出內容（可編輯）",
        "api_keys": "API 金鑰",
        "gemini_key": "Gemini API 金鑰",
        "openai_key": "OpenAI API 金鑰",
        "anthropic_key": "Anthropic API 金鑰",
        "grok_key": "XAI (Grok) 金鑰",
        "theme": "畫家風格主題",
        "dark_mode": "深色模式",
        "language": "介面語言",
        "style_wheel": "風格 Jackpot 轉轉樂",
        "smart_template": "Template A（Markdown 結構）",
        "smart_list": "List B（資料 / 觀察清單）",
        "smart_run": "執行智慧套版",
        "smart_preview": "預覽 (HTML)",
        "smart_raw": "原始 Markdown",
        "export_md": "下載 Markdown",
        "api_env_used": "已使用環境變數中的金鑰",
        "api_missing": "未偵測到環境變數，請在此輸入金鑰。",
        # Note Keeper
        "note_raw": "原始筆記（文字 / Markdown）",
        "note_transform": "整理筆記為結構化 Markdown",
        "note_transformed": "整理後筆記（Markdown，可編輯）",
        "note_view_mode": "筆記檢視模式",
        "note_view_md": "Markdown 檢視",
        "note_view_text": "純文字檢視",
        "note_magic_title": "AI 魔法功能",
        "note_magic_select": "選擇魔法",
        "note_magic_custom_prompt": "此魔法的額外指示（選填）",
        "note_magic_run": "執行魔法",
        "note_magic_result": "魔法結果",
        # SKILL Studio
        "skill_instr_title": "SKILL.md 撰寫說明（自然語言）",
        "skill_generate": "根據說明產生 SKILL.md",
        "skill_md_editor": "SKILL.md（Markdown，可編輯）",
        "skill_view_mode": "SKILL.md 檢視模式",
        "skill_default": "載入預設範例 SKILL.md",
        "skill_upload": "上傳 SKILL.md 檔案",
        "skill_download": "下載 SKILL.md",
        "skill_file_upload": "上傳文件（docx, txt, csv, pdf, json, md）",
        "skill_file_preview": "文件預覽",
        "skill_file_summary_prompt": "摘要提示詞",
        "skill_file_summary_btn": "產生文件摘要",
        "skill_file_summary_result": "文件摘要",
        "skill_apply_prompt": "在此文件上套用 SKILL.md 的提示詞",
        "skill_apply_btn": "執行 SKILL 至文件",
        "skill_apply_result": "結果（依 SKILL.md 執行）",
        "model": "模型",
        "provider": "供應商",
        "max_tokens": "最大 tokens",
        "temp": "溫度",
        "agent_output_view": "輸出檢視模式",
        "agent_output_md": "Markdown",
        "agent_output_text": "文字",
    },
}


# ------------------- WOW STATUS BADGE --------------

def render_status_badge(status: str):
    label_map = {
        "idle": "IDLE",
        "running": "RUN",
        "completed": "DONE",
        "error": "ERROR",
    }
    html = f'<span class="status-badge status-{status}">{label_map.get(status, status).upper()}</span>'
    st.markdown(html, unsafe_allow_html=True)


# ------------------- FILE HELPERS -------------------

def load_uploaded_file_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if isinstance(data, bytes):
        data_bytes = data
        text = None
    else:
        data_bytes = None
        text = data

    if name.endswith((".txt", ".md")):
        return (text or data_bytes.decode("utf-8", errors="ignore")).strip()
    if name.endswith(".json"):
        try:
            obj = json.loads(text or data_bytes.decode("utf-8", errors="ignore"))
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            return text or data_bytes.decode("utf-8", errors="ignore")
    if name.endswith(".csv"):
        if pd is None:
            return (text or data_bytes.decode("utf-8", errors="ignore"))
        try:
            buf = io.StringIO(text or data_bytes.decode("utf-8", errors="ignore"))
            df = pd.read_csv(buf)
            return df.to_markdown(index=False)
        except Exception:
            return (text or data_bytes.decode("utf-8", errors="ignore"))
    if name.endswith(".docx"):
        if docx is None:
            return "python-docx is not installed, unable to parse .docx. Please install python-docx."
        try:
            file_bytes = data_bytes or (text.encode("utf-8"))
            bio = io.BytesIO(file_bytes)
            document = docx.Document(bio)
            return "\n".join(p.text for p in document.paragraphs)
        except Exception as e:
            return f"Failed to parse .docx: {e}"
    if name.endswith(".pdf"):
        if PyPDF2 is None:
            return "PyPDF2 is not installed, unable to parse PDF. Please install PyPDF2."
        try:
            file_bytes = data_bytes or (text.encode("utf-8"))
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
            return "\n\n".join(pages_text)
        except Exception as e:
            return f"Failed to parse PDF: {e}"
    # Fallback
    return (text or data_bytes.decode("utf-8", errors="ignore")).strip()


def pdf_embed_html(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if not name.endswith(".pdf"):
        return None
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if not isinstance(data, bytes):
        return None
    b64 = base64.b64encode(data).decode("utf-8")
    return f"""
    <iframe
        src="data:application/pdf;base64,{b64}"
        width="100%"
        height="600px"
        style="border:1px solid rgba(148,163,184,0.5);border-radius:12px;"
    ></iframe>
    """


# ------------------- NOTE KEEPER LOGIC --------------

def run_note_transform(
    note_text: str,
    provider: ProviderLiteral,
    model: str,
    max_tokens: int,
    temperature: float,
    api_keys: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    content = f"[Raw note]\n{note_text}"
    dummy = AgentConfig(
        id="note-transform",
        name="Note Transform",
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        user_prompt="Transform this raw note into a structured markdown note with coral-colored keywords.",
    )
    return call_llm(dummy, NOTE_KEEPER_SYSTEM_PROMPT, content, api_keys)


def run_note_magic(
    magic_kind: str,
    note_markdown: str,
    extra_instructions: str,
    provider: ProviderLiteral,
    model: str,
    max_tokens: int,
    temperature: float,
    api_keys: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    if magic_kind == "transform":
        magic_prompt = "Re-transform and improve this note into even clearer, well-structured markdown."
    elif magic_kind == "summary":
        magic_prompt = "Produce a concise but information-rich markdown summary of this note."
    elif magic_kind == "action_items":
        magic_prompt = "Extract all concrete action items from this note as a checklist with owners and due dates if present."
    elif magic_kind == "translate_en":
        magic_prompt = "Translate this note into clear English while preserving structure and technical terms."
    elif magic_kind == "qa":
        magic_prompt = "Generate 5-10 thoughtful questions that someone should ask based on this note."
    else:
        magic_prompt = "Improve and clarify this note."

    if extra_instructions:
        magic_prompt += "\n\nAdditional user instructions:\n" + extra_instructions

    user_content = f"""[Existing note]
{note_markdown}

[Task]
{magic_prompt}
"""

    dummy = AgentConfig(
        id=f"note-magic-{magic_kind}",
        name=f"Note Magic – {magic_kind}",
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        user_prompt="Apply the requested magic on the note.",
    )
    return call_llm(dummy, NOTE_MAGIC_BASE_PROMPT, user_content, api_keys)


# ------------------- SKILL STUDIO LOGIC --------------

DEFAULT_SKILL_MD = """---
name: analyzing-pdf-audits
description: Analyzes PDF-based audit reports, extracts key findings, and summarizes nonconformities. Use when the user provides PDF audit reports or asks for audit finding synthesis.
---

# PDF Audit Analysis Skill

## Scope

- Input: Audit reports in PDF or extracted text
- Output: Structured analysis of:
  - Conformities
  - Nonconformities
  - Observations / opportunities for improvement
  - Suggested corrective actions (if requested)

## Usage pattern

1. Ensure the source text is complete and legible.
2. Identify and label all nonconformities with:
   - Clear factual description
   - Impact / risk explanation
   - Relevant clause or regulation (if known)
3. Summarize overall audit results with emphasis on risk and traceability.

## Example prompt

> Please analyze this audit report and produce:
> - Key findings by clause
> - A table of nonconformities
> - Suggested next steps

"""


def run_skill_author(
    instructions: str,
    provider: ProviderLiteral,
    model: str,
    max_tokens: int,
    temperature: float,
    api_keys: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    content = f"""[High-level instructions]
{instructions}

[Task]
Convert these instructions into a complete SKILL.md following the best practices described in the system prompt.
"""
    dummy = AgentConfig(
        id="skill-author",
        name="SKILL Author",
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        user_prompt="Generate SKILL.md.",
    )
    return call_llm(dummy, SKILL_AUTHOR_SYSTEM_PROMPT, content, api_keys)


def run_file_summary(
    file_text: str,
    user_prompt: str,
    provider: ProviderLiteral,
    model: str,
    max_tokens: int,
    temperature: float,
    api_keys: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    content = f"""[Document content]
{file_text}

[Summary instructions]
{user_prompt}
"""
    dummy = AgentConfig(
        id="file-summary",
        name="File Summary",
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        user_prompt="Summarize the document according to the instructions.",
    )
    return call_llm(dummy, FILE_SUMMARY_SYSTEM_PROMPT, content, api_keys)


def run_skill_on_file(
    skill_md: str,
    file_text: str,
    user_prompt: str,
    provider: ProviderLiteral,
    model: str,
    max_tokens: int,
    temperature: float,
    api_keys: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    content = f"""[SKILL.md]
{skill_md}

[Document content]
{file_text}

[User prompt]
{user_prompt}
"""
    dummy = AgentConfig(
        id="skill-apply",
        name="SKILL Apply",
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        user_prompt="Apply SKILL.md to the document according to the user prompt.",
    )
    return call_llm(dummy, SKILL_APPLY_SYSTEM_PROMPT, content, api_keys)


# ------------------- SESSION STATE & LOGGING -------

def init_session_state():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = load_agents_from_yaml()
    if "smart_result" not in st.session_state:
        st.session_state.smart_result = ""
    if "smart_template" not in st.session_state:
        st.session_state.smart_template = st.session_state.pipeline.template or ""
    if "smart_list" not in st.session_state:
        st.session_state.smart_list = st.session_state.pipeline.observations or ""
    if "ui_language" not in st.session_state:
        st.session_state.ui_language = "zh"
    if "theme_name" not in st.session_state:
        st.session_state.theme_name = "Van Gogh"
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    if "log_entries" not in st.session_state:
        st.session_state.log_entries = []
    if "manual_api_keys" not in st.session_state:
        st.session_state.manual_api_keys = {
            "gemini": "",
            "openai": "",
            "anthropic": "",
            "grok": "",
        }
    # Note Keeper
    if "note_raw" not in st.session_state:
        st.session_state.note_raw = ""
    if "note_transformed" not in st.session_state:
        st.session_state.note_transformed = ""
    if "note_magic_result" not in st.session_state:
        st.session_state.note_magic_result = ""
    # SKILL Studio
    if "skill_instr" not in st.session_state:
        st.session_state.skill_instr = ""
    if "skill_md" not in st.session_state:
        st.session_state.skill_md = DEFAULT_SKILL_MD
    if "skill_file_text" not in st.session_state:
        st.session_state.skill_file_text = ""
    if "skill_file_summary" not in st.session_state:
        st.session_state.skill_file_summary = ""
    if "skill_apply_result" not in st.session_state:
        st.session_state.skill_apply_result = ""


def log_event(level: str, message: str):
    st.session_state.log_entries.append(
        {"level": level, "message": message, "ts": time.strftime("%H:%M:%S")}
    )


# ------------------- STREAMLIT APP -----------------

def main():
    st.set_page_config(
        page_title="AuditFlow WOW Studio",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    lang = st.session_state.ui_language
    labels = UI_LABELS[lang]

    # Sidebar: API keys, language, theme, YAML
    with st.sidebar:
        st.markdown(f"### {labels['api_keys']}")

        # Gemini
        gemini_env = os.getenv("GEMINI_API_KEY")
        if gemini_env:
            st.caption(f"Gemini – {labels['api_env_used']}")
            gemini_ui = ""
        else:
            st.caption(f"Gemini – {labels['api_missing']}")
            gemini_ui = st.text_input(labels["gemini_key"], type="password")
        # OpenAI
        openai_env = os.getenv("OPENAI_API_KEY")
        if openai_env:
            st.caption(f"OpenAI – {labels['api_env_used']}")
            openai_ui = ""
        else:
            st.caption(f"OpenAI – {labels['api_missing']}")
            openai_ui = st.text_input(labels["openai_key"], type="password")
        # Anthropic
        anthropic_env = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_env:
            st.caption(f"Anthropic – {labels['api_env_used']}")
            anthropic_ui = ""
        else:
            st.caption(f"Anthropic – {labels['api_missing']}")
            anthropic_ui = st.text_input(labels["anthropic_key"], type="password")
        # Grok / XAI
        grok_env = os.getenv("XAI_API_KEY")
        if grok_env:
            st.caption(f"Grok (XAI) – {labels['api_env_used']}")
            grok_ui = ""
        else:
            st.caption(f"Grok (XAI) – {labels['api_missing']}")
            grok_ui = st.text_input(labels["grok_key"], type="password")

        st.session_state.manual_api_keys["gemini"] = gemini_ui
        st.session_state.manual_api_keys["openai"] = openai_ui
        st.session_state.manual_api_keys["anthropic"] = anthropic_ui
        st.session_state.manual_api_keys["grok"] = grok_ui

        st.markdown("---")
        st.markdown("### UI")
        lang_choice = st.radio(
            labels["language"],
            options=["zh", "en"],
            index=0 if lang == "zh" else 1,
        )
        st.session_state.ui_language = lang_choice
        labels = UI_LABELS[lang_choice]

        theme_name = st.selectbox(labels["theme"], options=list(PAINTER_THEMES.keys()))
        st.session_state.theme_name = theme_name
        dark_mode = st.toggle(labels["dark_mode"], value=st.session_state.dark_mode)
        st.session_state.dark_mode = dark_mode

        if st.button(labels["style_wheel"]):
            import random
            choice = random.choice(list(PAINTER_THEMES.keys()))
            st.session_state.theme_name = choice
            log_event("info", f"Style Jackpot selected painter theme: {choice}")

        st.markdown("---")
        st.markdown("### agents.yaml")
        if st.button("Reload agents.yaml from disk"):
            st.session_state.pipeline = load_agents_from_yaml()
            log_event("info", "Reloaded agents.yaml")

        yaml_str = export_agents_to_yaml(st.session_state.pipeline)
        st.download_button(
            "Download current agents.yaml",
            data=yaml_str,
            file_name="agents.yaml",
            mime="text/yaml",
        )

        uploaded_yaml = st.file_uploader("Upload agents.yaml", type=["yaml", "yml"])
        if uploaded_yaml:
            try:
                uploaded_text = uploaded_yaml.read()
                if isinstance(uploaded_text, bytes):
                    uploaded_text = uploaded_text.decode("utf-8", errors="replace")
                uploaded_text = uploaded_text.strip()
                data = yaml.safe_load(uploaded_text) if uploaded_text else {}
            except yaml.YAMLError as e:
                st.error(f"Uploaded YAML is invalid: {e}")
            else:
                pipe = (data or {}).get("pipeline", {}) if isinstance(data, dict) else {}
                agents_data = pipe.get("agents", []) or []

                agents: List[AgentConfig] = []
                for a in agents_data:
                    if not isinstance(a, dict):
                        continue
                    agents.append(
                        AgentConfig(
                            id=a.get("id") or str(uuid.uuid4()),
                            name=a.get("name", "Unnamed"),
                            provider=a.get("provider", "gemini"),
                            model=a.get("model", "gemini-2.5-flash"),
                            max_tokens=int(a.get("max_tokens", 12000)),
                            temperature=float(a.get("temperature", 0.2)),
                            user_prompt=a.get("user_prompt", ""),
                            system_prompt_suffix=a.get("system_prompt_suffix", ""),
                        )
                    )

                st.session_state.pipeline = PipelineState(
                    template=pipe.get("template", "") or "",
                    observations=pipe.get("observations", "") or "",
                    current_step_index=0,
                    agents=agents,
                    history={},
                )
                log_event("success", "Loaded agents from uploaded YAML.")

    apply_theme(st.session_state.theme_name, st.session_state.dark_mode)

    st.markdown(f"## {labels['title']}")
    st.markdown(
        '<div class="wow-chip">WOW UI · Multi-Model Agents · SKILL.md · Note Keeper</div>',
        unsafe_allow_html=True,
    )

    # Effective API keys
    api_keys = {
        "gemini": get_api_key("GEMINI_API_KEY", st.session_state.manual_api_keys["gemini"]),
        "openai": get_api_key("OPENAI_API_KEY", st.session_state.manual_api_keys["openai"]),
        "anthropic": get_api_key("ANTHROPIC_API_KEY", st.session_state.manual_api_keys["anthropic"]),
        "grok": get_api_key("XAI_API_KEY", st.session_state.manual_api_keys["grok"]),
    }

    tabs = st.tabs(
        [
            labels["pipeline_tab"],
            labels["smart_tab"],
            labels["note_tab"],
            labels["skill_tab"],
            labels["dashboard_tab"],
        ]
    )

    # ----------------- PIPELINE TAB -------------------
    with tabs[0]:
        pipeline: PipelineState = st.session_state.pipeline

        col1, col2 = st.columns(2)
        with col1:
            pipeline.template = st.text_area(
                labels["template"], value=pipeline.template, height=200
            )
        with col2:
            pipeline.observations = st.text_area(
                labels["observations"], value=pipeline.observations, height=200
            )

        st.markdown("### Agents")

        for idx, agent in enumerate(pipeline.agents):
            with st.expander(f"{idx+1}. {agent.name}", expanded=(idx == 0)):
                c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
                with c1:
                    provider = st.selectbox(
                        labels["provider"],
                        options=["gemini", "openai", "anthropic", "grok"],
                        index=["gemini", "openai", "anthropic", "grok"].index(
                            agent.provider
                        )
                        if agent.provider in ["gemini", "openai", "anthropic", "grok"]
                        else 0,
                        key=f"provider_{agent.id}",
                    )
                    agent.provider = provider
                with c2:
                    models = get_models_for_provider(provider)
                    model = st.selectbox(
                        labels["model"],
                        options=models,
                        index=models.index(agent.model) if agent.model in models else 0,
                        key=f"model_{agent.id}",
                    )
                    agent.model = model
                with c3:
                    agent.max_tokens = st.number_input(
                        labels["max_tokens"],
                        min_value=256,
                        max_value=120000,
                        value=int(agent.max_tokens) if agent.max_tokens else 12000,
                        step=512,
                        key=f"max_tokens_{agent.id}",
                    )
                with c4:
                    agent.temperature = st.slider(
                        labels["temp"],
                        min_value=0.0,
                        max_value=1.0,
                        value=float(agent.temperature),
                        step=0.05,
                        key=f"temp_{agent.id}",
                    )

                agent.user_prompt = st.text_area(
                    "User Prompt",
                    value=agent.user_prompt,
                    height=120,
                    key=f"up_{agent.id}",
                )
                agent.system_prompt_suffix = st.text_area(
                    "System Prompt Suffix",
                    value=agent.system_prompt_suffix,
                    height=80,
                    key=f"sp_{agent.id}",
                )

                effective_api_keys = {
                    "gemini": api_keys["gemini"],
                    "openai": api_keys["openai"],
                    "anthropic": api_keys["anthropic"],
                    "grok": api_keys["grok"],
                }
                if not effective_api_keys[agent.provider]:
                    st.warning(
                        f"No API key configured for provider `{agent.provider}`. "
                        "Set it in environment or sidebar before running this agent."
                    )

                # Status & run
                state = pipeline.history.get(agent.id, AgentRunState())

                col_run1, col_run2, col_status = st.columns([1, 1, 2])
                with col_run1:
                    if st.button(labels["run_agent"], key=f"run_{agent.id}"):
                        log_event(
                            "info",
                            f"Running agent {agent.name}. (provider={agent.provider}, model={agent.model})",
                        )
                        new_state = run_agent(pipeline, idx, api_keys)
                        pipeline.history[agent.id] = new_state
                        state = new_state

                with col_run2:
                    render_status_badge(state.status)

                with col_status:
                    if state.latency_ms is not None:
                        st.caption(f"Latency: {state.latency_ms:.0f} ms")

                if state.error:
                    st.error(f"Error: {state.error}")

                # Output view mode
                ov_col1, ov_col2 = st.columns([1, 3])
                with ov_col1:
                    out_view = st.radio(
                        labels["agent_output_view"],
                        options=[labels["agent_output_md"], labels["agent_output_text"]],
                        key=f"out_view_{agent.id}",
                        horizontal=True,
                    )
                output_key = f"out_{agent.id}"
                if output_key not in st.session_state:
                    st.session_state[output_key] = state.output or ""
                if state.status in ("completed", "error"):
                    st.session_state[output_key] = state.output or st.session_state[output_key]

                with ov_col2:
                    if out_view == labels["agent_output_md"]:
                        new_output = st.text_area(
                            labels["agent_output"],
                            key=output_key,
                            height=200,
                        )
                        # Also show rendered markdown preview
                        st.markdown("---")
                        st.markdown("Preview:")
                        st.markdown(new_output or "")
                    else:
                        new_output = st.text_area(
                            labels["agent_output"],
                            key=output_key,
                            height=200,
                        )

                if new_output != (state.output or ""):
                    state.output = new_output
                    pipeline.history[agent.id] = state

                st.download_button(
                    labels["export_md"],
                    data=new_output or "",
                    file_name=f"{agent.id}.md",
                    mime="text/markdown",
                    key=f"dl_{agent.id}",
                )

        if st.button(labels["run_pipeline"]):
            log_event("info", "Running full pipeline.")
            run_full_pipeline(pipeline, api_keys)

        st.session_state.pipeline = pipeline

    # ----------------- SMART REPLACE TAB -------------
    with tabs[1]:
        pipeline: PipelineState = st.session_state.pipeline

        colL, colR = st.columns(2)
        with colL:
            smart_template = st.text_area(
                labels["smart_template"],
                height=260,
                key="smart_template",
            )
            smart_list = st.text_area(
                labels["smart_list"],
                height=260,
                key="smart_list",
            )

            sr_provider: ProviderLiteral = st.selectbox(
                labels["provider"],
                options=["gemini", "openai", "anthropic", "grok"],
                key="sr_provider",
            )
            models = get_models_for_provider(sr_provider)
            sr_model = st.selectbox(labels["model"], options=models, key="sr_model")
            sr_max_tokens = st.number_input(
                labels["max_tokens"],
                min_value=256,
                max_value=120000,
                value=12000,
                step=512,
                key="sr_max_tokens",
            )
            sr_temp = st.slider(
                labels["temp"],
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key="sr_temp",
            )

            effective_api_keys = {
                "gemini": api_keys["gemini"],
                "openai": api_keys["openai"],
                "anthropic": api_keys["anthropic"],
                "grok": api_keys["grok"],
            }
            if not effective_api_keys[sr_provider]:
                env_name = {
                    "gemini": "GEMINI_API_KEY",
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "grok": "XAI_API_KEY",
                }[sr_provider]
                st.warning(
                    f"No API key found for provider `{sr_provider}`. "
                    f"Set `{env_name}` or enter a key in the sidebar."
                )

            if st.button(labels["smart_run"], key="smart_run_button"):
                log_event("info", "Running Smart Replace.")
                try:
                    res = run_smart_replace(
                        template_a=smart_template,
                        list_b=smart_list,
                        provider=sr_provider,
                        model=sr_model,
                        max_tokens=int(sr_max_tokens),
                        temperature=float(sr_temp),
                        api_keys=effective_api_keys,
                    )
                    result_text = res.get("text") or ""
                    st.session_state.smart_result = result_text
                    if not result_text.strip():
                        st.warning(
                            "Smart Replace completed but returned an empty response. "
                            "This can happen due to safety filters or model limits."
                        )
                    else:
                        log_event("success", "Smart Replace completed.")
                except Exception as e:
                    log_event("error", f"Smart Replace failed: {e}")
                    st.error(f"Smart Replace error: {e}")

        with colR:
            view_mode = st.radio(
                "View", options=[labels["smart_preview"], labels["smart_raw"]]
            )
            result = st.session_state.smart_result
            if view_mode == labels["smart_preview"]:
                if result.strip():
                    st.markdown(result, unsafe_allow_html=True)
                else:
                    st.info("No Smart Replace result yet. Run it from the left panel.")
            else:
                st.code(result or "", language="markdown")
            st.download_button(
                labels["export_md"],
                data=result or "",
                file_name="smart_replace.md",
                mime="text/markdown",
            )

    # ----------------- AI NOTE KEEPER TAB -------------
    with tabs[2]:
        st.markdown("### " + labels["note_tab"])
        colL, colR = st.columns(2)

        with colL:
            st.session_state.note_raw = st.text_area(
                labels["note_raw"],
                value=st.session_state.note_raw,
                height=260,
                key="note_raw",
            )

            nk_provider: ProviderLiteral = st.selectbox(
                labels["provider"],
                options=["gemini", "openai", "anthropic", "grok"],
                key="nk_provider",
            )
            nk_models = get_models_for_provider(nk_provider)
            nk_model = st.selectbox(labels["model"], options=nk_models, key="nk_model")
            nk_max_tokens = st.number_input(
                labels["max_tokens"],
                min_value=256,
                max_value=120000,
                value=12000,
                step=512,
                key="nk_max_tokens",
            )
            nk_temp = st.slider(
                labels["temp"],
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="nk_temp",
            )

            effective_api_keys = api_keys
            if st.button(labels["note_transform"], key="note_transform_btn"):
                if not effective_api_keys[nk_provider]:
                    st.error("Missing API key for selected provider.")
                else:
                    try:
                        res = run_note_transform(
                            st.session_state.note_raw,
                            nk_provider,
                            nk_model,
                            int(nk_max_tokens),
                            float(nk_temp),
                            effective_api_keys,
                        )
                        st.session_state.note_transformed = res.get("text") or ""
                        log_event("success", "Note transformed.")
                    except Exception as e:
                        log_event("error", f"Note transform failed: {e}")
                        st.error(f"Note transform error: {e}")

        with colR:
            view_mode = st.radio(
                labels["note_view_mode"],
                options=[labels["note_view_md"], labels["note_view_text"]],
                key="note_view_mode",
                horizontal=True,
            )
            if view_mode == labels["note_view_md"]:
                st.session_state.note_transformed = st.text_area(
                    labels["note_transformed"],
                    value=st.session_state.note_transformed,
                    height=260,
                    key="note_transformed_md",
                )
                st.markdown("---")
                st.markdown("Preview:")
                st.markdown(st.session_state.note_transformed or "")
            else:
                st.session_state.note_transformed = st.text_area(
                    labels["note_transformed"],
                    value=st.session_state.note_transformed,
                    height=260,
                    key="note_transformed_text",
                )

            st.download_button(
                labels["export_md"],
                data=st.session_state.note_transformed or "",
                file_name="note.md",
                mime="text/markdown",
            )

        st.markdown("---")
        st.markdown(f"#### {labels['note_magic_title']}")

        col_magicL, col_magicR = st.columns(2)
        with col_magicL:
            magic_kind = st.selectbox(
                labels["note_magic_select"],
                options=[
                    ("transform", "AI Transforming (improve structure)"),
                    ("summary", "Summarize note"),
                    ("action_items", "Action items checklist"),
                    ("translate_en", "Translate to English"),
                    ("qa", "Generate insightful questions"),
                ],
                format_func=lambda x: x[1],
                key="note_magic_kind",
            )[0]
            extra_prompt = st.text_area(
                labels["note_magic_custom_prompt"],
                height=120,
                key="note_magic_extra",
            )
            nm_provider: ProviderLiteral = st.selectbox(
                labels["provider"],
                options=["gemini", "openai", "anthropic", "grok"],
                key="nm_provider",
            )
            nm_models = get_models_for_provider(nm_provider)
            nm_model = st.selectbox(labels["model"], nm_models, key="nm_model")
            nm_max_tokens = st.number_input(
                labels["max_tokens"],
                min_value=256,
                max_value=120000,
                value=8000,
                step=512,
                key="nm_max_tokens",
            )
            nm_temp = st.slider(
                labels["temp"],
                min_value=0.0,
                max_value=1.0,
                value=0.4,
                step=0.05,
                key="nm_temp",
            )

            if st.button(labels["note_magic_run"], key="note_magic_run_btn"):
                if not api_keys[nm_provider]:
                    st.error("Missing API key for selected provider.")
                else:
                    try:
                        res = run_note_magic(
                            magic_kind,
                            st.session_state.note_transformed or st.session_state.note_raw,
                            extra_prompt,
                            nm_provider,
                            nm_model,
                            int(nm_max_tokens),
                            float(nm_temp),
                            api_keys,
                        )
                        st.session_state.note_magic_result = res.get("text") or ""
                        log_event("success", "Note magic executed.")
                    except Exception as e:
                        log_event("error", f"Note magic failed: {e}")
                        st.error(f"Note magic error: {e}")

        with col_magicR:
            st.markdown("##### " + labels["note_magic_result"])
            st.markdown(st.session_state.note_magic_result or "")
            st.download_button(
                labels["export_md"],
                data=st.session_state.note_magic_result or "",
                file_name="note_magic_result.md",
                mime="text/markdown",
            )

    # ----------------- SKILL STUDIO TAB -------------
    with tabs[3]:
        st.markdown("### " + labels["skill_tab"])
        col_topL, col_topR = st.columns(2)
        # --- SKILL Authoring
        with col_topL:
            st.markdown("#### 1. " + labels["skill_instr_title"])
            st.session_state.skill_instr = st.text_area(
                labels["skill_instr_title"],
                value=st.session_state.skill_instr,
                height=200,
                key="skill_instr",
            )
            sk_provider: ProviderLiteral = st.selectbox(
                labels["provider"],
                options=["gemini", "openai", "anthropic", "grok"],
                key="sk_provider",
            )
            sk_models = get_models_for_provider(sk_provider)
            sk_model = st.selectbox(labels["model"], sk_models, key="sk_model")
            sk_max_tokens = st.number_input(
                labels["max_tokens"],
                min_value=256,
                max_value=120000,
                value=12000,
                step=512,
                key="sk_max_tokens",
            )
            sk_temp = st.slider(
                labels["temp"],
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="sk_temp",
            )

            if st.button(labels["skill_generate"], key="skill_generate_btn"):
                if not api_keys[sk_provider]:
                    st.error("Missing API key for selected provider.")
                else:
                    try:
                        res = run_skill_author(
                            st.session_state.skill_instr,
                            sk_provider,
                            sk_model,
                            int(sk_max_tokens),
                            float(sk_temp),
                            api_keys,
                        )
                        st.session_state.skill_md = res.get("text") or ""
                        log_event("success", "SKILL.md generated from instructions.")
                    except Exception as e:
                        log_event("error", f"SKILL.md generation failed: {e}")
                        st.error(f"SKILL.md generation error: {e}")

        with col_topR:
            st.markdown("#### 2. " + labels["skill_md_editor"])
            skill_view_mode = st.radio(
                labels["skill_view_mode"],
                options=[labels["note_view_md"], labels["note_view_text"]],
                key="skill_view_mode",
                horizontal=True,
            )

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button(labels["skill_default"], key="skill_default_btn"):
                    st.session_state.skill_md = DEFAULT_SKILL_MD
            with col_btn2:
                uploaded_skill = st.file_uploader(labels["skill_upload"], type=["md"], key="skill_upload")
                if uploaded_skill:
                    content = uploaded_skill.read()
                    if isinstance(content, bytes):
                        content = content.decode("utf-8", errors="ignore")
                    st.session_state.skill_md = content

            if skill_view_mode == labels["note_view_md"]:
                st.session_state.skill_md = st.text_area(
                    labels["skill_md_editor"],
                    value=st.session_state.skill_md,
                    height=260,
                    key="skill_md_md",
                )
                st.markdown("---")
                st.markdown("Preview:")
                st.markdown(st.session_state.skill_md or "")
            else:
                st.session_state.skill_md = st.text_area(
                    labels["skill_md_editor"],
                    value=st.session_state.skill_md,
                    height=260,
                    key="skill_md_text",
                )

            st.download_button(
                labels["skill_download"],
                data=st.session_state.skill_md or "",
                file_name="SKILL.md",
                mime="text/markdown",
            )

        st.markdown("---")
        st.markdown("#### 3. " + labels["skill_file_upload"])
        col_fileL, col_fileR = st.columns(2)
        with col_fileL:
            uploaded_doc = st.file_uploader(
                labels["skill_file_upload"],
                type=["docx", "txt", "csv", "pdf", "json", "md"],
                key="skill_doc_upload",
            )
            if uploaded_doc:
                uploaded_doc_bytes = uploaded_doc.read()
                uploaded_doc.seek(0)
                # For preview and text, we re-wrap file into an in-memory buffer
                tmp_file = io.BytesIO(uploaded_doc_bytes)
                tmp_file.name = uploaded_doc.name
                # For text extraction
                st.session_state.skill_file_text = load_uploaded_file_text(
                    type("Tmp", (), {"name": uploaded_doc.name, "read": lambda: uploaded_doc_bytes})
                )
            st.markdown("##### " + labels["skill_file_preview"])
            if uploaded_doc:
                if uploaded_doc.name.lower().endswith(".pdf"):
                    uploaded_doc.seek(0)
                    pdf_html = pdf_embed_html(uploaded_doc)
                    if pdf_html:
                        st.components.v1.html(pdf_html, height=620, scrolling=True)
                # Text preview (first chars)
                preview_text = (st.session_state.skill_file_text or "")[:4000]
                st.markdown("Text preview (first ~4000 chars):")
                st.code(preview_text or "(no content extracted)", language="markdown")
            else:
                st.info("No document uploaded yet.")

        with col_fileR:
            # Summary
            st.markdown("##### " + labels["skill_file_summary_result"])
            sum_prompt_default_en = (
                "Write a comprehensive markdown summary of this document. "
                "Include overview, key sections, main arguments or data, and any visible risks."
            )
            sum_prompt_default_zh = (
                "請用 Markdown 撰寫一份完整摘要，包含：整體概觀、主要章節、關鍵重點與數據，以及明顯風險或問題。"
            )
            sum_prompt_default = sum_prompt_default_zh if lang == "zh" else sum_prompt_default_en
            summary_prompt = st.text_area(
                labels["skill_file_summary_prompt"],
                value=sum_prompt_default,
                height=120,
                key="skill_summary_prompt",
            )
            sf_provider: ProviderLiteral = st.selectbox(
                labels["provider"],
                options=["gemini", "openai", "anthropic", "grok"],
                key="sf_provider",
            )
            sf_models = get_models_for_provider(sf_provider)
            sf_model = st.selectbox(labels["model"], sf_models, key="sf_model")
            sf_max_tokens = st.number_input(
                labels["max_tokens"],
                min_value=256,
                max_value=120000,
                value=8000,
                step=512,
                key="sf_max_tokens",
            )
            sf_temp = st.slider(
                labels["temp"],
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key="sf_temp",
            )

            if st.button(labels["skill_file_summary_btn"], key="skill_summary_btn"):
                if not uploaded_doc:
                    st.error("Please upload a document first.")
                elif not api_keys[sf_provider]:
                    st.error("Missing API key for selected provider.")
                else:
                    try:
                        # Limit text length to keep within context budget
                        text_for_summary = (st.session_state.skill_file_text or "")[:24000]
                        res = run_file_summary(
                            text_for_summary,
                            summary_prompt,
                            sf_provider,
                            sf_model,
                            int(sf_max_tokens),
                            float(sf_temp),
                            api_keys,
                        )
                        st.session_state.skill_file_summary = res.get("text") or ""
                        log_event("success", "Document summary created.")
                    except Exception as e:
                        log_event("error", f"Document summary failed: {e}")
                        st.error(f"Summary error: {e}")

            st.markdown(st.session_state.skill_file_summary or "")
            st.download_button(
                labels["export_md"],
                data=st.session_state.skill_file_summary or "",
                file_name="document_summary.md",
                mime="text/markdown",
                key="skill_summary_dl",
            )

        st.markdown("---")
        st.markdown("#### 4. " + labels["skill_apply_result"])
        col_applyL, col_applyR = st.columns(2)
        with col_applyL:
            skill_apply_prompt = st.text_area(
                labels["skill_apply_prompt"],
                height=160,
                key="skill_apply_prompt",
            )
            sa_provider: ProviderLiteral = st.selectbox(
                labels["provider"],
                options=["gemini", "openai", "anthropic", "grok"],
                key="sa_provider",
            )
            sa_models = get_models_for_provider(sa_provider)
            sa_model = st.selectbox(labels["model"], sa_models, key="sa_model")
            sa_max_tokens = st.number_input(
                labels["max_tokens"],
                min_value=256,
                max_value=120000,
                value=12000,
                step=512,
                key="sa_max_tokens",
            )
            sa_temp = st.slider(
                labels["temp"],
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="sa_temp",
            )

            if st.button(labels["skill_apply_btn"], key="skill_apply_btn"):
                if not uploaded_doc:
                    st.error("Please upload a document first.")
                elif not st.session_state.skill_md.strip():
                    st.error("SKILL.md is empty. Please generate, paste, upload, or load default SKILL.md.")
                elif not api_keys[sa_provider]:
                    st.error("Missing API key for selected provider.")
                else:
                    try:
                        text_for_skill = (st.session_state.skill_file_text or "")[:24000]
                        res = run_skill_on_file(
                            st.session_state.skill_md,
                            text_for_skill,
                            skill_apply_prompt,
                            sa_provider,
                            sa_model,
                            int(sa_max_tokens),
                            float(sa_temp),
                            api_keys,
                        )
                        st.session_state.skill_apply_result = res.get("text") or ""
                        log_event("success", "SKILL applied to document.")
                    except Exception as e:
                        log_event("error", f"SKILL apply failed: {e}")
                        st.error(f"SKILL apply error: {e}")

        with col_applyR:
            st.markdown(st.session_state.skill_apply_result or "")
            st.download_button(
                labels["export_md"],
                data=st.session_state.skill_apply_result or "",
                file_name="skill_apply_result.md",
                mime="text/markdown",
                key="skill_apply_dl",
            )

    # ----------------- DASHBOARD TAB -----------------
    with tabs[4]:
        pipeline: PipelineState = st.session_state.pipeline
        runs = list(pipeline.history.values())
        total_latency = sum(r.latency_ms or 0 for r in runs)
        total_output_tokens = sum(r.output_tokens or 0 for r in runs)
        total_input_tokens = sum(r.input_tokens or 0 for r in runs)
        completed = sum(1 for r in runs if r.status == "completed")
        errored = sum(1 for r in runs if r.status == "error")
        running = sum(1 for r in runs if r.status == "running")
        total_agents = len(pipeline.agents) or 1
        progress_pct = completed / total_agents * 100

        st.markdown("### Pipeline Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Progress", f"{progress_pct:.0f}%")
        c2.metric("Total Runtime (ms)", f"{total_latency:.0f}")
        c3.metric("Input Tokens", total_input_tokens or "–")
        c4.metric("Output Tokens", total_output_tokens or "–")

        c5, c6, c7 = st.columns(3)
        c5.metric("Completed Agents", completed)
        c6.metric("Running Agents", running)
        c7.metric("Errored Agents", errored)

        if pipeline.agents:
            names = [a.name for a in pipeline.agents]
            latencies = [
                pipeline.history.get(a.id, AgentRunState()).latency_ms or 0
                for a in pipeline.agents
            ]
            outputs = [
                pipeline.history.get(a.id, AgentRunState()).output_tokens or 0
                for a in pipeline.agents
            ]
            statuses = [
                pipeline.history.get(a.id, AgentRunState()).status
                for a in pipeline.agents
            ]

            colA, colB = st.columns(2)
            with colA:
                fig_lat = px.bar(
                    x=names,
                    y=latencies,
                    labels={"x": "Agent", "y": "Latency (ms)"},
                    title="Agent Latency",
                    color=latencies,
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig_lat, use_container_width=True)
            with colB:
                fig_tok = px.bar(
                    x=names,
                    y=outputs,
                    labels={"x": "Agent", "y": "Output Tokens"},
                    title="Token Usage per Agent",
                    color=outputs,
                    color_continuous_scale="Plasma",
                )
                st.plotly_chart(fig_tok, use_container_width=True)

            # Status distribution
            status_counts = {
                "idle": statuses.count("idle"),
                "running": statuses.count("running"),
                "completed": statuses.count("completed"),
                "error": statuses.count("error"),
            }
            st.markdown("#### Agent Status Distribution")
            fig_status = px.pie(
                names=list(status_counts.keys()),
                values=list(status_counts.values()),
                title="Agent Status",
                color=list(status_counts.keys()),
                color_discrete_map={
                    "idle": "#9ca3af",
                    "running": "#f97316",
                    "completed": "#22c55e",
                    "error": "#ef4444",
                },
            )
            st.plotly_chart(fig_status, use_container_width=True)

        st.markdown("### Logs")
        for entry in reversed(st.session_state.log_entries[-200:]):
            color = {"info": "#60a5fa", "success": "#22c55e", "error": "#ef4444"}.get(
                entry["level"], "#9ca3af"
            )
            st.markdown(
                f"<span style='color:{color};font-weight:600'>[{entry['ts']}] {entry['level'].upper()}</span> "
                f"<span>{entry['message']}</span>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
