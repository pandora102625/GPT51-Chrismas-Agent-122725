import os
import time
import uuid
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


# ------------------- THEME ENGINE -----------------

FLOWER_THEMES = {
    "Sakura": {"primary": "#ff9aae", "secondary": "#ffe4ec", "accent": "#ff6b9d"},
    "Rose": {"primary": "#c21f3a", "secondary": "#ffe3e8", "accent": "#ff6f91"},
    "Lavender": {"primary": "#8e7cc3", "secondary": "#f3ecff", "accent": "#b4a7d6"},
    "Sunflower": {"primary": "#f1c232", "secondary": "#fff5cc", "accent": "#f6b26b"},
    "Lotus": {"primary": "#ff99cc", "secondary": "#ffe6f2", "accent": "#ff66b3"},
    "Orchid": {"primary": "#b565a7", "secondary": "#f7d9ff", "accent": "#e066ff"},
    "Peony": {"primary": "#e06666", "secondary": "#fde3e3", "accent": "#cc0000"},
    "Camellia": {"primary": "#d9534f", "secondary": "#fbe4e2", "accent": "#c9302c"},
    "Magnolia": {"primary": "#f6b26b", "secondary": "#fff2e5", "accent": "#e69138"},
    "Hydrangea": {"primary": "#6fa8dc", "secondary": "#e3f2fd", "accent": "#3c78d8"},
    "Cherry Blossom": {"primary": "#ffb3c6", "secondary": "#ffe6f0", "accent": "#ff6f91"},
    "Gardenia": {"primary": "#a4c2f4", "secondary": "#ecf3ff", "accent": "#6d9eeb"},
    "Jasmine": {"primary": "#f9cb9c", "secondary": "#fff5e6", "accent": "#f6b26b"},
    "Iris": {"primary": "#674ea7", "secondary": "#efe5ff", "accent": "#8e7cc3"},
    "Poppy": {"primary": "#e06666", "secondary": "#ffe0e0", "accent": "#cc0000"},
    "Daisy": {"primary": "#ffd966", "secondary": "#fff9e6", "accent": "#f1c232"},
    "Marigold": {"primary": "#f6b26b", "secondary": "#fff0de", "accent": "#e69138"},
    "Bluebell": {"primary": "#6d9eeb", "secondary": "#e5f1ff", "accent": "#3c78d8"},
    "Tulip": {"primary": "#e06666", "secondary": "#ffe2e2", "accent": "#cc0000"},
    "Wisteria": {"primary": "#b4a7d6", "secondary": "#f3ecff", "accent": "#8e7cc3"},
}


def apply_theme(theme_name: str, dark_mode: bool):
    theme = FLOWER_THEMES.get(theme_name, FLOWER_THEMES["Sakura"])
    bg = "#05030a" if dark_mode else "#ffffff"
    text = "#f7f7f7" if dark_mode else "#111827"

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
        background: radial-gradient(circle at top, var(--color-secondary) 0, var(--bg-color) 60%);
        color: var(--text-color);
    }}
    .audit-card {{
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1rem 1.2rem;
        background: rgba(0,0,0,0.02);
    }}
    .status-badge {{
        display:inline-flex;
        align-items:center;
        gap:0.3rem;
        padding:0.1rem 0.4rem;
        border-radius:999px;
        font-size:0.72rem;
        font-weight:600;
        text-transform:uppercase;
        letter-spacing:0.04em;
    }}
    .status-idle {{ background:#e5e7eb33; color:#6b7280; }}
    .status-running {{ background:var(--color-accent); color:white; }}
    .status-completed {{ background:#16a34a22; color:#16a34a; }}
    .status-error {{ background:#ef444422; color:#ef4444; }}
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
    """
    Safely extract text from a google-generativeai response.
    Avoids resp.text quick accessor, which can raise if no valid Part exists.
    """
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

    # Fallback to quick accessor; swallow any errors
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
    """
    Sample Grok (xAI) call, adapted from the official snippet.
    """
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
    """
    Safely load agents.yaml. If the file is missing or invalid, return
    an empty/default PipelineState instead of crashing.
    """
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
                max_tokens=int(a.get("max_tokens", 2048)),
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
        "title": "AuditFlow AI – Medical Device Quality Audit Pipeline",
        "pipeline_tab": "Pipeline",
        "smart_tab": "Smart Replace",
        "dashboard_tab": "Dashboard & Logs",
        "template": "Audit Template (Markdown)",
        "observations": "Raw Observations",
        "run_pipeline": "Run Full Pipeline",
        "run_agent": "Run This Agent Only",
        "agent_output": "Agent Output (Editable)",
        "api_keys": "API Keys",
        "gemini_key": "Gemini API Key",
        "openai_key": "OpenAI API Key",
        "anthropic_key": "Anthropic API Key",
        "grok_key": "XAI (Grok) API Key",
        "theme": "Flower Theme",
        "dark_mode": "Dark Mode",
        "language": "Language",
        "style_wheel": "Lucky Flower Wheel",
        "smart_template": "Template A (Markdown Structure)",
        "smart_list": "List B (Data / Observations)",
        "smart_run": "Run Smart Replacement",
        "smart_preview": "Preview (HTML)",
        "smart_raw": "Raw (Markdown)",
        "export_md": "Download as Markdown",
    },
    "zh": {
        "title": "AuditFlow AI – 醫療器材品質稽核智能流程",
        "pipeline_tab": "稽核流程管線",
        "smart_tab": "智慧套版 (Smart Replace)",
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
        "theme": "花卉主題",
        "dark_mode": "深色模式",
        "language": "介面語言",
        "style_wheel": "幸運花卉轉盤",
        "smart_template": "Template A（Markdown 結構）",
        "smart_list": "List B（資料 / 觀察清單）",
        "smart_run": "執行智慧套版",
        "smart_preview": "預覽 (HTML)",
        "smart_raw": "原始 Markdown",
        "export_md": "下載 Markdown",
    },
}


# ------------------- WOW STATUS BADGE --------------

def render_status_badge(status: str):
    label_map = {
        "idle": "IDLE",
        "running": "RUNNING",
        "completed": "DONE",
        "error": "ERROR",
    }
    html = f'<span class="status-badge status-{status}">{label_map.get(status, status).upper()}</span>'
    st.markdown(html, unsafe_allow_html=True)


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
        st.session_state.theme_name = "Sakura"
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


def log_event(level: str, message: str):
    st.session_state.log_entries.append(
        {"level": level, "message": message, "ts": time.strftime("%H:%M:%S")}
    )


# ------------------- STREAMLIT APP -----------------

def main():
    st.set_page_config(
        page_title="AuditFlow AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    lang = st.session_state.ui_language
    labels = UI_LABELS[lang]

    # Sidebar: API keys, language, theme, YAML
    with st.sidebar:
        st.markdown(f"### {labels['api_keys']}")

        gemini_ui = st.text_input(labels["gemini_key"], type="password")
        openai_ui = st.text_input(labels["openai_key"], type="password")
        anthropic_ui = st.text_input(labels["anthropic_key"], type="password")
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

        theme_name = st.selectbox(labels["theme"], options=list(FLOWER_THEMES.keys()))
        st.session_state.theme_name = theme_name
        dark_mode = st.toggle(labels["dark_mode"], value=st.session_state.dark_mode)
        st.session_state.dark_mode = dark_mode

        if st.button(labels["style_wheel"]):
            import random

            choice = random.choice(list(FLOWER_THEMES.keys()))
            st.session_state.theme_name = choice
            log_event("info", f"Lucky wheel selected theme: {choice}")

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
                            max_tokens=int(a.get("max_tokens", 2048)),
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

    # Effective API keys
    api_keys = {
        "gemini": get_api_key("GEMINI_API_KEY", st.session_state.manual_api_keys["gemini"]),
        "openai": get_api_key("OPENAI_API_KEY", st.session_state.manual_api_keys["openai"]),
        "anthropic": get_api_key("ANTHROPIC_API_KEY", st.session_state.manual_api_keys["anthropic"]),
        "grok": get_api_key("XAI_API_KEY", st.session_state.manual_api_keys["grok"]),
    }

    tabs = st.tabs([labels["pipeline_tab"], labels["smart_tab"], labels["dashboard_tab"]])

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
                        "Provider",
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
                    if provider == "gemini":
                        models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
                    elif provider == "openai":
                        models = ["gpt-5-nano", "gpt-4o-mini", "gpt-4.1-mini"]
                    elif provider == "anthropic":
                        models = ["claude-3.5-haiku", "claude-3.5-sonnet"]
                    else:
                        models = ["grok-4-fast-reasoning", "grok-3-mini"]
                    model = st.selectbox(
                        "Model",
                        options=models,
                        index=models.index(agent.model) if agent.model in models else 0,
                        key=f"model_{agent.id}",
                    )
                    agent.model = model
                with c3:
                    agent.max_tokens = st.number_input(
                        "Max tokens",
                        min_value=256,
                        max_value=32000,
                        value=int(agent.max_tokens),
                        step=256,
                        key=f"max_tokens_{agent.id}",
                    )
                with c4:
                    agent.temperature = st.slider(
                        "Temp",
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

                # ----------------- Status & run -----------------
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

                # -------------- Editable output (fixed) --------------
                output_key = f"out_{agent.id}"

                if output_key not in st.session_state:
                    st.session_state[output_key] = state.output or ""

                if state.status in ("completed", "error"):
                    st.session_state[output_key] = state.output or st.session_state[output_key]

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
                "Provider",
                options=["gemini", "openai", "anthropic", "grok"],
                key="sr_provider",
            )

            if sr_provider == "gemini":
                models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
            elif sr_provider == "openai":
                models = ["gpt-5-nano", "gpt-4o-mini", "gpt-4.1-mini"]
            elif sr_provider == "anthropic":
                models = ["claude-3.5-haiku", "claude-3.5-sonnet"]
            else:
                models = ["grok-4-fast-reasoning", "grok-3-mini"]

            sr_model = st.selectbox("Model", options=models, key="sr_model")
            sr_max_tokens = st.number_input(
                "Max tokens",
                min_value=256,
                max_value=32000,
                value=4096,
                step=256,
                key="sr_max_tokens",
            )
            sr_temp = st.slider(
                "Temp",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key="sr_temp",
            )

            provider_key_map = {
                "gemini": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "grok": "XAI_API_KEY",
            }
            effective_api_keys = {
                "gemini": api_keys["gemini"],
                "openai": api_keys["openai"],
                "anthropic": api_keys["anthropic"],
                "grok": api_keys["grok"],
            }

            if not effective_api_keys[sr_provider]:
                env_name = provider_key_map[sr_provider]
                st.warning(
                    f"No API key found for provider `{sr_provider}`. "
                    f"Set the environment variable `{env_name}` or enter a key in the sidebar."
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

    # ----------------- DASHBOARD TAB -----------------
    with tabs[2]:
        pipeline: PipelineState = st.session_state.pipeline
        runs = list(pipeline.history.values())
        total_latency = sum(r.latency_ms or 0 for r in runs)
        total_output_tokens = sum(r.output_tokens or 0 for r in runs)
        total_input_tokens = sum(r.input_tokens or 0 for r in runs)
        completed = sum(1 for r in runs if r.status == "completed")
        total_agents = len(pipeline.agents) or 1
        progress_pct = completed / total_agents * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Progress", f"{progress_pct:.0f}%")
        c2.metric("Total Runtime (ms)", f"{total_latency:.0f}")
        c3.metric("Input Tokens", total_input_tokens or "–")
        c4.metric("Output Tokens", total_output_tokens or "–")

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
