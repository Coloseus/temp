#!/usr/bin/env python3
"""
build_batches_v5.py — Smart Truncation + Keyword Contamination Guard
New fixes on top of v4 (build_batches_fixed.py):

FIX-A: local_snippet is now built directly from raw file lines using the
        finding's original line number — BEFORE smart_source annotation and
        truncation. Stored as batch["local_snippet"]. This eliminates the
        offset-drift bug where run_inference sliced the wrong lines from the
        annotated/truncated source_code string.

FIX-B: guarded_by_switch_tag now scans the ENTIRE enclosing function body
        (from ctags function start → next function start) instead of ±20 lines.
        This fixes the nethash false-positive where switch(n->type) was >20
        lines away from the individual cast line.

FIX-C: Smart truncation keyword contamination guard.
        Problem: the 200-line header block may contain FIXME/thread-unsafe
        comments for UNRELATED code. When the model sees those keywords in
        the header, it can mislabel the actual finding.
        Fix: in build_smart_source, the header block has hazard keywords
        replaced with a neutral placeholder. The ±context window around the
        finding is left untouched. Hazard keyword matching in the
        deterministic override layer now operates ONLY on local_snippet
        (the raw ±5 lines), never on full source_code.
"""

import json
import os
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_FILE  = "merged_candidates.json"
OUTPUT_FILE = "llm_triage_batches.json"
BIRD_SRC_DIR = "/home/jack/Work/StaticCodeAnalysis/BIRD/AI_Triage/bird-2.17.1"
TAGS_FILE   = "tags"
CG_FILE     = "bird.bc.callgraph.dot"

MAX_TEST_FINDINGS = 1000
MAX_FILE_LINES    = 6000
MAX_RAG_CHARS     = 8000
SNIPPET_CONTEXT   = 30   # lines in RAG definition blocks
LOCAL_SNIPPET_CTX = 5    # ±5 lines for local_snippet (keyword matching zone)
HEADER_LINES      = 200  # first N lines always included in smart_source
WINDOW_CONTEXT    = 600  # ±lines around finding in smart_source

# MODE 1: only high-value security-ish findings (no dead-code/backdoor heuristics)
# MODE 2: include potential backdoor indicators like unused/static code as well.
MODE = 1  # change to 2 to include SECURITY_RULES even if severity is low

ALLOWED_SEVERITIES = {"high", "critical", "blocker", "major", "warning"}
IGNORE_PATHS = ["test/", "doc/", "misc/", "distro/", "docker/", "tools/", "obj/"]

# Rules that are pure style/maintenance — not worth GPU time
IGNORE_RULES = {
    "c:S3776", "c:S134", "c:S999", "c:S954", "c:S1186", "c:S1121",
    "c:S1911", "c:S3358", "c:S125", "c:S923", "c:S5281", "c:S959",
    "c:S1820", "c:S3562", "c:S859", "c:S5270", "c:S936", "c:S1066", "c:S1134",
    "cpp/long-switch", "cpp/loop-variable-changed",
    "cpp/declaration-hides-parameter", "cpp/declaration-hides-variable",
    "cpp/local-variable-hides-global-variable",
    "cpp/use-of-goto", "cpp/commented-out-code", "cpp/trivial-switch",
}

# Rules that may indicate hidden/backdoor-like code paths even if tools label them as style.
SECURITY_RULES = {
    "unusedFunction",
    "staticFunction",
    "unusedStructMember",
    "unusedLabelConfiguration",
    "unreadVariable",
}

# Obvious partial-analysis noise from tools not seeing the full project
NOISE_RULES = {
    "missingInclude",
    "missingIncludeSystem",
    "unknownMacro",
}
C_KEYWORDS = {
    "int", "char", "void", "struct", "if", "else", "while", "for",
    "return", "static", "const", "unsigned", "long", "short", "double",
    "float", "typedef", "enum", "union", "switch", "case", "break",
    "continue", "default", "extern", "inline", "register", "sizeof",
    "volatile", "auto", "do", "goto",
}

# FIX-C: hazard keywords that must NOT leak from unrelated header code
# into the model's context. We mask them in the header block of smart_source.
_HAZARD_KEYWORDS_RE = re.compile(
    r'\b(thread[- ]unsafe|non[- ]atomic|data\s+race|race\s+condition'
    r'|buffer\s+overflow|stack\s+overflow|use.after.free|UAF'
    r'|injection|deadlock)\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Phase 1 — AST (ctags)
# ---------------------------------------------------------------------------
def load_ast() -> dict:
    ast_map: dict[str, list] = defaultdict(list)
    if not os.path.exists(TAGS_FILE):
        print(f"[WARN] Tags file not found: {TAGS_FILE}")
        return ast_map

    with open(TAGS_FILE, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            name     = parts[0]
            filepath = parts[1]
            if any(ign in filepath for ign in ["obj/", "test/", "doc/"]):
                continue

            line_num = -1
            kind     = ""
            for p in parts[3:]:
                if p.startswith("line:"):
                    try:
                        line_num = int(p.split(":")[1])
                    except ValueError:
                        pass
                elif p.startswith("kind:"):
                    kind = p.replace("kind:", "").strip()
                elif len(p) == 1 and p.isalpha():
                    kind = p

            if line_num == -1 or name in C_KEYWORDS or len(name) <= 3:
                continue

            kind_upper = kind.upper()
            if kind_upper in {"F", "FUNCTION", "S", "STRUCT", "M", "MACRO",
                               "D", "DEFINE", "T", "TYPEDEF", "G", "E", "ENUMERATOR"}:
                ast_map[name].append({"file": filepath, "line": line_num, "kind": kind})

    # FIX-B: also build a function-boundary map:
    # func_ranges[filepath] = sorted list of (start_line, func_name)
    # Used later to find the enclosing function body for switch scan.
    func_ranges: dict[str, list] = defaultdict(list)
    for name, entries in ast_map.items():
        for e in entries:
            if e["kind"].upper() in {"F", "FUNCTION"}:
                func_ranges[e["file"]].append((e["line"], name))

    for fp in func_ranges:
        func_ranges[fp].sort()

    print(f"[AST] {sum(len(v) for v in ast_map.values())} entries "
          f"({len(ast_map)} unique symbols), "
          f"{sum(len(v) for v in func_ranges.values())} function boundaries")
    return dict(ast_map), dict(func_ranges)

# ---------------------------------------------------------------------------
# Phase 2 — Call-graph
# ---------------------------------------------------------------------------
def load_callgraph():
    callers_of: dict[str, list] = defaultdict(list)
    callees_of: dict[str, list] = defaultdict(list)

    if not os.path.exists(CG_FILE):
        print(f"[WARN] Call-graph file not found: {CG_FILE}")
        return callers_of, callees_of

    with open(CG_FILE, "r", errors="ignore") as fh:
        content = fh.read()

    node_to_name: dict[str, str] = {}
    for m in re.finditer(r'(Node0x[0-9a-fA-F]+)[^\n]*label="\{([^}]+)\}"', content):
        node_id   = m.group(1)
        func_name = m.group(2).split("\\|")[0].strip()
        func_name = re.sub(r'\.(cold|\d+)$', '', func_name)
        node_to_name[node_id] = func_name

    for m in re.finditer(r'(Node0x[0-9a-fA-F]+)\s*->\s*(Node0x[0-9a-fA-F]+)', content):
        caller = node_to_name.get(m.group(1))
        callee  = node_to_name.get(m.group(2))
        if caller and callee and caller != callee:
            callers_of[callee].append(caller)
            callees_of[caller].append(callee)

    callers_of = {k: list(dict.fromkeys(v)) for k, v in callers_of.items()}
    callees_of = {k: list(dict.fromkeys(v)) for k, v in callees_of.items()}
    print(f"[CG] {len(node_to_name)} nodes")
    return callers_of, callees_of

# ---------------------------------------------------------------------------
# Phase 3 — Code snippet for RAG
# ---------------------------------------------------------------------------
def get_code_snippet(filepath: str, line_num: int, context: int = SNIPPET_CONTEXT) -> str:
    try:
        full_path = os.path.join(BIRD_SRC_DIR, filepath)
        with open(full_path, "r", errors="ignore") as fh:
            lines = fh.readlines()
        start = max(0, line_num - 1)
        end   = min(len(lines), line_num - 1 + context)
        return "".join(f"{i+1:4d} | {lines[i]}" for i in range(start, end)).rstrip()
    except Exception:
        return "Definition unavailable."

# ---------------------------------------------------------------------------
# Phase 4 — RAG context
# ---------------------------------------------------------------------------
def build_rag_context(src_lines, target_line_idx, ast_map, callers_of, callees_of):
    start = max(0, target_line_idx - 10)
    end   = min(len(src_lines), target_line_idx + 10)
    window_text = "".join(src_lines[start:end])

    words = set(re.findall(r'[a-zA-Z_]\w{3,}', window_text))
    words -= C_KEYWORDS

    context_blocks = []
    total_chars = 0

    for word in sorted(words):
        if total_chars >= MAX_RAG_CHARS:
            break
        if word not in ast_map:
            continue
        for entry in ast_map[word][:2]:
            kind_str = entry["kind"].upper()
            snippet  = get_code_snippet(entry["file"], entry["line"])
            block    = (f"--- [{kind_str}] '{word}' "
                        f"({entry['file']}:{entry['line']}) ---\n{snippet}")
            if kind_str in {"F", "FUNCTION"}:
                callers = callers_of.get(word, [])[:6]
                callees = callees_of.get(word, [])[:6]
                if callers:
                    block += f"\n[CG CALLERS]: {', '.join(callers)}"
                if callees:
                    block += f"\n[CG CALLEES]: {', '.join(callees)}"
            context_blocks.append(block)
            total_chars += len(block)
            if total_chars >= MAX_RAG_CHARS:
                break

    return "\n\n".join(context_blocks) if context_blocks else "No external dependencies detected."

# ---------------------------------------------------------------------------
# Phase 5 — Semantic profile
# FIX-B: guarded_by_switch_tag now scans the full enclosing function body
# ---------------------------------------------------------------------------
_ALLOCA_RE    = re.compile(r'\balloca\s*\(')
_LOOP_RE      = re.compile(r'\b(while|for|do)\s*[\({]')
_SWITCH_RE    = re.compile(r'\bswitch\s*\(')
_MUTEX_RE     = re.compile(r'\b(pthread_mutex|bfd_lock|rt_lock|\w+_lock)\b')
_CAST_RE      = re.compile(r'\(\s*struct\s+\w+\s*\*\s*\)')
_SPRINTF_RE   = re.compile(r'\b(sprintf|strcpy|strcat|gets|scanf)\s*\(')
_MEMCPY_RE    = re.compile(r'\bmemcpy\s*\(')
_FIXME_HAZARD = re.compile(
    r'FIXME.*?(thread.?unsafe|race|concurrent|locking|mutex|overflow|inject|execut)',
    re.IGNORECASE
)
_NETWORK_INPUT = re.compile(r'\b(recv|recvfrom|recvmsg|read|bgp_parse|decode_nlri)\b')

def get_enclosing_function_body(src_lines, target_line_idx, func_ranges, filepath):
    """
    FIX-B: Use ctags function boundaries to find the full body of the function
    that contains target_line_idx.  Returns the body as a single string.
    Falls back to ±60 lines if ctags data is unavailable.
    """
    # Normalise filepath key (ctags paths may omit leading ./)
    key = filepath.lstrip("./")
    ranges = func_ranges.get(key) or func_ranges.get("./" + key, [])

    if ranges:
        # Find the last function that starts at or before the target line
        enclosing_start = None
        for (fstart, fname) in ranges:
            if fstart - 1 <= target_line_idx:
                enclosing_start = fstart - 1  # convert to 0-based
            else:
                break

        if enclosing_start is not None:
            # Find the next function start to bound the body
            enclosing_end = len(src_lines)
            for (fstart, _) in ranges:
                if fstart - 1 > target_line_idx:
                    enclosing_end = fstart - 1
                    break
            return "".join(src_lines[enclosing_start:enclosing_end])

    # Fallback: ±60 lines
    start = max(0, target_line_idx - 60)
    end   = min(len(src_lines), target_line_idx + 60)
    return "".join(src_lines[start:end])


def build_semantic_profile(src_lines, target_line_idx, func_ranges, filepath):
    # ±20 lines for most checks
    start = max(0, target_line_idx - 20)
    end   = min(len(src_lines), target_line_idx + 20)
    window = "".join(src_lines[start:end])

    finding_line = src_lines[target_line_idx] if target_line_idx < len(src_lines) else ""

    contains_alloca = bool(_ALLOCA_RE.search(finding_line))
    inside_loop     = bool(_LOOP_RE.search(window))
    has_mutex       = bool(_MUTEX_RE.search(window))
    has_cast        = bool(_CAST_RE.search(window))
    has_unsafe_str  = bool(_SPRINTF_RE.search(window))
    has_memcpy      = bool(_MEMCPY_RE.search(window))
    developer_hazard = bool(_FIXME_HAZARD.search(window))
    network_input   = bool(_NETWORK_INPUT.search(window))

    # FIX-B: scan entire enclosing function body for switch guard
    func_body = get_enclosing_function_body(src_lines, target_line_idx,
                                             func_ranges, filepath)
    guarded_by_switch = bool(_SWITCH_RE.search(func_body))
    guarded_by_switch_tag = guarded_by_switch and has_cast and not network_input

    return {
        "contains_alloca":        contains_alloca,
        "inside_loop":            inside_loop,
        "guarded_by_switch_tag":  guarded_by_switch_tag,
        "has_mutex_nearby":       has_mutex,
        "has_pointer_cast":       has_cast,
        "has_unsafe_string_func": has_unsafe_str,
        "has_memcpy":             has_memcpy,
        "developer_hazard_notes": developer_hazard,
        "network_input_nearby":   network_input,
    }

# ---------------------------------------------------------------------------
# Phase 6 — local_snippet (FIX-A) and smart_source (FIX-C)
# ---------------------------------------------------------------------------
def build_local_snippet(lines, target_line_idx, ctx=LOCAL_SNIPPET_CTX):
    """
    FIX-A: Built from raw lines[], before any annotation or truncation.
    This is the ONLY zone used for hazard keyword matching in run_inference.
    Stored separately so run_inference never has to slice source_code.
    """
    start = max(0, target_line_idx - ctx)
    end   = min(len(lines), target_line_idx + ctx + 1)
    return "".join(f"{i+1:5d} | {lines[i]}" for i in range(start, end))


def build_smart_source(lines, target_line_idx):
    """
    FIX-C: The HEADER block (first 200 lines) has hazard keywords masked
    so unrelated FIXME/thread-unsafe comments in distant code can't
    contaminate the model's verdict on the current finding.
    The WINDOW block around the finding is left completely untouched.
    """
    def fmt(i):
        return f"{i+1:5d} | {lines[i]}"

    if len(lines) <= MAX_FILE_LINES:
        # Small file: mask hazard keywords only in lines outside the finding window
        window_start = max(0, target_line_idx - WINDOW_CONTEXT)
        window_end   = min(len(lines), target_line_idx + WINDOW_CONTEXT)
        parts = []
        for i in range(len(lines)):
            raw = fmt(i)
            if window_start <= i <= window_end:
                parts.append(raw)      # finding window — untouched
            else:
                parts.append(_HAZARD_KEYWORDS_RE.sub("[HAZARD-NOTE-UNRELATED]", raw))
        return "".join(parts)

    # Large file: header + window
    header_end   = HEADER_LINES
    window_start = max(header_end, target_line_idx - WINDOW_CONTEXT)
    window_end   = min(len(lines), target_line_idx + WINDOW_CONTEXT)

    # Header: mask hazard keywords (unrelated code)
    header_block = "".join(
        _HAZARD_KEYWORDS_RE.sub("[HAZARD-NOTE-UNRELATED]", fmt(i))
        for i in range(header_end)
    )

    parts = [header_block]
    if window_start > header_end:
        parts.append(f"\n\n... [{window_start - header_end} lines omitted] ...\n\n")

    # Window around finding: completely untouched
    parts.append("".join(fmt(i) for i in range(window_start, window_end)))

    if window_end < len(lines):
        parts.append(f"\n\n... [{len(lines) - window_end} lines omitted] ...\n\n")

    return "".join(parts)

# ---------------------------------------------------------------------------
# Phase 7 — Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("build_batches_v5.py")
    print("=" * 60)

    ast_map, func_ranges = load_ast()
    callers_of, callees_of = load_callgraph()

    with open(INPUT_FILE, "r") as fh:
        raw_findings = json.load(fh)

    # Filter + deduplicate
    filtered = []
    for f in raw_findings:
        rule_id = f.get("Rule_ID", "")
        sev     = str(f.get("Severity", "")).lower()
        filepath = f.get("File", "")

        # Drop obvious non-prod paths
        if any(ign in filepath for ign in IGNORE_PATHS):
            continue

        # Drop pure style/maintenance rules entirely
        if rule_id in IGNORE_RULES:
            continue

        # Drop tool-noise artefacts from partial analysis
        if rule_id in NOISE_RULES:
            continue

        if MODE == 1:
            # Strict: only high-value severities
            if sev not in ALLOWED_SEVERITIES:
                continue
        else:
            # MODE 2: keep SECURITY_RULES even if severity is low (style/info)
            if (sev not in ALLOWED_SEVERITIES) and (rule_id not in SECURITY_RULES):
                continue

        filtered.append(f)

    seen_keys: set = set()
    deduped = []
    for f in filtered:
        key = (f.get("File", ""), f.get("Line", 0), f.get("Rule_ID", ""))
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(f)

    pilot = deduped[:MAX_TEST_FINDINGS]
    print(f"[FILTER] {len(raw_findings)} total → {len(filtered)} filtered "
          f"→ {len(deduped)} deduped → {len(pilot)} pilot findings")

    by_file: dict[str, list] = defaultdict(list)
    for f in pilot:
        by_file[f["File"].lstrip("./")].append(f)

    batches = []
    batch_id = 0
    skipped  = 0

    for filepath, file_findings in by_file.items():
        full_path = os.path.join(BIRD_SRC_DIR, filepath)
        if not os.path.exists(full_path):
            print(f"[SKIP] Source not found: {full_path}")
            skipped += len(file_findings)
            continue

        with open(full_path, "r", errors="ignore") as fh:
            lines = fh.readlines()

        for finding in file_findings:
            line_idx = finding["Line"] - 1
            if not (0 <= line_idx < len(lines)):
                # Line 0 = file-level finding (e.g. missing-header-guard) → clamp to line 1
                if finding["Line"] == 0:
                    line_idx = 0
                else:
                    print(f"[SKIP] Line {finding['Line']} out of bounds in {filepath} "
                        f"(file has {len(lines)} lines) — Rule: {finding.get('Rule_ID')}")
                    skipped += 1
                    continue

            rag_context      = build_rag_context(lines, line_idx, ast_map,
                                                  callers_of, callees_of)
            semantic_profile = build_semantic_profile(lines, line_idx,
                                                       func_ranges, filepath)

            # FIX-A: local_snippet from raw lines, stored separately
            local_snippet = build_local_snippet(lines, line_idx)

            # FIX-C: smart_source with header hazard-keyword masking
            smart_source = build_smart_source(lines, line_idx)

            finding["RAG_Context"]      = rag_context
            finding["Semantic_Profile"] = semantic_profile
            # local_snippet attached to finding so run_inference can read it directly
            finding["local_snippet"]    = local_snippet

            batch_id += 1
            batches.append({
                "batch_id":    batch_id,
                "file":        filepath,
                "source_code": smart_source,
                "findings":    [finding],
            })

    with open(OUTPUT_FILE, "w") as fh:
        json.dump(batches, fh, indent=4)

    print(f"[DONE] {len(batches)} batches written to {OUTPUT_FILE} "
          f"({skipped} skipped)")

if __name__ == "__main__":
    main()
