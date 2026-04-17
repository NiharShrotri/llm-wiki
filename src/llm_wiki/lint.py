"""Wiki lint — health checks that find contradictions, orphans, broken links,
malformed frontmatter, and other issues that creep into a knowledge base.

Checks are categorized by severity:

  ERROR    — things that break linking or make pages unusable
  WARNING  — stylistic or structural issues worth cleaning up
  INFO     — suggestions and observations

Fast checks (the default) run entirely in Python and take a few seconds.
Deep checks (--deep) use Qwen3 to detect contradictions across pairs of
pages that share entities/concepts — much slower, opt-in only.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from . import config as cfg
from . import page_writer
from . import slugify


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class CheckId(str, Enum):
    BROKEN_WIKILINK = "broken_wikilink"
    ORPHAN_PAGE = "orphan_page"
    MISSING_FRONTMATTER = "missing_frontmatter"
    INVALID_FRONTMATTER = "invalid_frontmatter"
    MALFORMED_WIKILINK = "malformed_wikilink"
    MISSING_CONCEPT_PAGE = "missing_concept_page"
    STALE_SOURCE_REF = "stale_source_ref"
    NOISE_IN_SYNTHESIS = "noise_in_synthesis"
    CONTRADICTION = "contradiction"


@dataclass
class LintIssue:
    """A single issue found during linting."""

    check: CheckId
    severity: Severity
    page: str               # Relative to wiki root, e.g. 'entities/qwen.md'
    message: str
    suggestion: str = ""
    fixable: bool = False   # True if --fix can auto-resolve it
    context: dict[str, Any] = field(default_factory=dict)  # check-specific data


@dataclass
class LintReport:
    """The complete result of a lint run."""

    issues: list[LintIssue] = field(default_factory=list)
    pages_checked: int = 0
    fast_checks_run: list[str] = field(default_factory=list)
    deep_check_run: bool = False
    auto_fixed: int = 0
    duration_seconds: float = 0.0

    @property
    def errors(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def infos(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == Severity.INFO]

    @property
    def health_score(self) -> int:
        """A 0-100 score based on issue density. 100 = perfectly clean."""
        if self.pages_checked == 0:
            return 100
        error_weight = 5
        warning_weight = 2
        info_weight = 1
        penalty = (
            len(self.errors) * error_weight
            + len(self.warnings) * warning_weight
            + len(self.infos) * info_weight
        )
        # Normalize: assume ~10 penalty per page on average is "bad"
        max_penalty = self.pages_checked * 10
        if max_penalty == 0:
            return 100
        score = max(0, 100 - int(100 * penalty / max_penalty))
        return score


# ---------------------------------------------------------------------------
# Page inventory (shared by many checks)
# ---------------------------------------------------------------------------


@dataclass
class PageInventory:
    """Cached state of the wiki: every page's path, frontmatter, and links."""

    pages: dict[str, page_writer.ParsedPage] = field(default_factory=dict)  # relpath -> parsed
    outgoing_links: dict[str, list[str]] = field(default_factory=dict)      # relpath -> [targets]
    incoming_links: dict[str, list[str]] = field(default_factory=dict)      # relpath -> [sources]
    all_slugs: set[str] = field(default_factory=set)                        # e.g. 'entities/qwen'
    raw_paths: set[str] = field(default_factory=set)                        # files in raw/


_NOISE_PAGES = {"index.md", "log.md"}

# Directory prefix → page type
_PAGE_TYPES = ("sources", "entities", "concepts", "synthesis")


def _build_inventory(paths: cfg.WikiPaths) -> PageInventory:
    """Walk wiki/ and build a cached PageInventory for all downstream checks."""
    inv = PageInventory()

    # 1. Walk wiki/ subdirectories
    for subdir in _PAGE_TYPES:
        d = paths.wiki / subdir
        if not d.exists():
            continue
        for md_path in sorted(d.glob("*.md")):
            if md_path.name.startswith(".") or md_path.name.startswith("lint-report-"):
                continue
            try:
                content = md_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parsed = page_writer.parse_page(content)
            relpath = f"{subdir}/{md_path.name}"
            inv.pages[relpath] = parsed

            # Also record the slug without .md, which is how wikilinks reference it
            slug_no_ext = f"{subdir}/{md_path.stem}"
            inv.all_slugs.add(slug_no_ext)
            inv.all_slugs.add(f"{subdir}/{md_path.stem}.md")  # support both forms

    # 2. Build forward + reverse link graphs
    for relpath, parsed in inv.pages.items():
        links = page_writer.extract_wikilinks(parsed.body)
        # Also check frontmatter for wikilink-like fields
        fm_sources = parsed.frontmatter.get("sources", []) or []
        if isinstance(fm_sources, list):
            for s in fm_sources:
                if isinstance(s, str) and s:
                    links.append(s)

        normalized = [_normalize_link(link) for link in links if link]
        inv.outgoing_links[relpath] = normalized

        for target in normalized:
            inv.incoming_links.setdefault(target, []).append(relpath)

    # 3. Scan raw/ for stale-ref checks
    if paths.raw.exists():
        for raw_path in paths.raw.rglob("*"):
            if raw_path.is_file() and not raw_path.name.startswith("."):
                try:
                    rel = str(raw_path.relative_to(paths.root))
                    inv.raw_paths.add(rel)
                except ValueError:
                    pass

    return inv


def _normalize_link(link: str) -> str:
    """Normalize a wikilink target for comparison.

    Strips .md suffix, qmd:// prefix, pipe-based aliases, and leading/trailing
    slashes. So all these become the same thing:

        entities/qwen
        entities/qwen.md
        qmd://llm-wiki-pages/entities/qwen
        /qmd://llm-wiki-pages/entities/qwen
        sources/quick-notes-on-qwen.md
        sources/quick-notes-on-qwen|Q Notes
    """
    if not link:
        return ""
    link = link.strip()
    # Pipe alias: [[foo|Foo Page]] — keep the target (left side)
    if "|" in link:
        link = link.split("|", 1)[0]
    # Strip .md suffix
    if link.endswith(".md"):
        link = link[:-3]
    # Strip qmd:// URI prefix with optional collection name
    link = re.sub(r"^/?qmd://[^/]+/", "", link)
    # Strip leading slashes
    link = link.lstrip("/")
    return link


# ---------------------------------------------------------------------------
# Individual checks (each returns a list of LintIssue)
# ---------------------------------------------------------------------------


def check_broken_wikilinks(inv: PageInventory) -> list[LintIssue]:
    """Flag [[wikilinks]] whose target page doesn't exist.

    If the target's basename matches an existing page in a different
    subdirectory, suggest the corrected path and mark the issue as fixable.
    Otherwise, it's a genuine error.
    """
    issues: list[LintIssue] = []

    # Build a reverse lookup: basename -> list of existing full slugs.
    # So 'quick-notes-on-qwen' maps to ['sources/quick-notes-on-qwen'] if
    # that page exists. We use this to suggest corrections for bare-basename
    # wikilinks like [[quick-notes-on-qwen]] that should be
    # [[sources/quick-notes-on-qwen]].
    basename_lookup: dict[str, list[str]] = {}
    for slug in inv.all_slugs:
        if "/" in slug and not slug.endswith(".md"):
            basename = slug.rsplit("/", 1)[1]
            basename_lookup.setdefault(basename, []).append(slug)

    for relpath, targets in inv.outgoing_links.items():
        for target in targets:
            if not target:
                continue
            if target in inv.all_slugs or f"{target}.md" in inv.all_slugs:
                continue

            # Try to recover: does the target's basename match an existing
            # page in some subdirectory?
            target_basename = target.rsplit("/", 1)[-1]
            candidates = basename_lookup.get(target_basename, [])

            if len(candidates) == 1:
                # Unambiguous single match — downgrade to warning and mark fixable
                correct = candidates[0]
                issues.append(
                    LintIssue(
                        check=CheckId.BROKEN_WIKILINK,
                        severity=Severity.WARNING,
                        page=relpath,
                        message=f"Broken wikilink: [[{target}]] (should be [[{correct}]])",
                        suggestion=(
                            f"Target exists at '{correct}' — run `wiki lint --fix` "
                            f"to auto-correct."
                        ),
                        fixable=True,
                        context={
                            "old_target": target,
                            "new_target": correct,
                            "location": "body",
                        },
                    )
                )
            elif len(candidates) > 1:
                # Ambiguous — multiple pages share the basename, can't guess
                options = ", ".join(f"[[{c}]]" for c in candidates)
                issues.append(
                    LintIssue(
                        check=CheckId.BROKEN_WIKILINK,
                        severity=Severity.ERROR,
                        page=relpath,
                        message=f"Broken wikilink: [[{target}]] (ambiguous)",
                        suggestion=f"Did you mean one of: {options}?",
                        fixable=False,
                        context={"target": target},
                    )
                )
            else:
                # Genuinely missing — no existing page with that basename
                issues.append(
                    LintIssue(
                        check=CheckId.BROKEN_WIKILINK,
                        severity=Severity.ERROR,
                        page=relpath,
                        message=f"Broken wikilink: [[{target}]]",
                        suggestion=f"Either create {target}.md or remove the link.",
                        fixable=False,
                        context={"target": target},
                    )
                )
    return issues


def check_orphan_pages(inv: PageInventory) -> list[LintIssue]:
    """Find pages with no incoming wikilinks from any other page.

    Source pages and index/log are exempt (they're entry points, not
    navigation targets). Synthesis pages are also exempt since they're
    often user-saved answers that don't need backlinks.
    """
    issues: list[LintIssue] = []
    for relpath in inv.pages:
        page_type = relpath.split("/", 1)[0] if "/" in relpath else ""
        if page_type in {"sources", "synthesis"}:
            continue
        slug_no_ext = relpath[:-3] if relpath.endswith(".md") else relpath
        incoming = inv.incoming_links.get(slug_no_ext, []) + inv.incoming_links.get(
            relpath, []
        )
        # Don't count self-references
        incoming = [i for i in incoming if i != relpath]
        if not incoming:
            issues.append(
                LintIssue(
                    check=CheckId.ORPHAN_PAGE,
                    severity=Severity.WARNING,
                    page=relpath,
                    message="No incoming wikilinks — page is an orphan in the graph.",
                    suggestion="Link to this page from a related entity/concept, or delete it.",
                    fixable=False,
                )
            )
    return issues


def check_frontmatter(inv: PageInventory) -> list[LintIssue]:
    """Verify every page has required frontmatter fields."""
    issues: list[LintIssue] = []
    required_by_type = {
        "entities": {"title", "type", "created", "updated"},
        "concepts": {"title", "type", "created", "updated"},
        "sources": {"title", "type", "created"},
        "synthesis": {"title", "type", "created"},
    }

    for relpath, parsed in inv.pages.items():
        page_type = relpath.split("/", 1)[0]
        required = required_by_type.get(page_type, set())
        if not parsed.frontmatter:
            issues.append(
                LintIssue(
                    check=CheckId.MISSING_FRONTMATTER,
                    severity=Severity.ERROR,
                    page=relpath,
                    message="Page has no YAML frontmatter.",
                    suggestion="Add frontmatter with title, type, created, updated.",
                    fixable=False,
                )
            )
            continue
        missing = required - set(parsed.frontmatter.keys())
        if missing:
            issues.append(
                LintIssue(
                    check=CheckId.INVALID_FRONTMATTER,
                    severity=Severity.WARNING,
                    page=relpath,
                    message=f"Frontmatter missing: {', '.join(sorted(missing))}",
                    suggestion="Add the missing fields manually or re-ingest.",
                    fixable=False,
                )
            )
    return issues


def check_malformed_wikilinks(inv: PageInventory, paths: cfg.WikiPaths) -> list[LintIssue]:
    """Find wikilinks with fixable formatting problems:

    - [[foo.md]] instead of [[foo]]
    - [[qmd://llm-wiki-pages/foo]] instead of [[foo]]
    - [[/foo]] with leading slash
    - frontmatter source entries with qmd:// URI prefixes
    """
    issues: list[LintIssue] = []

    # Use a raw pattern to extract ALL wikilink literals from the body
    body_pattern = re.compile(r"\[\[([^\]]+?)\]\]")

    for relpath, parsed in inv.pages.items():
        raw_body = parsed.body

        # Body-level wikilinks
        for match in body_pattern.finditer(raw_body):
            raw_link = match.group(1)
            normalized = _normalize_link(raw_link)
            # Extract the link target (before any |alias)
            target = raw_link.split("|", 1)[0].strip()
            if target != normalized and normalized:
                issues.append(
                    LintIssue(
                        check=CheckId.MALFORMED_WIKILINK,
                        severity=Severity.WARNING,
                        page=relpath,
                        message=f"Malformed wikilink: [[{target}]] should be [[{normalized}]]",
                        suggestion="Run `wiki lint --fix` to auto-correct.",
                        fixable=True,
                        context={
                            "old_target": target,
                            "new_target": normalized,
                            "location": "body",
                        },
                    )
                )

        # Frontmatter `sources` and `sources_consulted` list entries
        for key in ("sources", "sources_consulted"):
            values = parsed.frontmatter.get(key, [])
            if not isinstance(values, list):
                continue
            for val in values:
                if not isinstance(val, str) or not val:
                    continue
                normalized = _normalize_link(val)
                if val != normalized and normalized:
                    issues.append(
                        LintIssue(
                            check=CheckId.MALFORMED_WIKILINK,
                            severity=Severity.WARNING,
                            page=relpath,
                            message=f"Malformed frontmatter `{key}` entry: {val!r}",
                            suggestion="Run `wiki lint --fix` to strip URI prefixes and .md suffixes.",
                            fixable=True,
                            context={
                                "old_target": val,
                                "new_target": normalized,
                                "location": "frontmatter",
                                "field": key,
                            },
                        )
                    )
    return issues


def check_missing_concepts(inv: PageInventory, threshold: int = 3) -> list[LintIssue]:
    """Flag terms mentioned 3+ times across pages that don't have their own page.

    Looks at wikilink *targets* — if several pages link to [[something]] but
    'something' doesn't exist, that's a hint it deserves a page of its own.
    """
    issues: list[LintIssue] = []

    # Count broken link targets
    target_counts: Counter[str] = Counter()
    target_sources: dict[str, list[str]] = defaultdict(list)
    for relpath, targets in inv.outgoing_links.items():
        seen_in_this_page: set[str] = set()
        for target in targets:
            if not target:
                continue
            if target in inv.all_slugs or f"{target}.md" in inv.all_slugs:
                continue
            # Only count each target once per source page
            if target in seen_in_this_page:
                continue
            seen_in_this_page.add(target)
            target_counts[target] += 1
            target_sources[target].append(relpath)

    for target, count in target_counts.most_common():
        if count >= threshold:
            issues.append(
                LintIssue(
                    check=CheckId.MISSING_CONCEPT_PAGE,
                    severity=Severity.INFO,
                    page=target_sources[target][0],
                    message=(
                        f"'{target}' referenced by {count} pages but has no page of its own."
                    ),
                    suggestion=f"Consider creating {target}.md",
                    fixable=False,
                    context={"target": target, "referenced_by": target_sources[target]},
                )
            )
    return issues


def check_stale_source_refs(inv: PageInventory, paths: cfg.WikiPaths) -> list[LintIssue]:
    """Flag pages whose `sources:` frontmatter references files that no longer
    exist in raw/ or wiki/sources/.
    """
    issues: list[LintIssue] = []
    for relpath, parsed in inv.pages.items():
        sources = parsed.frontmatter.get("sources", []) or []
        if not isinstance(sources, list):
            continue
        for src in sources:
            if not isinstance(src, str):
                continue
            normalized = _normalize_link(src)
            if not normalized:
                continue
            # Should be sources/<slug>
            if not normalized.startswith("sources/"):
                continue
            source_file = paths.wiki / (normalized + ".md")
            if not source_file.exists():
                issues.append(
                    LintIssue(
                        check=CheckId.STALE_SOURCE_REF,
                        severity=Severity.WARNING,
                        page=relpath,
                        message=f"Source reference '{normalized}' doesn't exist.",
                        suggestion=f"The source page was deleted or renamed. Remove or update the reference.",
                        fixable=False,
                        context={"target": normalized},
                    )
                )
    return issues


def check_noise_in_synthesis_sources(inv: PageInventory) -> list[LintIssue]:
    """Flag synthesis pages that cite `log.md` or `index.md` as sources.

    These are auto-maintained navigation files, not content. QMD sometimes
    returns them as search hits because their text matches the query, and
    they end up in synthesis page frontmatter. That's noise, not signal.
    """
    issues: list[LintIssue] = []
    for relpath, parsed in inv.pages.items():
        if not relpath.startswith("synthesis/"):
            continue
        for key in ("sources", "sources_consulted"):
            values = parsed.frontmatter.get(key, []) or []
            if not isinstance(values, list):
                continue
            for val in values:
                if not isinstance(val, str):
                    continue
                normalized = _normalize_link(val)
                base = normalized.rsplit("/", 1)[-1]
                if base in {"index", "log"} or normalized in {"index", "log"}:
                    issues.append(
                        LintIssue(
                            check=CheckId.NOISE_IN_SYNTHESIS,
                            severity=Severity.WARNING,
                            page=relpath,
                            message=f"Synthesis cites navigation file '{val}' as a source.",
                            suggestion="Run `wiki lint --fix` to remove noise from sources list.",
                            fixable=True,
                            context={
                                "location": "frontmatter",
                                "field": key,
                                "remove_value": val,
                            },
                        )
                    )
    return issues


# ---------------------------------------------------------------------------
# Deep check — LLM-powered contradiction detection (opt-in)
# ---------------------------------------------------------------------------


def check_contradictions_deep(
    inv: PageInventory,
    paths: cfg.WikiPaths,
    client,  # OllamaClient
    max_pairs: int = 10,
) -> list[LintIssue]:
    """Use Qwen3 to scan pairs of pages that share entities/concepts and
    flag potentially contradictory claims.

    This is slow — one LLM call per pair of pages. We limit to `max_pairs`
    to keep the runtime bounded.
    """
    from .llm import ChatMessage, LLMError
    from .prompts import CONTRADICTION_DETECTION_PROMPT

    issues: list[LintIssue] = []

    # 1. Identify pairs of pages that share outgoing wikilinks
    page_link_sets: dict[str, set[str]] = {}
    for relpath, targets in inv.outgoing_links.items():
        page_type = relpath.split("/", 1)[0] if "/" in relpath else ""
        if page_type in ("entities", "concepts"):
            page_link_sets[relpath] = set(t for t in targets if t)

    pairs: list[tuple[str, str, int]] = []
    paths_list = list(page_link_sets.keys())
    for i, a in enumerate(paths_list):
        for b in paths_list[i + 1 :]:
            overlap = len(page_link_sets[a] & page_link_sets[b])
            if overlap >= 1:
                pairs.append((a, b, overlap))

    # Sort by overlap desc — check most connected pairs first
    pairs.sort(key=lambda t: -t[2])
    pairs = pairs[:max_pairs]

    # 2. For each pair, ask Qwen3 to find contradictions
    for rel_a, rel_b, _overlap in pairs:
        page_a = inv.pages[rel_a]
        page_b = inv.pages[rel_b]

        prompt = CONTRADICTION_DETECTION_PROMPT.format(
            path_a=rel_a,
            path_b=rel_b,
            content_a=_trim_for_prompt(page_a.to_markdown()),
            content_b=_trim_for_prompt(page_b.to_markdown()),
        )
        messages = [
            ChatMessage(
                role="system",
                content="You are a careful fact-checker looking for contradictions.",
            ),
            ChatMessage(role="user", content=prompt),
        ]

        try:
            response = client.chat(messages, thinking=False, temperature=0.2)
        except LLMError:
            continue

        response = response.strip()
        if not response or response.upper().startswith("NONE"):
            continue

        issues.append(
            LintIssue(
                check=CheckId.CONTRADICTION,
                severity=Severity.WARNING,
                page=rel_a,
                message=f"Potential contradiction with {rel_b}",
                suggestion=response[:500],
                fixable=False,
                context={"other_page": rel_b},
            )
        )

    return issues


def _trim_for_prompt(text: str, max_chars: int = 3000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[... truncated ...]"


# ---------------------------------------------------------------------------
# Auto-fix
# ---------------------------------------------------------------------------


def _apply_fixes_to_page(parsed: page_writer.ParsedPage, fixes: list[LintIssue]) -> bool:
    """Apply all fixable issues to a single ParsedPage in place. Returns True
    if anything was changed.
    """
    changed = False
    body = parsed.body

    for issue in fixes:
        if not issue.fixable:
            continue
        ctx = issue.context

        if issue.check in (CheckId.MALFORMED_WIKILINK, CheckId.BROKEN_WIKILINK):
            old = ctx.get("old_target", "")
            new = ctx.get("new_target", "")
            location = ctx.get("location", "body")
            if not old or not new or old == new:
                continue

            if location == "body":
                # Replace [[old]] or [[old|alias]] with [[new]] (preserving alias)
                old_esc = re.escape(old)
                # Replace without alias
                body = re.sub(
                    rf"\[\[{old_esc}\]\]", f"[[{new}]]", body
                )
                # Replace with alias
                body = re.sub(
                    rf"\[\[{old_esc}\|([^\]]*)\]\]",
                    rf"[[{new}|\1]]",
                    body,
                )
                changed = True

            elif location == "frontmatter":
                field = ctx.get("field", "sources")
                values = parsed.frontmatter.get(field, []) or []
                if isinstance(values, list):
                    new_values = []
                    for v in values:
                        if isinstance(v, str) and v == old:
                            new_values.append(new)
                        else:
                            new_values.append(v)
                    parsed.frontmatter[field] = new_values
                    changed = True

        elif issue.check == CheckId.NOISE_IN_SYNTHESIS:
            field = ctx.get("field", "sources")
            remove_value = ctx.get("remove_value", "")
            values = parsed.frontmatter.get(field, []) or []
            if isinstance(values, list):
                new_values = [v for v in values if v != remove_value]
                if len(new_values) != len(values):
                    parsed.frontmatter[field] = new_values
                    changed = True

    if changed:
        parsed.body = body
    return changed


def apply_fixes(paths: cfg.WikiPaths, issues: list[LintIssue]) -> int:
    """Apply all fixable issues. Returns the count of pages modified.

    Runs in a loop: apply → re-lint → apply → re-lint. This is because some
    fixes cascade (e.g. normalizing a malformed wikilink can reveal a noise
    issue that the first pass couldn't see because the value was obscured).
    Bounded to 5 iterations to prevent pathological loops.
    """
    total_modified = 0
    current_issues = issues
    max_iterations = 5

    for _iteration in range(max_iterations):
        fixable = [i for i in current_issues if i.fixable]
        if not fixable:
            break

        by_page: dict[str, list[LintIssue]] = defaultdict(list)
        for issue in fixable:
            by_page[issue.page].append(issue)

        pages_modified_this_round = 0
        for relpath, page_issues in by_page.items():
            full_path = paths.wiki / relpath
            if not full_path.exists():
                continue
            try:
                content = full_path.read_text(encoding="utf-8")
            except OSError:
                continue
            parsed = page_writer.parse_page(content)
            if _apply_fixes_to_page(parsed, page_issues):
                full_path.write_text(parsed.to_markdown(), encoding="utf-8")
                pages_modified_this_round += 1

        if pages_modified_this_round == 0:
            break
        total_modified += pages_modified_this_round

        # Re-lint to find any new fixable issues revealed by this round's changes
        new_report = run_lint(paths, deep=False, client=None)
        current_issues = new_report.issues

    # Refresh the QMD search index so modified pages are reflected in queries.
    # Only runs if we actually modified anything. Non-fatal on failure.
    if total_modified > 0:
        try:
            from . import search
            search.update_index(paths, embed=True)
        except Exception:
            pass

    return total_modified


# ---------------------------------------------------------------------------
# Top-level: run_lint
# ---------------------------------------------------------------------------


def run_lint(
    paths: cfg.WikiPaths,
    *,
    deep: bool = False,
    client=None,  # OllamaClient, required if deep=True
) -> LintReport:
    """Run all fast checks, plus deep checks if requested.

    Returns a LintReport containing all issues, ordered by severity then page.
    """
    import time

    started = time.monotonic()
    report = LintReport()

    inv = _build_inventory(paths)
    report.pages_checked = len(inv.pages)

    # Fast checks
    fast_check_fns: list[tuple[str, Any]] = [
        ("broken_wikilinks", lambda: check_broken_wikilinks(inv)),
        ("orphan_pages", lambda: check_orphan_pages(inv)),
        ("frontmatter", lambda: check_frontmatter(inv)),
        ("malformed_wikilinks", lambda: check_malformed_wikilinks(inv, paths)),
        ("missing_concepts", lambda: check_missing_concepts(inv)),
        ("stale_source_refs", lambda: check_stale_source_refs(inv, paths)),
        ("noise_in_synthesis", lambda: check_noise_in_synthesis_sources(inv)),
    ]
    for name, fn in fast_check_fns:
        report.issues.extend(fn())
        report.fast_checks_run.append(name)

    # Deep check (LLM-powered)
    if deep and client is not None:
        report.deep_check_run = True
        report.issues.extend(check_contradictions_deep(inv, paths, client))

    # Sort: errors first, then warnings, then infos. Within each, by page.
    severity_order = {
        Severity.ERROR: 0,
        Severity.WARNING: 1,
        Severity.INFO: 2,
    }
    report.issues.sort(key=lambda i: (severity_order[i.severity], i.page, i.check.value))

    report.duration_seconds = time.monotonic() - started
    return report


# ---------------------------------------------------------------------------
# Markdown rendering of a report (for --save and display)
# ---------------------------------------------------------------------------


def render_report_markdown(report: LintReport, paths: cfg.WikiPaths) -> str:
    """Render a LintReport as a markdown document suitable for saving."""
    today = page_writer.today_iso()
    lines: list[str] = []
    lines.append("---")
    lines.append(f"title: Lint Report {today}")
    lines.append("type: synthesis")
    lines.append("tags: [lint, health-check]")
    lines.append(f"created: '{today}'")
    lines.append(f"updated: '{today}'")
    lines.append(f"health_score: {report.health_score}")
    lines.append("---")
    lines.append("")
    lines.append(f"# Lint Report — {today}")
    lines.append("")
    lines.append(f"**Health score:** {report.health_score}/100")
    lines.append(f"**Pages checked:** {report.pages_checked}")
    lines.append(f"**Duration:** {report.duration_seconds:.2f}s")
    lines.append("")
    lines.append(
        f"**Summary:** {len(report.errors)} errors · "
        f"{len(report.warnings)} warnings · {len(report.infos)} infos"
    )
    if report.auto_fixed:
        lines.append(f"**Auto-fixed:** {report.auto_fixed} issues")
    lines.append("")

    def _section(title: str, issues: list[LintIssue]) -> None:
        if not issues:
            return
        lines.append(f"## {title} ({len(issues)})")
        lines.append("")
        for issue in issues:
            lines.append(f"### `{issue.page}`")
            lines.append("")
            lines.append(f"- **Check:** {issue.check.value}")
            lines.append(f"- **Message:** {issue.message}")
            if issue.suggestion:
                lines.append(f"- **Suggestion:** {issue.suggestion}")
            if issue.fixable:
                lines.append("- **Auto-fixable:** yes (use `wiki lint --fix`)")
            lines.append("")

    _section("Errors", report.errors)
    _section("Warnings", report.warnings)
    _section("Info", report.infos)

    if not report.issues:
        lines.append("## Clean! 🎉")
        lines.append("")
        lines.append("No issues found. Your wiki is in good shape.")
        lines.append("")

    return "\n".join(lines)
