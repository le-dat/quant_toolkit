"""Kiểm tra điều kiện tiên quyết (preflight) cho Layer 2.6.

Kiểm tra cấu hình môi trường và các nguồn dữ liệu hồ dữ liệu nội bộ (ho_du_lieu).
Tuân thủ bất biến I1: Tuyệt đối không gọi REST/HTTP ngoài luồng.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Thêm thư mục chứa gói pseud vào sys.path
root = Path(__file__).resolve().parent
if str(root.parent) not in sys.path:
    sys.path.insert(0, str(root.parent))
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from rich.console import Console
from rich.table import Table

from pseud.config.accessor import get_env_config, reset_env_config


@dataclass(frozen=True)
class CheckResult:
    """Kết quả của một bài kiểm tra tiên quyết (preflight check)."""

    name: str
    status: str  # "ready", "error", "not_configured", "skipped"
    message: str
    impact: str  # ảnh hưởng nếu kiểm tra này thất bại
    critical: bool = False


def _check_llm_provider() -> CheckResult:
    """Xác minh cấu hình nhà cung cấp LLM từ biến môi trường."""
    from pseud.providers.llm import _ensure_dotenv, _sync_provider_env, provider_diagnostics

    _ensure_dotenv()
    reset_env_config()
    _cfg = get_env_config()
    provider = _cfg.llm.langchain_provider.strip()
    model = _cfg.llm.langchain_model_name.strip()

    if not provider:
        return CheckResult(
            name="LLM Provider",
            status="not_configured",
            message="LANGCHAIN_PROVIDER not set in .env",
            impact="agent cannot function",
            critical=True,
        )
    if not model:
        return CheckResult(
            name=f"LLM ({provider})",
            status="not_configured",
            message="LANGCHAIN_MODEL_NAME not set in .env",
            impact="agent cannot function",
            critical=True,
        )

    _sync_provider_env()
    diagnostics = provider_diagnostics()
    diag_hint = (
        f"base={diagnostics['base_url']} "
        f"timeout={diagnostics['timeout_seconds']}s "
        f"retries={diagnostics['max_retries']}"
    )

    return CheckResult(
        name=f"LLM ({provider})",
        status="ready",
        message=f"{model} configured | {diag_hint}",
        impact="",
    )


def _check_content_filter_threshold() -> CheckResult:
    """Báo cáo ngưỡng cảnh báo bộ lọc nội dung đã cấu hình."""
    threshold = get_env_config().agent_tuning.content_filter_warning_threshold
    return CheckResult(
        name="Content Filter Threshold",
        status="ready",
        message=f"{threshold:.0%} (set via CONTENT_FILTER_WARNING_THRESHOLD)",
        impact="",
    )


_STATUS_DISPLAY = {
    "ready": ("[green]OK[/green]", "green"),
    "error": ("[red]FAIL[/red]", "red"),
    "not_configured": ("[yellow]N/A[/yellow]", "yellow"),
    "skipped": ("[dim]SKIP[/dim]", "dim"),
}


def run_preflight(console: Optional[Console] = None) -> List[CheckResult]:
    """Chạy tất cả các kiểm tra tiên quyết và in kết quả."""
    if console is None:
        console = Console()

    checks = [
        _check_llm_provider,
        _check_content_filter_threshold,
    ]

    results: List[CheckResult] = []
    for check_fn in checks:
        results.append(check_fn())

    table = Table(show_header=False, show_edge=False, padding=(0, 1), expand=False)
    table.add_column(width=4)
    table.add_column(width=18)
    table.add_column()

    for r in results:
        icon, color = _STATUS_DISPLAY[r.status]
        detail = r.message
        if r.status in ("error", "not_configured") and r.impact:
            detail = f"{r.message} ({r.impact})"
        table.add_row(icon, f"[{color}]{r.name}[/{color}]", f"[{color}]{detail}[/{color}]")

    console.print()
    console.print("[bold]Preflight Check[/bold]")
    console.print(table)

    has_critical = any(r.critical and r.status != "ready" for r in results)
    if has_critical:
        console.print("\n[bold red]Critical check failed - agent cannot start without a working LLM provider.[/bold red]")
    else:
        ready_count = sum(1 for r in results if r.status == "ready")
        console.print(f"\n[dim]{ready_count}/{len(results)} services ready[/dim]")

    console.print()
    return results
