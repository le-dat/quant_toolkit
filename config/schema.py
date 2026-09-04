"""Schema cấu hình agent cấu trúc cho tích hợp MCP client trong Kairos v3."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class MCPServerConfig(BaseModel):
    """Cấu hình cho một MCP server đơn lẻ."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    type: str = Field(default="streamableHttp")
    url: str = Field(default="")
    command: str = Field(default="")
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    init_timeout: float = Field(default=60.0)
    enabled_tools: list[str] = Field(default_factory=lambda: ["*"])


class MCPServerConfigOverride(BaseModel):
    """Cấu hình ghi đè cho một MCP server."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    type: Optional[str] = None
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[list[str]] = None
    env: Optional[dict[str, str]] = None
    init_timeout: Optional[float] = None
    enabled_tools: Optional[list[str]] = None


class AgentConfig(BaseModel):
    """Cấu hình tổng thể cho Agent trong Kairos v3."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    mcp_servers: dict[str, MCPServerConfig] = Field(
        default_factory=dict, alias="mcpServers"
    )


class AgentConfigOverride(BaseModel):
    """Cấu hình ghi đè tổng thể cho Agent."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    mcp_servers: Optional[dict[str, MCPServerConfigOverride]] = Field(
        default=None, alias="mcpServers"
    )
