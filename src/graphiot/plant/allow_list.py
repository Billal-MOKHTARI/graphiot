"""Gateway allow-list loader (also baseline B1).

Loads ``plant/allow_list.yaml`` and answers "may task type T use tool A on tag
κ?". Tag patterns may be exact names or a trailing ``.*`` prefix glob.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

TOOLS = {
    "read_tag", "write_tag", "read_work_order",
    "read_stage_status", "load_recipe", "acknowledge_alarm",
}


class TaskTypePolicy(BaseModel):
    description: str = ""
    tools: list[str]
    write_tags: list[str] = Field(default_factory=list)
    read_tags: list[str] = Field(default_factory=list)

    def _match(self, patterns: list[str], tag: str) -> bool:
        return any(fnmatch.fnmatchcase(tag, p) for p in patterns)

    def may_use_tool(self, tool: str) -> bool:
        return tool in self.tools

    def may_write(self, tag: str) -> bool:
        return "write_tag" in self.tools and self._match(self.write_tags, tag)

    def may_read(self, tag: str) -> bool:
        return "read_tag" in self.tools and self._match(self.read_tags, tag)


class AllowList(BaseModel):
    meta: dict = Field(default_factory=dict)
    common_read_tools: list[str] = Field(default_factory=list)
    task_types: dict[str, TaskTypePolicy]

    def policy(self, task_type: str) -> TaskTypePolicy:
        if task_type not in self.task_types:
            raise KeyError(f"unknown task type {task_type!r}")
        return self.task_types[task_type]

    def decide(self, task_type: str, tool: str, tag: str | None) -> tuple[bool, str]:
        """Return (allowed, reason). The gateway logs the reason on every call."""
        if task_type not in self.task_types:
            return False, f"unknown task type {task_type!r}"
        pol = self.task_types[task_type]
        if not pol.may_use_tool(tool):
            return False, f"tool {tool!r} not allowed for task type {task_type!r}"
        if tool == "write_tag":
            if tag is None:
                return False, "write_tag without a tag"
            if not pol.may_write(tag):
                return False, f"write to {tag!r} not in allow-list for {task_type!r}"
        if tool == "read_tag":
            if tag is None:
                return False, "read_tag without a tag"
            if not pol.may_read(tag):
                return False, f"read of {tag!r} not in allow-list for {task_type!r}"
        return True, "ok"


def load_allow_list(path: str | Path, *, tag_names: set[str] | None = None) -> AllowList:
    data = yaml.safe_load(Path(path).read_text())
    al = AllowList.model_validate(data)
    for tname, pol in al.task_types.items():
        bad = set(pol.tools) - TOOLS
        if bad:
            raise ValueError(f"task type {tname!r}: unknown tools {sorted(bad)}")
        if tag_names is not None:
            for tag in [*pol.write_tags, *pol.read_tags]:
                if tag.endswith(".*"):
                    prefix = tag[:-2]
                    if not any(n.startswith(prefix) for n in tag_names):
                        raise ValueError(f"task type {tname!r}: glob {tag!r} matches no tag")
                elif tag not in tag_names:
                    raise ValueError(f"task type {tname!r}: unknown tag {tag!r}")
    return al
