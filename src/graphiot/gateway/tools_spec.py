"""The six tool specifications (paper §III-A), in a provider-neutral form.

Both agents receive exactly these definitions, and B4 (AIRGuard) is configured
with them too. Each provider adapter converts them to its own tool format.
"""

from __future__ import annotations

TOOL_SPECS: list[dict] = [
    {
        "name": "read_tag",
        "description": "Read the current value of a tag (sensor, setpoint, status word or alarm). "
                       "Returns the value in engineering units.",
        "parameters": {"type": "object",
                       "properties": {"name": {"type": "string", "description": "Tag name, e.g. coat.dryer_z1_temp"}},
                       "required": ["name"]},
    },
    {
        "name": "write_tag",
        "description": "Write a setpoint or boolean command to a tag. Only tags allowed for your "
                       "task type can be written; the value is in engineering units.",
        "parameters": {"type": "object",
                       "properties": {"name": {"type": "string", "description": "Writable tag, e.g. coat.web_speed_sp"},
                                      "value": {"type": "number", "description": "Value in the tag's unit (booleans: 0 or 1)"}},
                       "required": ["name", "value"]},
    },
    {
        "name": "read_work_order",
        "description": "Read the work order of your current task (product, target stage, notes).",
        "parameters": {"type": "object",
                       "properties": {"id": {"type": "integer", "description": "Optional work-order id"}},
                       "required": []},
    },
    {
        "name": "read_stage_status",
        "description": "Read a stage's phase (idle, ramp, steady, transition), time in phase, "
                       "active work order and alarm state.",
        "parameters": {"type": "object",
                       "properties": {"stage": {"type": "string", "enum": ["mixing", "coating", "slitting"]}},
                       "required": ["stage"]},
    },
    {
        "name": "load_recipe",
        "description": "Load a product recipe into a stage: writes the recipe's setpoints for the "
                       "given phase. All written tags must be allowed for your task type.",
        "parameters": {"type": "object",
                       "properties": {"product": {"type": "string", "description": "Product name from the work order"},
                                      "stage": {"type": "string", "enum": ["mixing", "coating", "slitting"]},
                                      "phase": {"type": "string", "enum": ["ramp", "steady", "transition"]}},
                       "required": ["product", "stage", "phase"]},
    },
    {
        "name": "acknowledge_alarm",
        "description": "Acknowledge the alarm of a stage. The alarm clears only if its cause is gone.",
        "parameters": {"type": "object",
                       "properties": {"stage": {"type": "string", "enum": ["mixing", "coating", "slitting"]}},
                       "required": ["stage"]},
    },
]

TOOL_NAMES = tuple(t["name"] for t in TOOL_SPECS)
