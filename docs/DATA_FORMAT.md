# Data format of a run folder

`runs/<name>_<utc>/` — Parquet (Arrow) tables plus the raw JSONL they are built from.
**Monitor-visible** tables are the only inputs of the graph builder and every detector
(through `graphiot.data.monitor_view`); everything else is ground truth or oracle output.

| File | Monitor | Content |
|---|---|---|
| `config.yaml`, `meta.json` | – | exact config; hash, git commit, source-tree hash, seeds, versions, backend, device |
| `testbed/docker-compose.yml`, `testbed/bundles/*.zip`, `testbed/up_report.json` | – | rendered topology, deployed PLC programs (MD5), deploy/health report |
| `capture/gateway_log.parquet` | ✅ | one row per tool call: `ts, ts_response, call_id, task_id, agent_id, tool, tag, stage, decision, error, n_modbus, fcs` |
| `capture/commands.parquet` | ✅ | one row per tag-write attempt: `ts, call_id, task_id, agent_id, tool, tag, value, raw, function_code, decision` (`forwarded`/`denied`) |
| `capture/tasks.parquet` | ✅ | `ts, task_id, task_type, target_stage, requested_tags, agent_id, work_order_id` — no values |
| `capture/tags.parquet` | ✅ | 1 Hz HMI polls, long format: `ts, plc, tag, value` (engineering units) |
| `capture/modbus.parquet` | ✅ | Modbus/TCP transactions from the pcaps: `ts, ts_response, src/dst ip+port, transaction_id, unit_id, function_code, address, count, write_values, read_values, exception, request_bytes, response_bytes` |
| `capture/flows.parquet` | ✅ | NetFlow-style records (5 s interim export): `flow_start, flow_end, end_reason, src/dst ip+port, protocol, IN/OUT_BYTES, IN/OUT_PKTS, FLOW_DURATION_MILLISECONDS, TCP flags, TTL and length extremes, throughput, size histogram, TCP windows, retransmissions` |
| `capture/pcap/cell-*.pcap` | (raw) | tcpdump of the cell bridge, 60 s rotation |
| `capture/gateway_full.jsonl` | ❌ | audit log with full results (work orders etc.) |
| `groundtruth/tasks_full.parquet` | ❌ | tasks with text, kind, planned phase, override flag |
| `groundtruth/agent_periods.parquet` | ❌ | `start_ts, end_ts, actor (operator/red_team), task_id, episode_id, ended` |
| `groundtruth/episodes.parquet` | ❌ | red-team episodes: technique, model, models actually served, outcome |
| `groundtruth/plant_state.parquet` | ❌ | live simulator state at 1 Hz (noise-free) + actuators — twin outputs |
| `groundtruth/llm_log.jsonl`, `llm_turns.parquet` | ❌ | every prompt, model turn, tool exchange |
| `oracle/episode_labels.parquet` | ❌ | `harmful` (proximate cause, D11), `harmful_literal` (paper window), caused outputs |
| `oracle/command_labels.parquet` | ❌ | commands + `harmful` (leave-one-out) + actor |
| `oracle/violations.parquet`, `fidelity.json`, `summary.json` | ❌ | violations with attribution; twin vs logged tags; counts and shares |
| `results/pilot.json`, `pilot_table.{csv,tex}`, `pilot_commands.parquet` | ❌ | go/no-go test and the pilot table |
