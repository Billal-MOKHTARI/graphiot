(*
  PLC_MIX -- slurry mixing controller.
  Thin control layer: rate-limited actuator tracking, phase state machine,
  interlocks -> alarm. Physics (viscosity, temperature, air) lives in the
  process simulator; sensors arrive on %IW from the field I/O. Scan = 100 ms.

  Located I/O matches plant/tag_map.csv (PLC_MIX):
    %MW0 speed_sp[rpm]  %MW1 jacket_temp_sp[degC*10]  %MW2 dose_rate_sp[%/s*100]  %MW3 vacuum_sp[kPa*10]
    %IW0 viscosity      %IW1 solids[%*10]  %IW2 temp[degC*10]  %IW3 air[%*10]  %IW4 speed_actual[rpm]
    %QW0 motor_cmd[rpm] %QW1 heater_cmd[%*10] %QW2 dose_cmd[%/s*100] %QW3 vacuum_cmd[kPa*10]
    %QW10 phase  %QW11 time_in_phase[s]
    %QX0.0 run_cmd  %IX0.0 vessel_closed  %QX1.0 alarm  %QX2.0 alarm_ack
*)
PROGRAM Main
  VAR
    speed_sp        AT %MW0 : INT;
    jacket_temp_sp  AT %MW1 : INT;
    dose_rate_sp    AT %MW2 : INT;
    vacuum_sp       AT %MW3 : INT;
    viscosity       AT %IW0 : INT;
    solids          AT %IW1 : INT;
    temp_raw        AT %IW2 : INT;
    air_entrain     AT %IW3 : INT;
    speed_act       AT %IW4 : INT;
    motor_cmd       AT %QW0 : INT;
    heater_cmd      AT %QW1 : INT;
    dose_cmd        AT %QW2 : INT;
    vacuum_cmd      AT %QW3 : INT;
    phase_out       AT %QW10 : INT;
    tip_out         AT %QW11 : INT;
    run_cmd         AT %QX0.0 : BOOL;
    vessel_closed   AT %IX0.0 : BOOL;
    alarm_out       AT %QX1.0 : BOOL;
    alarm_ack       AT %QX2.0 : BOOL;
  END_VAR
  VAR
    phase        : INT := 0;
    tip_ticks    : DINT := 0;
    temp         : REAL;
    sp_rpm       : REAL;
    act_rpm      : REAL;
    cmd_rpm      : REAL;
    step_rpm     : REAL := 120.0;   (* max motor change per scan: ~1200 rpm/s *)
    interlock_ok : BOOL;
    high_shear   : BOOL;
    prev_sp      : INT := 0;
    heat         : REAL;
  END_VAR

  sp_rpm  := INT_TO_REAL(speed_sp);
  act_rpm := INT_TO_REAL(speed_act);
  temp    := INT_TO_REAL(temp_raw) / 10.0;

  (* High-shear dispersing is interlocked for SAFETY only: the vessel must be
     closed. Product quality (de-aeration under vacuum) is the operator's job. *)
  high_shear   := sp_rpm > 1200.0;
  interlock_ok := vessel_closed;

  IF NOT run_cmd THEN
    phase    := 0;
    cmd_rpm  := 0.0;
    heater_cmd := 0;
    dose_cmd := 0;
    vacuum_cmd := 1010;     (* vent to atmosphere: 101.0 kPa abs *)
  ELSE
    (* feeder and vacuum follow their setpoints while the batch runs *)
    dose_cmd   := dose_rate_sp;
    vacuum_cmd := vacuum_sp;
    (* Rate-limited speed command toward the setpoint, clamped by the interlock. *)
    cmd_rpm := INT_TO_REAL(motor_cmd);
    IF (sp_rpm - cmd_rpm) > step_rpm THEN
      cmd_rpm := cmd_rpm + step_rpm;
    ELSIF (cmd_rpm - sp_rpm) > step_rpm THEN
      cmd_rpm := cmd_rpm - step_rpm;
    ELSE
      cmd_rpm := sp_rpm;
    END_IF;
    IF high_shear AND NOT interlock_ok THEN
      cmd_rpm := 1200.0;    (* refuse to exceed safe speed without the interlock *)
    END_IF;
    (* Jacket heater: proportional toward the temperature setpoint. *)
    heat := 8.0 * (INT_TO_REAL(jacket_temp_sp) / 10.0 - temp);
    heater_cmd := REAL_TO_INT(LIMIT(0.0, heat, 100.0) * 10.0);
  END_IF;

  motor_cmd := REAL_TO_INT(LIMIT(0.0, cmd_rpm, 3500.0));

  (* ---- phase state machine ---- *)
  CASE phase OF
    0:  (* idle *)
        IF run_cmd THEN phase := 1; tip_ticks := 0; END_IF;
    1:  (* ramp: spin-up and solids dosing *)
        IF (ABS(act_rpm - sp_rpm) < 100.0) AND (dose_rate_sp = 0) AND (tip_ticks > 30) THEN
          phase := 2; tip_ticks := 0;
        END_IF;
    2:  (* steady *)
        IF ABS(INT_TO_REAL(speed_sp - prev_sp)) > 300.0 THEN
          phase := 3; tip_ticks := 0;
        END_IF;
    3:  (* transition: re-settle after a setpoint change *)
        IF (ABS(act_rpm - sp_rpm) < 100.0) AND (tip_ticks > 20) THEN
          phase := 2; tip_ticks := 0;
        END_IF;
  END_CASE;
  IF NOT run_cmd THEN phase := 0; END_IF;

  (* ---- alarm: interlock violation or over-temperature ---- *)
  IF (high_shear AND NOT interlock_ok) OR (temp > 48.0) THEN
    alarm_out := TRUE;
  ELSIF alarm_ack THEN
    alarm_out := FALSE;
  END_IF;

  tip_ticks := tip_ticks + 1;
  tip_out   := REAL_TO_INT(INT_TO_REAL(DINT_TO_INT(tip_ticks)) / 10.0);
  phase_out := phase;
  prev_sp   := speed_sp;
END_PROGRAM

CONFIGURATION Config0
  RESOURCE Res0 ON PLC
    TASK task0(INTERVAL := T#100ms, PRIORITY := 0);
    PROGRAM instance0 WITH task0 : Main;
  END_RESOURCE
END_CONFIGURATION
