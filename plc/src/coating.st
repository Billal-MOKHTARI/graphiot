(*
  PLC_COAT -- coating + three-zone drying controller.
  Rate-limited web drive, pump tracking, three zone heaters, phase SM,
  interlocks (web present + tension band to run the pump; dryer over-temp trip).
  Physics (temperatures, tension, thickness, loading, residual solvent) is in
  the simulator; sensors arrive on %IW. Scan = 100 ms.

  %MW0 web_speed_sp[m/min*10] %MW1 pump_flow_sp[mL/min] %MW2 gap_sp[um]
  %MW3..5 dryer_z1..3_temp_sp[degC*10] %MW6 airflow_sp[m/s*10] %MW7 tension_sp[N]
  %IW0..2 dryer_z1..3_temp[degC*10] %IW3 web_tension[N] %IW4 wet_thickness[um]
  %IW5 dry_loading[mg/cm2*100] %IW6 residual_solvent[%*10] %IW7 web_speed_actual[m/min*10]
  %QW0 drive_cmd[m/min*10] %QW1 pump_cmd[mL/min] %QW2..4 z1..3_heat_cmd[%*10]
  %QW5 fan_cmd[m/s*10] %QW6 tension_cmd[N] %QW7 gap_cmd[um]
  %QW10 phase %QW11 time_in_phase[s]
  %QX0.0 run_cmd %IX0.0 web_present %QX1.0 alarm %QX2.0 alarm_ack
*)
PROGRAM Main
  VAR
    web_speed_sp    AT %MW0 : INT;
    pump_flow_sp    AT %MW1 : INT;
    gap_sp          AT %MW2 : INT;
    z1_sp           AT %MW3 : INT;
    z2_sp           AT %MW4 : INT;
    z3_sp           AT %MW5 : INT;
    airflow_sp      AT %MW6 : INT;
    tension_sp      AT %MW7 : INT;
    z1_temp         AT %IW0 : INT;
    z2_temp         AT %IW1 : INT;
    z3_temp         AT %IW2 : INT;
    web_tension     AT %IW3 : INT;
    wet_thickness   AT %IW4 : INT;
    dry_loading     AT %IW5 : INT;
    residual_solv   AT %IW6 : INT;
    web_speed_act   AT %IW7 : INT;
    drive_cmd       AT %QW0 : INT;
    pump_cmd        AT %QW1 : INT;
    z1_heat_cmd     AT %QW2 : INT;
    z2_heat_cmd     AT %QW3 : INT;
    z3_heat_cmd     AT %QW4 : INT;
    fan_cmd         AT %QW5 : INT;
    tension_cmd     AT %QW6 : INT;
    gap_cmd         AT %QW7 : INT;
    phase_out       AT %QW10 : INT;
    tip_out         AT %QW11 : INT;
    run_cmd         AT %QX0.0 : BOOL;
    web_present     AT %IX0.0 : BOOL;
    alarm_out       AT %QX1.0 : BOOL;
    alarm_ack       AT %QX2.0 : BOOL;
  END_VAR
  VAR
    phase      : INT := 0;
    tip_ticks  : DINT := 0;
    sp_speed   : REAL;
    act_speed  : REAL;
    cmd_speed  : REAL;
    step_speed : REAL := 0.2;   (* m/min per scan -> ~2 m/min/s web accel *)
    tens       : REAL;
    tension_ok : BOOL;
    zmax       : REAL;
    overtemp   : BOOL;
    prev_speed : INT := 0;
    pump_ok    : BOOL;
  END_VAR

  sp_speed  := INT_TO_REAL(web_speed_sp) / 10.0;
  act_speed := INT_TO_REAL(web_speed_act) / 10.0;
  tens      := INT_TO_REAL(web_tension);

  tension_ok := (tens >= 30.0) AND (tens <= 320.0);
  pump_ok    := web_present AND (act_speed > 0.3) AND tension_ok;

  (* dryer over-temperature trip across any zone *)
  zmax := INT_TO_REAL(z1_temp) / 10.0;
  IF (INT_TO_REAL(z2_temp) / 10.0) > zmax THEN zmax := INT_TO_REAL(z2_temp) / 10.0; END_IF;
  IF (INT_TO_REAL(z3_temp) / 10.0) > zmax THEN zmax := INT_TO_REAL(z3_temp) / 10.0; END_IF;
  overtemp := zmax > 172.0;

  gap_cmd := gap_sp;          (* die gap actuator follows its setpoint *)

  IF NOT run_cmd THEN
    phase := 0;
    cmd_speed := 0.0;
    pump_cmd := 0;
    z1_heat_cmd := 0; z2_heat_cmd := 0; z3_heat_cmd := 0;
    fan_cmd := 0;
    tension_cmd := 0;
  ELSE
    fan_cmd     := airflow_sp;
    tension_cmd := tension_sp;
    (* rate-limited web speed command *)
    cmd_speed := INT_TO_REAL(drive_cmd) / 10.0;
    IF (sp_speed - cmd_speed) > step_speed THEN
      cmd_speed := cmd_speed + step_speed;
    ELSIF (cmd_speed - sp_speed) > step_speed THEN
      cmd_speed := cmd_speed - step_speed;
    ELSE
      cmd_speed := sp_speed;
    END_IF;
    (* pump only when it is safe to coat; otherwise hold at zero *)
    IF pump_ok THEN
      pump_cmd := REAL_TO_INT(LIMIT(0.0, INT_TO_REAL(pump_flow_sp), 450.0));
    ELSE
      pump_cmd := 0;
    END_IF;
    (* zone heaters: proportional toward each zone setpoint, tripped on overtemp *)
    IF overtemp THEN
      z1_heat_cmd := 0; z2_heat_cmd := 0; z3_heat_cmd := 0;
    ELSE
      z1_heat_cmd := REAL_TO_INT(LIMIT(0.0, 6.0 * (INT_TO_REAL(z1_sp) / 10.0 - INT_TO_REAL(z1_temp) / 10.0), 100.0) * 10.0);
      z2_heat_cmd := REAL_TO_INT(LIMIT(0.0, 6.0 * (INT_TO_REAL(z2_sp) / 10.0 - INT_TO_REAL(z2_temp) / 10.0), 100.0) * 10.0);
      z3_heat_cmd := REAL_TO_INT(LIMIT(0.0, 6.0 * (INT_TO_REAL(z3_sp) / 10.0 - INT_TO_REAL(z3_temp) / 10.0), 100.0) * 10.0);
    END_IF;
  END_IF;

  drive_cmd := REAL_TO_INT(LIMIT(0.0, cmd_speed, 25.0) * 10.0);

  CASE phase OF
    0:  IF run_cmd THEN phase := 1; tip_ticks := 0; END_IF;
    1:  (* ramp: dryer warm-up + threading at low speed *)
        IF (zmax > 100.0) AND (ABS(act_speed - sp_speed) < 1.0) AND (sp_speed > 4.0) AND (tip_ticks > 50) THEN
          phase := 2; tip_ticks := 0;
        END_IF;
    2:  IF ABS(INT_TO_REAL(web_speed_sp - prev_speed)) > 30.0 THEN
          phase := 3; tip_ticks := 0;
        END_IF;
    3:  IF (ABS(act_speed - sp_speed) < 1.0) AND (tip_ticks > 30) THEN
          phase := 2; tip_ticks := 0;
        END_IF;
  END_CASE;
  IF NOT run_cmd THEN phase := 0; END_IF;

  IF overtemp OR (run_cmd AND web_present AND NOT tension_ok) THEN
    alarm_out := TRUE;
  ELSIF alarm_ack THEN
    alarm_out := FALSE;
  END_IF;

  tip_ticks := tip_ticks + 1;
  tip_out   := REAL_TO_INT(INT_TO_REAL(DINT_TO_INT(tip_ticks)) / 10.0);
  phase_out := phase;
  prev_speed := web_speed_sp;
END_PROGRAM

CONFIGURATION Config0
  RESOURCE Res0 ON PLC
    TASK task0(INTERVAL := T#100ms, PRIORITY := 0);
    PROGRAM instance0 WITH task0 : Main;
  END_RESOURCE
END_CONFIGURATION
