(*
  PLC_SLIT -- slitting controller.
  Rate-limited line drive, rewind torque, phase SM, knife interlock
  (tension established + speed in band before the knives may engage).
  Physics (tension, edge burr, rewind alignment) is in the simulator.
  Scan = 100 ms.

  %MW0 line_speed_sp[m/min*10] %MW1 tension_sp[N] %MW2 rewind_torque_sp[Nm*10]
  %IW0 web_tension[N] %IW1 edge_burr[um] %IW2 rewind_align[mm*100] %IW3 line_speed_actual[m/min*10]
  %QW0 drive_cmd[m/min*10] %QW1 rewind_cmd[Nm*10] %QW2 tension_cmd[N] %QW3 knife_cmd[0/1]
  %QW10 phase %QW11 time_in_phase[s]
  %QX0.0 run_cmd %QX0.1 knife_engage_cmd %IX0.0 tension_ok %QX1.0 alarm %QX2.0 alarm_ack
*)
PROGRAM Main
  VAR
    line_speed_sp     AT %MW0 : INT;
    tension_sp        AT %MW1 : INT;
    rewind_torque_sp  AT %MW2 : INT;
    web_tension       AT %IW0 : INT;
    edge_burr         AT %IW1 : INT;
    rewind_align      AT %IW2 : INT;
    line_speed_act    AT %IW3 : INT;
    drive_cmd         AT %QW0 : INT;
    rewind_cmd        AT %QW1 : INT;
    tension_cmd       AT %QW2 : INT;
    knife_cmd         AT %QW3 : INT;
    phase_out         AT %QW10 : INT;
    tip_out           AT %QW11 : INT;
    run_cmd           AT %QX0.0 : BOOL;
    knife_engage_cmd  AT %QX0.1 : BOOL;
    tension_ok_in     AT %IX0.0 : BOOL;
    alarm_out         AT %QX1.0 : BOOL;
    alarm_ack         AT %QX2.0 : BOOL;
  END_VAR
  VAR
    phase       : INT := 0;
    tip_ticks   : DINT := 0;
    sp_speed    : REAL;
    act_speed   : REAL;
    cmd_speed   : REAL;
    step_speed  : REAL := 0.3;    (* m/min per scan *)
    tens        : REAL;
    speed_band  : BOOL;
    knife_ok    : BOOL;
    prev_speed  : INT := 0;
  END_VAR

  sp_speed  := INT_TO_REAL(line_speed_sp) / 10.0;
  act_speed := INT_TO_REAL(line_speed_act) / 10.0;
  tens      := INT_TO_REAL(web_tension);

  speed_band := (act_speed >= 3.0) AND (act_speed <= 40.0);
  knife_ok   := tension_ok_in AND (tens >= 25.0) AND speed_band;

  IF NOT run_cmd THEN
    phase := 0;
    cmd_speed := 0.0;
    rewind_cmd := 0;
    tension_cmd := 0;
    knife_cmd := 0;
  ELSE
    tension_cmd := tension_sp;
    (* knife interlock enforced here: the actuator only moves when it is safe *)
    IF knife_engage_cmd AND knife_ok THEN
      knife_cmd := 1;
    ELSE
      knife_cmd := 0;
    END_IF;
    cmd_speed := INT_TO_REAL(drive_cmd) / 10.0;
    IF (sp_speed - cmd_speed) > step_speed THEN
      cmd_speed := cmd_speed + step_speed;
    ELSIF (cmd_speed - sp_speed) > step_speed THEN
      cmd_speed := cmd_speed - step_speed;
    ELSE
      cmd_speed := sp_speed;
    END_IF;
    rewind_cmd := REAL_TO_INT(LIMIT(0.0, INT_TO_REAL(rewind_torque_sp) / 10.0, 60.0) * 10.0);
  END_IF;

  drive_cmd := REAL_TO_INT(LIMIT(0.0, cmd_speed, 45.0) * 10.0);

  CASE phase OF
    0:  IF run_cmd THEN phase := 1; tip_ticks := 0; END_IF;
    1:  (* ramp: accelerate and establish tension/alignment *)
        IF knife_ok AND knife_engage_cmd AND (ABS(act_speed - sp_speed) < 1.5) AND (tip_ticks > 40) THEN
          phase := 2; tip_ticks := 0;
        END_IF;
    2:  IF ABS(INT_TO_REAL(line_speed_sp - prev_speed)) > 50.0 THEN
          phase := 3; tip_ticks := 0;
        END_IF;
    3:  IF (ABS(act_speed - sp_speed) < 1.5) AND (tip_ticks > 30) THEN
          phase := 2; tip_ticks := 0;
        END_IF;
  END_CASE;
  IF NOT run_cmd THEN phase := 0; END_IF;

  (* alarm: knives engaged without the interlock, or tension lost while running *)
  IF (knife_engage_cmd AND NOT knife_ok) OR (run_cmd AND tension_ok_in AND (tens < 15.0)) THEN
    alarm_out := TRUE;
  ELSIF alarm_ack THEN
    alarm_out := FALSE;
  END_IF;

  tip_ticks := tip_ticks + 1;
  tip_out   := REAL_TO_INT(INT_TO_REAL(DINT_TO_INT(tip_ticks)) / 10.0);
  phase_out := phase;
  prev_speed := line_speed_sp;
END_PROGRAM

CONFIGURATION Config0
  RESOURCE Res0 ON PLC
    TASK task0(INTERVAL := T#100ms, PRIORITY := 0);
    PROGRAM instance0 WITH task0 : Main;
  END_RESOURCE
END_CONFIGURATION
