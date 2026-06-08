' ===================================================================
' launch_run_monitor.vbs -- start the experiment-control monitor
' WINDOWLESS (no terminal), the way you'd normally run it on the
' terminal. Double-click it, or use the desktop shortcut that points
' here (pin that to the taskbar).
'
'   backend : pyctrl (run_monitor's default)
'   flags   : --bind-tailscale --enable-remote-controls
'   output  : log\monitor_log\run_monitor_<YYYYMMDD_HHMMSS>.log
'             (the pyctrl backend it spawns also writes its own
'              organized log\pyctrl_log\ files)
'
' The Tkinter control window + the Dash dashboard still appear -- only
' the console/terminal is gone. To change flags or watch logs live,
' launch from a terminal instead:
'   <env>\python.exe -m yb_analysis.scripts.run_monitor --bind-tailscale --enable-remote-controls
'
' Edit PROJ / PY below if the project or conda-env path ever moves.
' ===================================================================
Option Explicit

Dim PROJ, PY, logdir, ts, logfile, q, cmdline, fso, sh, d

PROJ = "c:\msys64\home\Ybtweezer-PC2\projects\experiment-control"
PY   = "C:\Users\Ybtweezer-PC2\anaconda3\envs\yb_analysis\python.exe"

logdir = PROJ & "\log\monitor_log"
Set fso = CreateObject("Scripting.FileSystemObject")
If Not fso.FolderExists(PROJ & "\log") Then fso.CreateFolder PROJ & "\log"
If Not fso.FolderExists(logdir) Then fso.CreateFolder logdir

d = Now
ts = Year(d) _
   & Right("0" & Month(d), 2) & Right("0" & Day(d), 2) & "_" _
   & Right("0" & Hour(d), 2) & Right("0" & Minute(d), 2) & Right("0" & Second(d), 2)
logfile = logdir & "\run_monitor_" & ts & ".log"

' Build:  cmd /c " "PY" -m yb_analysis.scripts.run_monitor <flags> >> "LOG" 2>&1 "
' (Chr(34) = a literal double-quote; the outer cmd /c "..." wrapper plus
'  quoted exe/log paths is the robust form that survives spaces in paths.)
q = Chr(34)
cmdline = "cmd /c " & q & " " & q & PY & q _
        & " -m yb_analysis.scripts.run_monitor --bind-tailscale --enable-remote-controls" _
        & " >> " & q & logfile & q & " 2>&1 " & q

Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = PROJ      ' so `python -m yb_analysis...` resolves the package
sh.Run cmdline, 0, False        ' window style 0 = hidden; don't wait
