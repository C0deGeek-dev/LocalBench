Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint e);'
while($true){ [void][W.P]::SetThreadExecutionState([uint32]2147483649); Start-Sleep 240 }
