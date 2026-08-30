$env:JAVA_HOME = 'C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot'
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
& 'C:\Users\David\.localbench\aider-arms.ps1' -Phase solve -Languages go,cpp,rust,javascript,java -Arms warm,claude-code -Limit 0 -RunTag '-v1' -EvalTimeout 900
