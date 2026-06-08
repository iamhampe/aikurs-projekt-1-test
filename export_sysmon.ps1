# Exportera Sysmon-loggar

Get-WinEvent `
-LogName "Microsoft-Windows-Sysmon/Operational" |

Select-Object `
TimeCreated,
Id,
Message |

Export-Csv `
"..\data\raw\sysmon_logs.csv" `
-NoTypeInformation