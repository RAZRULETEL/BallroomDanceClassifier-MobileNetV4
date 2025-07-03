# Counts files in each directory (not recursively) and prints with folder name
$directories = Get-ChildItem -Directory

# Loop through each directory and count files
$results = $directories | ForEach-Object {
    $dir = $_
    $fileCount = (Get-ChildItem -File -Path $dir.FullName | Measure-Object).Count
    [PSCustomObject]@{
        Directory = $dir.Name
        FileCount = $fileCount
    }
}

# Sort by file count (descending)
$sortedResults = $results | Sort-Object FileCount -Descending

# Display the sorted results in a table
$sortedResults | Format-Table -AutoSize