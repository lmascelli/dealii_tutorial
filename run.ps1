clear

if (!$args[0]) {
    Write-Output "Specify a folder"
}
else {
    $project = $args[0]
    switch ($args[1]) {
      'build' { $COMMAND_BUILD = $True }
      'run' { $COMMAND_RUN = $True }
      'clear' { $COMMAND_CLEAR = $True }
      'new' { $COMMAND_NEW = $True}
      default {
          Write-Output "Specify a command for ${args[0]}"
      }
    }

    if (Test-Path -PathType Container -Path "$project") {
        Push-Location $args[0]
            
        if ($COMMAND_CLEAR) {
            Remove-Item -Force -Path 'build' -Recurse -ErrorAction Ignore 
        }
        elseif ($COMMAND_BUILD -or $COMMAND_RUN) {
            Write-Output Get-Location
            if ((Test-Path -PathType Container -Path "build") -eq $False) {
                New-Item -ItemType Directory -Name 'build'
            }
            Push-Location build
        
            cmake .. -G"Ninja" -DCMAKE_EXPORT_COMPILE_COMMANDS=True
            cmake --build .
            Copy-Item -Force -Path compile_commands.json -Destination ../..
            Pop-Location
        }
       else {
            Write-Output 'Command unkwown'
        }
        if ($COMMAND_RUN) {
            Push-Location build
            ninja run
            Pop-Location
        }
        Pop-Location
    }
    else {
        if ($COMMAND_NEW) {
          New-Item -ItemType Directory -Name $project
          Copy-Item -Path "step1/CMakeLists.txt" -Destination $project
          New-Item -ItemType File -Path $project -Name ($project + ".cpp")
          echo ($project + "/.cache/") >> .gitignore
          echo ($project + "/build/") >> .gitignore
        } else {
         Write-Output "project ${args[0]} not found"
        }
    }
}
