; Ochroma Engine Installer
; NSIS Script

!define PRODUCT_NAME "Ochroma Engine"
!define PRODUCT_VERSION "0.1.0"
!define PRODUCT_PUBLISHER "Ochroma"
!define PRODUCT_WEB_SITE "https://github.com/supergrahn/ochroma"

Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "../dist/OchromaEngine-Setup-${PRODUCT_VERSION}.exe"
InstallDir "$PROGRAMFILES64\Ochroma Engine"
InstallDirRegKey HKLM "Software\Ochroma" "InstallDir"
RequestExecutionLevel admin

; Modern UI
!include "MUI2.nsh"

; Interface settings
!define MUI_ABORTWARNING
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"
!define MUI_WELCOMEFINISHPAGE_BITMAP "${NSISDIR}\Contrib\Graphics\Wizard\win.bmp"

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "../LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; Uninstaller pages
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; Language
!insertmacro MUI_LANGUAGE "English"

; Installation
Section "Ochroma Engine" SecEngine
    SetOutPath "$INSTDIR"

    ; Engine binary
    File "../dist/ochroma-windows/ochroma.exe"

    ; Example games
    SetOutPath "$INSTDIR\examples"
    File "../dist/ochroma-windows/walking_sim.exe"
    File "../dist/ochroma-windows/platformer.exe"

    ; Documentation
    SetOutPath "$INSTDIR\docs"
    File "../dist/ochroma-windows/README.md"
    File "../dist/ochroma-windows/getting_started.md"
    File "../dist/ochroma-windows/CONTROLS.txt"

    ; Assets directory
    SetOutPath "$INSTDIR\assets"
    File "../dist/ochroma-windows/assets/README.md"

    ; Create start menu shortcuts
    CreateDirectory "$SMPROGRAMS\Ochroma Engine"
    CreateShortcut "$SMPROGRAMS\Ochroma Engine\Ochroma Engine.lnk" "$INSTDIR\ochroma.exe"
    CreateShortcut "$SMPROGRAMS\Ochroma Engine\Walking Simulator.lnk" "$INSTDIR\examples\walking_sim.exe"
    CreateShortcut "$SMPROGRAMS\Ochroma Engine\Platformer.lnk" "$INSTDIR\examples\platformer.exe"
    CreateShortcut "$SMPROGRAMS\Ochroma Engine\Uninstall.lnk" "$INSTDIR\uninstall.exe"

    ; Create desktop shortcut
    CreateShortcut "$DESKTOP\Ochroma Engine.lnk" "$INSTDIR\ochroma.exe"

    ; Write uninstaller
    WriteUninstaller "$INSTDIR\uninstall.exe"

    ; Write registry keys
    WriteRegStr HKLM "Software\Ochroma" "InstallDir" "$INSTDIR"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Ochroma" "DisplayName" "${PRODUCT_NAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Ochroma" "UninstallString" "$INSTDIR\uninstall.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Ochroma" "DisplayVersion" "${PRODUCT_VERSION}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Ochroma" "Publisher" "${PRODUCT_PUBLISHER}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Ochroma" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Ochroma" "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Ochroma" "NoRepair" 1

    ; Register .ochroma_map file association
    WriteRegStr HKCR ".ochroma_map" "" "OchromaMap"
    WriteRegStr HKCR "OchromaMap" "" "Ochroma Map File"
    WriteRegStr HKCR "OchromaMap\shell\open\command" "" '"$INSTDIR\ochroma.exe" "%1"'

    ; Register .ply file association (open with Ochroma)
    WriteRegStr HKCR ".ply\OpenWithProgids" "OchromaPLY" ""
    WriteRegStr HKCR "OchromaPLY" "" "Gaussian Splat File"
    WriteRegStr HKCR "OchromaPLY\shell\open\command" "" '"$INSTDIR\ochroma.exe" "%1"'

SectionEnd

; Uninstaller
Section "Uninstall"
    ; Remove files
    Delete "$INSTDIR\ochroma.exe"
    Delete "$INSTDIR\examples\walking_sim.exe"
    Delete "$INSTDIR\examples\platformer.exe"
    Delete "$INSTDIR\docs\README.md"
    Delete "$INSTDIR\docs\getting_started.md"
    Delete "$INSTDIR\docs\CONTROLS.txt"
    Delete "$INSTDIR\assets\README.md"
    Delete "$INSTDIR\uninstall.exe"

    ; Remove directories
    RMDir "$INSTDIR\examples"
    RMDir "$INSTDIR\docs"
    RMDir "$INSTDIR\assets"
    RMDir "$INSTDIR"

    ; Remove shortcuts
    Delete "$SMPROGRAMS\Ochroma Engine\Ochroma Engine.lnk"
    Delete "$SMPROGRAMS\Ochroma Engine\Walking Simulator.lnk"
    Delete "$SMPROGRAMS\Ochroma Engine\Platformer.lnk"
    Delete "$SMPROGRAMS\Ochroma Engine\Uninstall.lnk"
    RMDir "$SMPROGRAMS\Ochroma Engine"
    Delete "$DESKTOP\Ochroma Engine.lnk"

    ; Remove registry keys
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Ochroma"
    DeleteRegKey HKLM "Software\Ochroma"
    DeleteRegKey HKCR ".ochroma_map"
    DeleteRegKey HKCR "OchromaMap"
    DeleteRegKey HKCR "OchromaPLY"
SectionEnd
