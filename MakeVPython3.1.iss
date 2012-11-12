; Create installer for VPython using Inno Setup Compiler (www.innosetup.com)
; Assumes Python and numpy already installed.

; Make sure version numbers are correct for VPython and numpy.
; Also, make sure that vidle\config-main.def has the correct Python version number for documentation

[Setup]
AppName=VPython for Python 3.1
AppVerName=VPython 5.72
AppPublisherURL=http://vpython.org
DefaultDirName={code:MyConst}

SourceDir=C:\Python31
DisableProgramGroupPage=yes
DirExistsWarning=no
DisableStartupPrompt=yes
OutputBaseFilename=VPython-Win-Py3.1-5.72
OutputDir=c:\workspace

[Files]
; The following overwrite is no longer performed, now that VIDLE is an option:
; Make sure that config-main.def has
; editor-on-startup and autosave set to 1
; and help set for Visual before building package.
;Source: "Lib\idlelib\config-main.def"; DestDir: "{app}\Lib\idlelib\"; Flags: uninsneveruninstall

Source: "Lib\site-packages\vis\cvisual.pyd"; DestDir: "{app}\Lib\site-packages\vis\"; Components: Visual
Source: "Lib\site-packages\vis\*.py"; DestDir: "{app}\Lib\site-packages\vis\"; Components: Visual
Source: "Lib\site-packages\vis\*.tga"; DestDir: "{app}\Lib\site-packages\vis\"; Components: Visual

Source: "Lib\site-packages\visual\*.py"; DestDir: "{app}\Lib\site-packages\visual\"; Components: Visual

; Execute compilevisual.py from the CVS files to compile the .pyc files:
Source: "Lib\site-packages\vis\*.pyc"; DestDir: "{app}\Lib\site-packages\vis\"; Components: Visual
Source: "Lib\site-packages\visual\*.pyc"; DestDir: "{app}\Lib\site-packages\visual\"; Components: Visual

Source: "c:\workspace\vpython-core2\license.txt"; DestDir: "{app}\Lib\site-packages\visual\"; Components: Visual

; Need to have installed numpy, FontTools, ttfquery, and Polygon before building Visual, so components available in site-packages:
Source: "Lib\site-packages\numpy*egg-info"; DestDir: "{app}\Lib\site-packages\"; Components: numpy
Source: "Lib\site-packages\numpy\*"; DestDir: "{app}\Lib\site-packages\numpy\"; Components: numpy; Flags: recursesubdirs

Source: "Lib\site-packages\FontTools.pth"; DestDir: "{app}\Lib\site-packages\"; Components: FontTools
Source: "Lib\site-packages\FontTools\*"; DestDir: "{app}\Lib\site-packages\FontTools\"; Components: FontTools; Flags: recursesubdirs

Source: "Lib\site-packages\TTFQuery*egg-info"; DestDir: "{app}\Lib\site-packages\"; Components: ttfquery
Source: "Lib\site-packages\ttfquery\*"; DestDir: "{app}\Lib\site-packages\ttfquery\"; Components: ttfquery; Flags: recursesubdirs

Source: "Lib\site-packages\Polygon*egg-info"; DestDir: "{app}\Lib\site-packages\"; Components: Polygon
Source: "Lib\site-packages\Polygon\*"; DestDir: "{app}\Lib\site-packages\Polygon\"; Components: Polygon; Flags: recursesubdirs

Source: "Lib\site-packages\vidle\*.py"; DestDir: "{app}\Lib\site-packages\vidle\"; Components: VIDLE; Flags: recursesubdirs
Source: "Lib\site-packages\vidle\*.pyc"; DestDir: "{app}\Lib\site-packages\vidle\"; Components: VIDLE; Flags: recursesubdirs
Source: "Lib\site-packages\vidle\*.pyw"; DestDir: "{app}\Lib\site-packages\vidle\"; Components: VIDLE; Flags: recursesubdirs
Source: "Lib\site-packages\vidle\*.txt"; DestDir: "{app}\Lib\site-packages\vidle\"; Components: VIDLE; Flags: recursesubdirs
Source: "Lib\site-packages\vidle\*.py"; DestDir: "{app}\Lib\site-packages\vidle\"; Components: VIDLE; Flags: recursesubdirs
Source: "Lib\site-packages\vidle\*.bat"; DestDir: "{app}\Lib\site-packages\vidle\"; Components: VIDLE; Flags: recursesubdirs
Source: "Lib\site-packages\vidle\*.def"; DestDir: "{app}\Lib\site-packages\vidle\"; Components: VIDLE; Flags: recursesubdirs
Source: "Lib\site-packages\vidle\*.gif"; DestDir: "{app}\Lib\site-packages\vidle\"; Components: VIDLE; Flags: recursesubdirs
Source: "Lib\site-packages\vidle\*.icns"; DestDir: "{app}\Lib\site-packages\vidle\"; Components: VIDLE; Flags: recursesubdirs

Source: "Lib\site-packages\visual\examples\*.py"; DestDir: "{app}\Lib\site-packages\visual\examples\"; Components: Examples
Source: "Lib\site-packages\visual\examples\*.tga"; DestDir: "{app}\Lib\site-packages\visual\examples\"; Components: Examples

Source: "c:\workspace\vpython-core2\docs\index.html"; DestDir: "{app}\Lib\site-packages\visual\docs\"; Components: Documentation
Source: "c:\workspace\vpython-core2\docs\visual\*.html"; DestDir: "{app}\Lib\site-packages\visual\docs\visual\"; Components: Documentation
Source: "c:\workspace\vpython-core2\docs\visual\*.txt"; DestDir: "{app}\Lib\site-packages\visual\docs\visual\"; Components: Documentation
Source: "c:\workspace\vpython-core2\docs\visual\*.pdf"; DestDir: "{app}\Lib\site-packages\visual\docs\visual\"; Components: Documentation
Source: "c:\workspace\vpython-core2\docs\visual\*.gif"; DestDir: "{app}\Lib\site-packages\visual\docs\visual\"; Components: Documentation
Source: "c:\workspace\vpython-core2\docs\visual\*.css"; DestDir: "{app}\Lib\site-packages\visual\docs\visual\"; Components: Documentation
Source: "c:\workspace\vpython-core2\docs\visual\images\*.jpg"; DestDir: "{app}\Lib\site-packages\visual\docs\visual\images"; Components: Documentation

[Components]
Name: Visual; Description: "The Visual extension module for Python"; Types: full compact custom; Flags: fixed
Name: numpy; Description: "numpy 1.5.1 {code:NumpyStatus|C:\Python31}"; Types: full; Check: CheckNumpy( 'C:\Python31' )

; FontTools, ttfquery, and Polygon are needed by the 3D text object
Name: FontTools; Description: "FontTools 2.3 {code:FontToolsStatus|C:\Python31}"; Types: full; Check: CheckFontTools( 'C:\Python31' )
Name: ttfquery; Description: "ttfquery 1.0.4 {code:ttfqueryStatus|C:\Python31}"; Types: full; Check: Checkttfquery( 'C:\Python31' )
Name: Polygon; Description: "Polygon 3.0a1 {code:PolygonStatus|C:\Python31}"; Types: full; Check: CheckPolygon( 'C:\Python31' )

Name: Documentation; Description: "Documentation for the Visual extension to Python"; Types: full
Name: Examples; Description: "Example programs"; Types: full
Name: VIDLE; Description: "VIDLE: improved version of the IDLE program editor"; Types: full custom

[Tasks]
Name: desktopicon; Description: "Create a desktop icon to start VIDLE"

[Icons]
Name: "{commondesktop}\VIDLE for VPython"; Filename: "{app}\pythonw.exe"; Parameters: "{app}\Lib\site-packages\vidle\idle.pyw"; WorkingDir: "{app}\Lib\site-packages\visual\examples"; IconFilename: "{app}\DLLs\py.ico"; Tasks: desktopicon
Name: "{commonstartmenu}\VIDLE for VPython"; Filename: "{app}\pythonw.exe"; Parameters: "{app}\Lib\site-packages\vidle\idle.pyw"; WorkingDir: "{app}\Lib\site-packages\visual\examples"; IconFilename: "{app}\DLLs\py.ico"
; commonstartmenu puts a choice on the "All Programs" list.
; commonstartup puts a choice inside "All Programs/Startup"
; commonfavorites puts a choice in Internet Explorer!


; This code file contains a ShouldSkipPage function which looks
; for an appropriate version of python.exe,
; and if it is found we skip the "select a directory" page.

[Code]
program Setup;

// Try to discover where Python is actually installed.
function MyConst(Param: String): String;
var Exist1, Exist2: Boolean;
begin
    Exist1 := FileExists( ExpandConstant('{reg:HKLM\Software\Python\PythonCore\3.1\InstallPath,}\python.exe'));
    if Exist1 then
      Result := ExpandConstant('{reg:HKLM\Software\Python\PythonCore\3.1\InstallPath,}')
    else
      begin
      Exist2 := FileExists( ExpandConstant('{reg:HKCU\Software\Python\PythonCore\3.1\InstallPath,}\python.exe'));
      if Exist2 then
        Result := ExpandConstant('{reg:HKCU\Software\Python\PythonCore\3.1\InstallPath,}')
      else
        Result := 'C:\'
      end
end;

function ShouldSkipPage(CurPage: Integer): Boolean;
var Result1, Result2: Boolean;
begin
  case CurPage of
    wpSelectDir:
      begin
      Result1 := FileExists( ExpandConstant('{reg:HKLM\Software\Python\PythonCore\3.1\InstallPath,}\python.exe'));
      Result2 := FileExists( ExpandConstant('{reg:HKCU\Software\Python\PythonCore\3.1\InstallPath,}\python.exe'));
      Result := Result1 or Result2
      if not Result then
         MsgBox('Could not locate where Python 3.1 is installed.' #13 'You will be asked where python.exe is located.', mbInformation, MB_OK);
      end
    else
      Result := False;
  end;
end;

// Need a function to determine if numpy is already installed.
function NumpyAvailable( BasePath: String): Boolean;
begin
  Result := DirExists( BasePath + '\Lib\site-packages\numpy');
end;

// Choose a modifying string for the user-visible numpy component.
function NumpyStatus( Param: String): String;
begin
  if NumpyAvailable( Param) then
    Result := '(found)'
  else
    Result := '(Numpy must be selected)'
end;

// Don't clobber an existing installation of numpy
function CheckNumpy( Param: String): Boolean;
begin
  Result := not NumpyAvailable( Param);
end;

//-------------------------------

// FontTools, ttfquery, and Polygon are needed by the 3D text object

// Need a function to determine if FontTools is already installed.
function FontToolsAvailable( BasePath: String): Boolean;
begin
  Result := DirExists( BasePath + '\Lib\site-packages\FontTools');
end;

// Choose a modifying string for the user-visible FontTools component.
function FontToolsStatus( Param: String): String;
begin
  if FontToolsAvailable( Param) then
    Result := '(found)'
  else
    Result := '(FontTools must be selected)'
end;

// Don't clobber an existing installation of FontTools
function CheckFontTools( Param: String): Boolean;
begin
  Result := not FontToolsAvailable( Param);
end;

//-------------------------------

// Need a function to determine if ttfquery is already installed.
function ttfqueryAvailable( BasePath: String): Boolean;
begin
  Result := DirExists( BasePath + '\Lib\site-packages\ttfquery');
end;

// Choose a modifying string for the user-visible ttfquery component.
function ttfqueryStatus( Param: String): String;
begin
  if ttfqueryAvailable( Param) then
    Result := '(found)'
  else
    Result := '(ttfquery must be selected)'
end;

// Don't clobber an existing installation of ttfquery
function Checkttfquery( Param: String): Boolean;
begin
  Result := not ttfqueryAvailable( Param);
end;

//-------------------------------

// Need a function to determine if Polygon is already installed.
function PolygonAvailable( BasePath: String): Boolean;
begin
  Result := DirExists( BasePath + '\Lib\site-packages\Polygon');
end;

// Choose a modifying string for the user-visible Polygon component.
function PolygonStatus( Param: String): String;
begin
  if PolygonAvailable( Param) then
    Result := '(found)'
  else
    Result := '(Polygon must be selected)'
end;

// Don't clobber an existing installation of Polygon
function CheckPolygon( Param: String): Boolean;
begin
  Result := not PolygonAvailable( Param);
end;
