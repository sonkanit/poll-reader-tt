rd /s/q tmp
md tmp
rd /s/q results
md results
FOR %%i IN ("*.pdf") DO (
	echo Extract images from %%i
	echo ..\..\3rdParties\ImageMagick\convert.exe -density 200 "%%i" ".\tmp\%%i-%%03d.jpg"
	..\..\3rdParties\ImageMagick\convert.exe -density 200 "%%i" ".\tmp\%%i-%%03d.jpg"
	FOR %%j IN (".\tmp\%%i*.jpg") DO ..\..\Release\PollReader.exe "%%j" null ".\results\%%i.csv"
)