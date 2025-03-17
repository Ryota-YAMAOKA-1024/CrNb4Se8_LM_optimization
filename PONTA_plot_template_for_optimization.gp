datafile1='./result_for_optimization.txt'

set title "nuclear scattering at HHL+H0L (optimized)" font "Arial,18"

# for png
set term png
set out './Fobs_vs_Fcal_for_optimization.png'

set style line 1 lt 1 lc "#ff0000" lw 1 pt 7 ps 1.5
set style line 2 lt 1 lc "#0000FF" lw 1 pt 7 ps 1.5
set style line 11 lt 1 lc "#000000" lw 1 pt 6 ps 1.5

set size ratio 1  

set xlabel '|F_{cal}|' font "Arial,18"
set ylabel '|F_{obs}|' font "Arial,18"

set xrange[0:200]
set yrange[0:200]

plot datafile1 u 4:5:6 with yer pt 7 ps 1.5 lc "red" notitle ,x notitle
