# CrNb4Se8_LM_optimization
CrNb4Se8をLM法で構造最適化するプログラム
## How to use 
python CrNb4Se8_LM.py (input file name)  
input file nameを設定しない場合、Fobs_Nuc.txtとして扱われます。
## Output files
- Fcal_Fobs_Nuc.txt :
Fcalを計算し、スケールファクターsを最適化してFobsとfitさせた結果
- result_for_optimization.txt :
LM法による構造最適化の結果
- Fobs_vs_Fcal_for_optimization.png :
最適化後のFcal-Fobs plot
- PONTA_plot_template_for_optimization.gp :
Fcal-Fobs plotを出力するためのgnuplot script
