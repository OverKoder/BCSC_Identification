GENE_SIGNATURES = {
    'BC_10_1016_N': ['CDH1','CLDN12','CLDN3','CLDN4','CLDN7','CLDN8','CTNNB1','DSP', 'NIFK',  'MMP9','OCLN',  'VIM','ZEB1','ZEB2'],
    'BC_10_1016_P': ['CDH1','CLDN12','CLDN3','CLDN4','CLDN7','CLDN8','CTNNB1','DSP', 'NIFK',  'MMP9','OCLN', 'VIM','ZEB1','ZEB2'],
    'CC_10_1016_N': ['CDH1','CLDN1','CLDN4','CLDN7','CLDN8','FN1','FOXC2','MKI67',  'OCLN', 'SNAI2','VIM','ZEB1'],
    'CC_10_1016_P': ['CDH1','CLDN1','CLDN4','CLDN7','CLDN8','FN1','FOXC2','MKI67',  'OCLN','SNAI2','VIM','ZEB1'],
    'BT_10_1016_N': ['CDH1','CLDN11','CLDN18','CLDN2','CLDN3','CLDN4','CLDN7','FN1', 'MKI67','NIFK',  'MMP2','MMP9','OCLN',  'TWIST1','VIM','ZEB1'],
    'BT_10_1016_P': ['CDH1','CLDN11','CLDN18','CLDN2','CLDN3','CLDN4','CLDN7','MKI67','NIFK', 'MMP2','MMP9','OCLN', 'TWIST1','VIM','ZEB1'],
    'BCSC_TC_10_1038': ['ANGPT2','ARHGAP23','ARHGEF17','ATP2B4','BTNL9','DAB2IP','EDNRA','ENG','FLT1','FMNL3','GJA4','HBA1','HBA2','ITGA7','JAG2','LGI4','CCDC80','MALL','NDUFA4L2','NLRC5','NOS3','NOTCH3','PDGFA','PODXL','PTP4A3','RASGRP3','ROBO4','RYR2','SHROOM4','SNTB1','STAB1','SYNPO','TBX2','TINAGL1','UTRN','VASH1'],
    'BT_SC_10_1038': ['ALDH3A2','ALDH1A3','ALDH2','CD24A','ITGB1','ITGB6','EPCAM','CD14','KIT','PROM1','LY6A','CD44A','MME','ITGB3','SOX9','NECTIN4','FLOT2','VEGTA']
}

TRUE_VALUES = {
    'BC_10_1016_N': [-4.655, -2.03, -2.435, -6.34, -2.05, -1.43, 1.67, -7.92, -1.36, 1.5, -2.75, 2.56, 3, 3.14],
    'BC_10_1016_P': [1.57, 2.4, 3.225, 2.51, 2.13, 1.86, -1.58, 1.69, 1.5, -2.44, 1.8, -3.575, -2.815, -2.363],
    'CC_10_1016_N': [-1.94, -1.5, -2.48, -2.01, -6.38, 3.286, 1.29, -1.29, -2.3825, 1.39, 2.12, 1.93],
    'CC_10_1016_P': [1.62, 1.71, 2.635, 1.73, 7.01, -1.838, -1.23, 1.21, 2.4325, -1.78, 1.65, -2.365],
    'BT_10_1016_N': [-1.53, -1.4, -1.58, -1.25, -2.41, -1.64, -1.51, 3.5675, -1.32, -1.22, 1.31, 1.23, -1.19, 1.64, 4.33, 1.4],
    'BT_10_1016_P': [1.87, 1.53, 1.27, 1.16, 1.68, 1.46, 1.24, 1.19, 1.22, -2.185, -1.2, 1.7, -5.09, -1.41, -2.26],
    'BCSC_TC_10_1038': [-4.483, -1.896, -1.84, -1.148,-5.272, -2.021, -3.15, -2.807, -3.566, -2.452, -5.833, -7.448, 
    -7.696, -4.4, -2.9, -4.07, 2.417, -1.647, -5.38, -2.135, -3.513, -2.792, -2.715, -2.369, -2.817, 
    -4.135, -4.395, -2.591, -3.084, -3.368, -3.475, -3.109,-3.493, -2.7, -1.652, -2.08]
}

# Temporal list for data checking and preprocessing
ALL_GENES = ['ANGPT2','ARHGAP23','ARHGEF17','ATP2B4','BTNL9','CD2','CD3E','CD3G','CDH1','CLDN1','CLDN11','CLDN12','CLDN18',
    'CLDN2','CLDN3','CLDN4','CLDN7','CLDN8','CTNNB1','DAB2IP','DSP','EDNRA','ENG','FCGR1A','FCGR3A','FLT1','FMNL3',
    'FN1','FOXC2','GJA4','HBA1','HBA2','HERC2P2','ITGA7','ITGB2','ITGB4','JAG2','LGI4','LINC01279','MALL','MKI67',
    'MKI67IP','MME','MMP2','MMP9','MTMR9LP','NDUFA4L2','NLRC5','NOS3','NOTCH3','OCLN','PECAM1','PDGFA','PODXL',
    'PTP4A3','RASGRP3','ROBO4','RYR2','SHROOM4','SNAI2','SNTB1','STAB1','SYNPO','TBX2','TINAGL1','TWIST1','UTRN',
    'VASH1','VIM','ZEB1','ZEB2']

# Temporal list for data checking and preprocessing
GENES = ['ANGPT2','ARHGAP23','ARHGEF17','ATP2B4','BTNL9','CDH1','CLDN1','CLDN11','CLDN12','CLDN18','CLDN2','CLDN3','CLDN4','CLDN7',
'CLDN8','CTNNB1','DAB2IP','DSP','EDNRA','ENG','FLT1','FMNL3','FN1','FOXC2','GJA4','HBA1','HBA2','HERC2P2','ITGA7','JAG2','LGI4','LINC01279','MALL','MKI67',
'MKI67IP','MMP2','MMP9','MTMR9LP','NDUFA4L2','NLRC5','NOS3','NOTCH3','OCLN','PDGFA','PODXL','PTP4A3','RASGRP3','ROBO4','RYR2','SHROOM4',
'SNAI2','SNTB1','STAB1','SYNPO','TBX2','TINAGL1','TWIST1','UTRN','VASH1','VIM','ZEB1','ZEB2']

# tDR-000620, ESA
# Temporal list for data checking and preprocessing
WILEY_TABLE = [
    ['ABCG2'],
    ['ALDH1A1','CD44'],
    ['ALDH1A1'],
    ['CEBPD'],
    ['CCR5'],
    ['CD44','CD24','PECAM1', 'MME', 'ITGB4', 'CD3E', 'CD3G', 'FCGR1A', 'FCGR3A', 'ITGB2', 'CD2'],
    ['CD44','CD24','EPCAM'],
    ['CD44','ITGA6','PROM1'],
    ['CD44','CD24','B3GALT5'],
    ['PROCR','B3GALT5'],
    ['ITGA6','DLL1','DNER'],
    ['ITGA6','ITGB3'],
    ['ITGAV'],
    ['CD70'],
    ['GJB2'],
    ['CXCR2'],
    ['B4GALNT1'],
    ['GLO1'],
    ['EPAS1'],
    ['LGR5'],
    ['MUC1'],
    ['NECTIN4'],
    ['PROCR'],
    ['RUNX1'],
    ['SDC1']
]

# Temporal list for data checking and preprocessing
WILEY_SIGN = [
    ['+'],
    ['+HI','+'],
    ['+'],
    ['+'],
    ['+'],
    ['+'],
    ['+','-','-','-','-','-','-','-','-','-','-'],
    ['+','-LO','+'],
    ['+','+HI','+HI'],
    ['+','+LO','+'],
    ['+','+HI','+HI'],
    ['+','+'],
    ['+'],
    ['+'],
    ['+'],
    ['+'],
    ['+'],
    ['+'],
    ['+'],
    ['+HI'],
    ['+'],
    ['+'],
    ['+'],
    ['+'],
    ['+'],
]

# Temporal list for data checking and preprocessing
WILEY_TARGET = [
    ['EPCAM','ITGA6','CD24','MME','PROM1'],
    ['GSTP1','ABCB1','CHEK1','KRT8','KRT18','KRT19'],
    [],
    ['IL6', 'HIF1A', 'CD44', 'CDH2','VIM', 'CDH1', 'TWIST1', 'STAT3','MYC', 'NANOG', 'KLF4', 'POU5F1', 'SOX2', 'FBXW7', 'NOTCH1'],
    ['FANCB', 'LIG3', 'POLE', 'CRY1', 'PIK3CA'],
    [],
    [],
    ['SOX2','BMI1', 'NANOG'],

    #globo-H
    ['B3GALT5','CASP3','CASP8','CASP9','FUT4'],
    ['SERPINB5','TOP2A','KRT5','TP63','SOX4','CD24','ADRM1','DNER','DLL1','JAG1'],
    ['ABCG2','ALDH1A1','PROM1', 'GLI1','TP63','KRT5','KRT6A','KRT14','KRT18', 'TGFB1','SERPINE1', 'IL6', 'IGFBP3', 'FOXC2', 'CDH2', 'SMN1', 'SNAI1', 'TWIST1', 'ZEB1'],
    ['SNAI2'],
    ['CDH1','VIM'],
    ['NANOG','PTK2','POU5F1','SOX2'],
    ['ALDH1A1','ABCG2','NOTCH1','SOX2','NANOG'],
    ['ST8SIA1','MMP2','MMP7','MMP19','CDH2','VIM', 'CDH1', 'CD44', 'CD24'],
    ['ALDH1A1'],
    ['MYC','POU5F1','NANOG','HEY2','CTNNB1','AXIN2','BIRC5'],
    ['CDH1','CTNNB1','VIM','FN1','SNAI1','SNAI2','CCND1','MYC','KRT14','KRT18','CD44','CD24'],
    ['ABCG2','KRT18','KRT19','EPCAM','ITGA6'],
    ['CD44','PROM1','PIK3CA','AKT1','CTNNB1','CDH1','VIM'],
    ['VIM','CDH1','SNAI2','FOXC2','ALDH1A1','CD44','CD24','PROM1','CXCR4','ABCG2'],
    ['CDH1','VIM','FN1','VEGFA','MMP13','MMP9','CXCR4','CXCL12','ZEB1','TWIST1','CD24','CD44'],
    ['POU5F1','CD44', 'ITGB1', 'CD24', 'KRT14','KRT18','KRT19','ACTA2'],
    ['CD44','CD24','ALDH1A1','NOTCH1','NOTCH3','NOTCH4','HEY1','GLI1','IL6','CXCL8','IL6ST','STAT3','NFKB1','CCL20','EGFR']
]