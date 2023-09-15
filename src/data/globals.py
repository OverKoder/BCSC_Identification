ID1_2_LABEL = {
    'Endothelial': 0,
    'NA': 1,
    'Fibroblasts': 2,
    'Epithelial': 3,
    'B_cells': 4,
    'T_cells': 5,
    'Mf': 6,
    'DC': 7,
    'Mast_cells': 8
}

LABEL_2_ID1 = {
    0: 'Endothelial',
    1: 'NA',
    2: 'Fibroblasts',
    3: 'Epithelial',
    4: 'B_cells',
    5: 'T_cells',
    6: 'Mf',
    7: 'DC',
    8: 'Mast_cells'
}

ID2_2_LABEL = {
    'Venous': 0,
    'EC': 1,
    'NA': 2,
    'Tip_cells': 3,
    'Capillary': 4,
    'Arterial': 5,
    'iCAF': 6,
    'myCAF': 7,
    'dPVL': 8,
    'imPVL': 9,
    '31': 10,
    'Lymph': 11,
    'Plasma': 12,
    'Bcell': 13,
    'CD8': 14,
    'CD4': 15,
    'Epithelial': 16,
    'NK': 17,
    'T_prolif': 18,
    'Mf': 19,
    'Mf_CCR2': 20,
    'Mf_CXCR3': 21,
    'monocyte': 22,
    'DC': 23,
    'F_prolif': 24,
    '39': 25,
    '40':26
}

LABEL_2_ID2 = {
    0: 'Venous',
    1: 'EC',
    2: 'NA',
    3: 'Tip_cells',
    4: 'Capillary',
    5: 'Arterial',
    6: 'iCAF',
    7: 'myCAF',
    8: 'dPVL',
    9: 'imPVL',
    10: '31',
    11: 'Lymph',
    12: 'Plasma',
    13: 'Bcell',
    14: 'CD8',
    15: 'CD4',
    16: 'Epithelial',
    17:'NK',
    18:'T_prolif',
    19:'Mf',
    20: 'Mf_CCR2',
    21: 'Mf_CXCR3',
    22: 'monocyte',
    23: 'DC',
    24: 'F_prolif',
    25: '39',
    26: '40'
}


ID3_2_LABEL = {
    'Venous': 0,
    'EC': 1,
    'NA': 2,
    'Tip_cells': 3,
    'Capillary': 4,
    'Arterial': 5,
    'iCAF': 6,
    'myCAF': 7,
    'dPVL': 8,
    'imPVL': 9,
    '31': 10,
    'Lymph': 11,
    'Plasma': 12,
    'Bcell': 13,
    'CD8': 14,
    'CD4': 15,
    'Epithelial': 16,
    'NK': 17,
    'T_prolif': 18,
    'Mf': 19,
    'Mf_CCR2': 20,
    'Mf_CXCR3': 21,
    'monocyte': 22,
    'DC': 23,
    'F_prolif': 24,
    'Mast_cells': 25,
    '40':26
}

LABEL_2_ID3 = {
    0: 'Venous',
    1: 'EC',
    2: 'NA',
    3: 'Tip_cells',
    4: 'Capillary',
    5: 'Arterial',
    6: 'iCAF',
    7: 'myCAF',
    8: 'dPVL',
    9: 'imPVL',
    10: '31',
    11: 'Lymph',
    12: 'Plasma',
    13: 'Bcell',
    14: 'CD8',
    15: 'CD4',
    16: 'Epithelial',
    17:'NK',
    18:'T_prolif',
    19:'Mf',
    20: 'Mf_CCR2',
    21: 'Mf_CXCR3',
    22: 'monocyte',
    23: 'DC',
    24: 'F_prolif',
    25: 'Mast_cells',
    26: '40'
}

GENE_SIGNATURES = {
    0: ['CTNNB1' ,'MMP9' ,'VIM' ,'ZEB1' ,'ZEB2'],
    1: ['CDH1','CLDN12','CLDN3','CLDN4','CLDN7','CLDN8','DSP','MKI67','OCLN'],
    2: ['FN1','FOXC2','SNAI2','VIM','ZEB1'],
    3: ['CDH1','CLDN1','CLDN4','CLDN7','CLDN8','MKI67','OCLN','VIM'],
    4: ['FN1','MMP2','MMP9','TWIST1','VIM','ZEB1'],
    5: ['CDH1','CLDN11','CLDN18','CLDN2','CLDN3','CLDN4','CLDN7','MKI67','OCLN'],
    6: ['ANGPT2', 'ARHGAP23', 'ARHGEF17', 'ATP2B4', 'BTNL9', 'CCDC80', 'DAB2IP', 'EDNRA', 'ENG', 'FLT1', 'FMNL3', 'GJA4', 'HBA1', 'HBA2', 'HERC2', 'ITGA7', 'JAG2', 'LGI4', 'MALL', 'NDUFA4L2', 'NLRC5', 'NOS3', 'NOTCH3', 'PDGFA', 'PODXL', 'PTP4A3', 'RASGRP3', 'ROBO4', 'RYR2', 'SHROOM4', 'SNTB1', 'STAB1', 'SYNPO', 'TBX2', 'TINAGL1', 'UTRN', 'VASH1']
}
#MTMR9LP in 6 missing

ALL_GENES = ['ANGPT2','ARHGAP23','ARHGEF17','ATP2B4','BTNL9','CD2','CD3E','CD3G','CDH1','CLDN1','CLDN11','CLDN12','CLDN18',
    'CLDN2','CLDN3','CLDN4','CLDN7','CLDN8','CTNNB1','DAB2IP','DSP','EDNRA','ENG','FCGR1A','FCGR3A','FLT1','FMNL3',
    'FN1','FOXC2','GJA4','HBA1','HBA2','HERC2P2','ITGA7','ITGB2','ITGB4','JAG2','LGI4','LINC01279','MALL','MKI67',
    'MKI67IP','MME','MMP2','MMP9','MTMR9LP','NDUFA4L2','NLRC5','NOS3','NOTCH3','OCLN','PECAM1','PDGFA','PODXL',
    'PTP4A3','RASGRP3','ROBO4','RYR2','SHROOM4','SNAI2','SNTB1','STAB1','SYNPO','TBX2','TINAGL1','TWIST1','UTRN',
    'VASH1','VIM','ZEB1','ZEB2']